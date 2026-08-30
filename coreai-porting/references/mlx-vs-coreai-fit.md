# CoreAI vs MLX vs both — the fit journal

**This is a JOURNAL, not a decision tree.** It exists to answer "given this model, which runtime
does it belong on?" — and as of 2026-08-29 we have **two data points**. Everything marked
ASSUMED below is a hypothesis awaiting a port, and must not be used as a basis for a decision
without saying so out loud.

**Every port adds a row.** That is the point of the collection.

---

## What "both" means

Not a hedge. MEASURED on Real-ESRGAN: the MLX and CoreAI siblings serve **different tiers of
the same product** — CoreAI/ANE for the low-energy, memory-constrained, fixed-tile path; MLX/GPU
for the flexible-geometry, whole-frame path. Shipping both and routing at runtime is a real
answer, and on that model it was the right one.

The corollary is a documentation duty: **the two ports have inverted geometry contracts** (see
below), and each package must say so, or a future maintainer will "fix" one toward the other.

---

## Decided axes (MEASURED)

### 1. Energy — CoreAI/ANE wins, but the size of the win depends on the opponent
**MEASURED (SRVGG, M5 Max, t128):** ANE ties well-tuned MLX-GPU on wall clock while drawing
**≈4.5–4.9× less energy per frame** (~17 W vs ~83 W over idle), and does not thermally throttle.

**REFINED 2026-08-29 — name the opponent.** That 4.7× is against **MLX-GPU fp32**. Measured
against **CoreAI-GPU** (fp16, static — a far stronger opponent, which the same receipt records
beating MLX-GPU by 2.2–2.4×), the ANE advantage is **1.94×**:

| lane | W over idle | inf/s | mJ/inference |
|---|---|---|---|
| CoreAI-GPU | 106.2 | 808.4 | 131.4 |
| CoreAI-ANE | **12.4** | 183.4 | **67.6** |

The consistency check works: 4.7 ÷ ~2.3 ≈ 2.0. **Always state which GPU path the ANE is being
compared against** — "4.7× less energy" and "1.94× less energy" are the same hardware, different
baselines, and quoting the wrong one oversells by 2.4×.

Thermal behaviour still favours the ANE unambiguously: MEASURED GPU sag **1620 → 1568 MHz within
10 s** of sustained load; the ANE lane showed none, and held the GPU at its **338 MHz idle
clock** throughout — which is also the residency oracle.

→ If the workload is sustained, battery-bound, or thermally constrained, this axis alone can
decide it.

### 2. Host memory — CoreAI/ANE wins by orders of magnitude
**MEASURED (SRVGG, 1080p→×4):** ANE activations stay on-die, so process footprint is host-side
accumulation buffers only — **19 MB resident / 0.86 GB peak**, vs the MLX sibling's **21.24 GB**
whole-frame.

→ If the host is memory-governed, this is not a tiebreak, it is the answer.

### 3. Geometry — the contracts are INVERTED
**MEASURED.** CoreAI tile/canvas geometry is a **build-time property of the asset**: one
executable per static shape. MLX geometry is **runtime-injectable**.

→ A model whose input geometry genuinely varies at runtime costs CoreAI one asset per shape.
Count the shapes before committing.

### 4. Attention-heavy graphs — CoreAI needs rewriting, and it may still be blocked
**MEASURED (Moebius, 226M latent-diffusion UNet):** rank-6 reshapes from multi-index einsums are
hard-rejected (rank ≤ 5), and a **compositional ANECCompiler bug** blocked the graph even after
the rewrite. Filed as `apple/coreai-models#138`; still open.

→ Worth noting the consolation prize is real: the ANE-motivated rewrite made the **GPU** lane
4.1× faster. A CoreAI attempt that fails to reach the ANE can still pay for itself.

### 5. First-load cost scales hard with graph size
**MEASURED:** E5RT specialization ~8 s at 1.4M params, **254 s** at 226M. OS-cached after.

→ For a large model in an interactive app, this is a product decision, not just a number.

---

## ASSUMED axes — hypotheses, not findings

Listed so they can be closed by measurement rather than argued about.

| Axis | Hypothesis | How to close it |
|---|---|---|
| **Autoregressive / KV-cache models** | MLX is the better fit; CoreAI `state_names` exists but is unproven for us | Port one small stateful model. **No LLM in the LibreYOLO pool** — needs a separate candidate |
| **Small convnets** | CoreAI/ANE should dominate on every axis | Phase 1 classifiers (`resnet`, `mobilenetv4`) |
| **Dev velocity** | MLX iterates far faster; CoreAI export/specialize cycle is slow | Time both loops on the same model and record it |
| **Quantization below fp16** | Unknown for CoreAI; MLX quant is well understood | Phase 3 compression sweep |
| **Multi-function / promptable models** | CoreAI's multi-function asset may beat MLX's re-encode | SAM family, Phase 3 |
| **Very large models (>1B)** | E5RT cost and asset size may make CoreAI impractical | Untested; may stay untested |

---

## Provisional triage — USE WITH THE CAVEAT

Good enough to *order investigation*, **not** good enough to close a decision. Say which axes
are ASSUMED whenever you use it.

```text
Is the geometry fixed at build time?           no  → MLX leads; count CoreAI shapes first
Is the host memory- or energy-governed?        yes → CoreAI leads, strongly (MEASURED)
Is it attention/einsum-heavy?                  yes → expect ANE rewrite work; may stay blocked
Is it autoregressive with a KV cache?          yes → MLX (ASSUMED — unproven)
Is it a plain convnet at modest size?          yes → CoreAI (ASSUMED beyond SRVGG)
Does it serve two product tiers?               yes → BOTH is a real answer (MEASURED once)
```

---

## The journal

| Date | Model | Class | Verdict | Basis |
|---|---|---|---|---|
| 2026-07-31 | Real-ESRGAN SRVGG (1.4M) | plain dense convnet | **BOTH** — CoreAI/ANE fast tier, MLX flexible tier | MEASURED: parity 58–69 dB, energy 4.5–4.9×, memory 19 MB vs 21.24 GB |
| 2026-08-29 | Real-ESRGAN SRVGG re-port (calibration) | plain dense convnet | **BOTH, confirmed** | MEASURED: ANE 68.60 dB, 5.65 ms, 67.6 mJ/inf; CoreAI-GPU 72.62 dB, 1.40 ms, 131.4 mJ/inf. ANE **1.94×** more energy-efficient but **4× slower** and **4 dB less accurate** than CoreAI-GPU |
| 2026-08-01 | Moebius UNet (226M) | latent-diffusion, λ-attention | **MLX** — CoreAI/ANE blocked | MEASURED: rank-6 reject, then compositional ANECCompiler bug (upstream #138). GPU lane still gained 4.1× from the rewrite |

*Add one row per port. A row with no measurement is not a row.*

---

## What would make this skill actually good

The honest gap: **two ports is not a fit model.** To route confidently we need coverage across
the axes above — at minimum a small convnet, an attention model that *succeeds* on ANE, a
multi-function asset, and one stateful model. Until then this file's job is to be explicit
about what it does not know.
