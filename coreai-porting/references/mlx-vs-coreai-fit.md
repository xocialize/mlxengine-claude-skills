# CoreAI vs MLX vs both — the fit journal

**A JOURNAL, not a decision tree.** Anything marked ASSUMED is a hypothesis awaiting a port and
must not close a decision without saying so. Every port adds a row.

---

## First: the common framing is wrong

The usual advice is **"CoreAI for production, MLX for research."** Four measured ports say that
is not a useful axis — it implies the two are the same capability at different maturity.

| | CPU | GPU | **ANE** |
|---|---|---|---|
| MLX | yes | yes (Metal) | **no** |
| CoreAI | yes | yes (MPSGraph) | **yes** |

**The Neural Engine is the only thing either framework can do that the other cannot.** Everything
else is engineering ergonomics, not capability. So the question decomposes:

1. **Is this graph ANE-resident, and does that matter for this workload?** — a capability
   question, measurable *before* committing to a port.
2. **If not** — static/AOT (CoreAI) or dynamic/JIT (MLX)? — ergonomics, and far less
   consequential.

### Two measured facts that make the common framing actively misleading

**(a) The ANE is not "faster."** MEASURED (Real-ESRGAN SRVGG, fp16, 128²) — the ANE lane is
**4× SLOWER** than CoreAI's own GPU lane:

| | CoreAI-GPU | CoreAI-ANE |
|---|---|---|
| latency | **1.40 ms** | 5.65 ms |
| energy / inference | 131.4 mJ | **67.6 mJ** (1.94×) |
| clock under sustained load | 1620 → **1568 MHz, sagging** | GPU held at **338 MHz idle** |
| host memory, 1080p ×4 | — | **19 MB resident / 0.86 GB peak** vs the MLX sibling's 21.24 GB |
| fp16 parity | **72.62 dB** | 68.60 dB |

Reaching for CoreAI *for speed* is usually wrong. Reach for the ANE when the workload is
**energy-, thermal-, or host-memory-bound** — and accept latency and ~4 dB as the price.

**(a2) A CoreAI process currently dies after ~8,000 inferences.** MEASURED: an uncatchable
Swift precondition on IOSurface output storage, on **every** compute unit, not fixed by releasing
outputs or reloading the model (`apple/coreai-torch#75`). MLX has no equivalent ceiling. **For
any sustained high-rate workload this outranks every energy argument above** — an efficiency win
is worth nothing if the process cannot stay alive. → `runtime-limits.md`.

**(b) Choosing CoreAI does not mean getting the ANE.** MEASURED (deformable conv, decoder
scale): the "ANE" lane ran **5.5× slower than simply using the GPU** (39 vs 216 inf/s) because
`gather` was rejected and the graph partitioned. **A CoreAI port that silently misses the ANE can
be worse than the MLX path it replaces**, and nothing raises.

---

## The question that actually decides it

Answerable in about a minute with `scripts/ane_preflight.py` — export, load on the ANE lane,
read what it rejected.

### Rejections we have MEASURED

| Op | Where it comes from |
|---|---|
| `gather` | scattered reads — deformable conv, grid sample, learned-offset sampling |
| `bitwise_and` | bool mask chains (also **silently miscompiles on CPU**, `coreai-torch#74`) |
| `slice_scatter` | in-place writes into a slice, `mask[:, :n, m:] = ...` |
| `scaled_dot_product_attention` | **sometimes** — "preserved as a composite" ≠ ANE-eligible |
| `einops.rearrange` in window attention | produces **rank-6** tensors → hard ANE reject. Grep for it before exporting |
| rank > 5 tensors | multi-index einsums decompose through rank-6 reshapes |

**Two traps in reading that table:**

1. **Eligibility is SCALE-DEPENDENT.** MEASURED: a formulation with **0** rejections at 16² had
   **36** at 128². Never clear a graph for the ANE at toy resolution.
2. **It is not a static allowlist.** The same op is accepted in one graph and rejected in
   another — one SDPA instance rejected while its siblings passed. Run the pre-flight.

### The shape of the answer

```text
Will this process do more than ~8,000 inferences without restarting?
  yes -> MLX, or accept process recycling (coreai-torch#75). This gate comes FIRST.
  no  -> continue:

Does the workload care about energy, heat, or host memory more than latency?
  no  -> the ANE is not the reason to be here. Pick on ergonomics:
         fixed geometry + AOT -> CoreAI;  runtime shapes, fast iteration,
         mature quant, weight streaming -> MLX.
  yes -> run ane_preflight:
           clean       -> CoreAI/ANE is a real, differentiated win
           partitions  -> STOP. Measured slower than the plain GPU lane.
                          Eliminate the rejected op, or use MLX.
```

The middle branch is the one nobody writes about, and it is the common case.

Note what dropped out: "is it a convnet", "is it attention-heavy", "production vs research".
Those are proxies. **Op-level ANE eligibility is the thing itself**, and it is cheap to measure.

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
| **Autoregressive / KV-cache models** | MLX is the better fit; CoreAI `state_names` unproven for us | **Upstream now ships LLM recipes** (`phi`, `smollm2`, `gemma3`, `mistral`, `mixtral`, `gpt_oss`, `muse_glimmer`) **and a `CoreaiStatefulExporter` KV-cache reference** — the candidate gap is closed, the measurement is not |
| ~~Small convnets~~ | **PARTLY DECIDED.** ANE eligibility and speed: yes, universally. fp16 accuracy: **family-dependent** — BN+ReLU nets pass, SE-bearing nets fail the 50 dB gate. Cause not yet established (activation hypothesis tested and rejected) | MEASURED 2026-08-30, `case-wave2-convnets.md` |
| **Dev velocity** | MLX iterates far faster; CoreAI export/specialize cycle is slow | Time both loops on the same model and record it |
| **Quantization below fp16** | Unknown for CoreAI; MLX quant is well understood | Phase 3 compression sweep |
| **Multi-function / promptable models** | CoreAI's multi-function asset may beat MLX's re-encode | SAM family, Phase 3 |
| ~~Scattered-read models~~ | **DECIDED: not ANE-resident.** `gather` is rejected; the graph partitions and runs slower than the plain GPU lane | MEASURED 2026-08-29, `case-deformable-conv.md` |
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
| 2026-08-30 | SCUNet **rank-5 rewrite** | window attention | **still CoreAI-GPU / MLX — but eligibility is no longer the reason** | MEASURED: rank 6→5 took ANE rejections 678→0 with residency PROVEN (338 MHz idle clock) and ANE 8.7× faster; yet still 3.6× slower than the GPU lane, parity 39.12 dB, energy a wash. **Eligibility and suitability are independent.** |
| 2026-08-30 | SCUNet (17.9M, Swin-Conv UNet) baseline | window attention | **CoreAI-GPU or MLX — NOT the ANE** | MEASURED: converts fine (peak RSS 1.05 GB), but rank-6 from `einops.rearrange` → 678 ANE rejections → **ANE 30× SLOWER than the GPU lane** (249.3 vs 8.2 ms) and 37.78 vs 71.22 dB. → `case-scunet-window-attention.md` |
| 2026-08-30 | DRUNet (32.6M, DPIR restoration) | plain conv+ReLU UNet, no BN | **either — CoreAI viable** | MEASURED: 0 ANE rejections, fp16 ANE 64.12 dB vs golden (GPU 75.34), but ANE 3.198 ms vs GPU 2.642 — the ANE's speed edge disappears at this size |
| 2026-08-30 | Wave 2 convnets (7 families) | plain convnets | **CoreAI/ANE for BN+ReLU nets; check per model** | MEASURED: all ANE-eligible (0 rejections), 1.3–5.7× faster than the GPU lane, but fp16 parity splits — `resnet*`/`mobilenetv4` pass (−3…+5 dB vs GPU), `efficientnet*`/`mobilenetv3`/`convnext` fail (−10…−30 dB). → `case-wave2-convnets.md` |
| 2026-08-30 | EoMT (encoder-only mask transformer) | ViT segmentation | **CoreAI-GPU** | MEASURED: capture fixed, fp32 134 dB; fp16 ANE 43.58 dB (below the 50 dB gate) and `slice_scatter`/SDPA rejected → partitions |
| 2026-08-29 | Deformable conv (DCNv2) — BiRefNet's blocker | scattered-read decoder | **MLX / GPU, not ANE** | MEASURED: converts exactly (3.6e-07 vs torchvision) but `gather` is ANE-rejected at scale; "ANE" lane 5.5× slower than the GPU lane |
| 2026-08-01 | Moebius UNet (226M) | latent-diffusion, λ-attention | **MLX** — CoreAI/ANE blocked | MEASURED: rank-6 reject, then compositional ANECCompiler bug (upstream #138). GPU lane still gained 4.1× from the rewrite |

*Add one row per port. A row with no measurement is not a row.*

---

## The single best available A/B — Wan 2.1

**`apple/coreai-models` now ships an official Wan 2.1 T2V 1.3B recipe** (added since our
2026-07-31 snapshot), with `--compression 4bit-asym | 8bit` and a `videodiffusion-runner`.

This is the **cleanest routing experiment available to us**: we have deep, measured MLX expertise
on exactly this family (`wan-video`, the `wan-core` Swift substrate, `BlockStreamer`, the
decode-memory lever ladder). Running the official CoreAI recipe against our own MLX numbers on
the *same model, same machine* would produce a decisive row for video diffusion — the one class
where this table is empty.

Cost: multi-GB download, long export. Value: the highest-signal row this journal could gain.
It is a **routing** experiment, distinct from a porting-craft exercise.

## What would make this skill actually good

The honest gap: **two ports is not a fit model.** To route confidently we need coverage across
the axes above — at minimum a small convnet, an attention model that *succeeds* on ANE, a
multi-function asset, and one stateful model. Until then this file's job is to be explicit
about what it does not know.
