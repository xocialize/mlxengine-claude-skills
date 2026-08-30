# Case study — unblocking `deform_conv2d` (the BiRefNet blocker)

A complete worked pass through the missing-lowering triage ladder, MEASURED end to end on
2026-08-29 (macOS 27.0 / 26A5421a, M5 Max, `coreai-torch==0.4.1`, torch 2.11.0).

**Outcome in one line: the op converts and runs, but deformable convolution is NOT
ANE-resident at production scale — the blocker is `gather`, not `deform_conv2d`.**

---

## 0. Isolate before you fight

BiRefNet is a 1024² matting model. Learning a new lowering API *while* fighting a large model is
bad isolation. Build a minimal modulated-DCN repro first (8 ch, 16², ~2.5 K params), solve it
there, then scale.

**But see §4 — the toy scale gave a FALSE GREEN LIGHT.** Isolate to learn; always re-verify
eligibility at production scale.

## 1. The raw failure

```
ValueError: unable to handle call function op:
            target: deform_conv2d.default, namespace: torchvision
```

`run_decompositions(get_decomp_table())` leaves it untouched — it is a `torchvision` op, not
`aten`, so no core decomposition exists. Ladder steps 1–3 are unavailable.

## 2. Approach A beat Approach B before B was written

The obvious move is a custom lowering (`register_torch_lowering`). The *better* move was to
decompose the op **in PyTorch** into ops the converter already supports — arithmetic, `gather`,
reshape, matmul — and swap it in before export.

**No custom lowering, no private `coreai._compiler.dialects` API, no version fragility.**
Verified exact against `torchvision.ops.deform_conv2d`: **max |Δ| = 3.6e-07** across
offset=0/random × mask present/absent.

Implementation: `coreai-collection/recipes/deform_conv2d/decompose.py`.

> Generalisable: before reaching for a custom lowering, ask whether the op can be **written out
> of the graph in PyTorch**. That path is checkable against the original op numerically, which a
> hand-written CoreAI-op lowering is not.

## 3. Keep rank ≤ 5 by construction, not by repair

The natural formulation carries `[N, C, kh, kw, Ho, Wo]` — **rank 6**, which the ANE hard-rejects.
Fold the kernel taps into one axis `T = kh*kw` up front:

- sampling coords `[N, T, Ho, Wo]` (rank 4)
- gathered values `[N, C, T, Ho, Wo]` (rank 5)
- im2col matmul `[N, C*T, Ho*Wo]` (rank 3)

Numerically identical, and rank-legal from the first draft.

## 4. The validity mask — three formulations, and only one is clean

Out-of-bounds taps need zero weight. Written naively that is a bool AND chain. **MEASURED, all
three algebraically identical (max |Δ| = 3.6e-07 vs torchvision):**

| formulation | CPU parity | ANE rejections (16²) | ANE rejections (128²) |
|---|---|---|---|
| `(iy>=0) & (iy<H) & (ix>=0) & (ix<W)` | **10.64 dB — WRONG** | 24 (`bitwise_and_*`, `lt_*`) | — |
| `(iy>=0).to(dt) * (iy<H).to(dt) * …` | exact | 16 | — |
| `clamp(iy+1,0,1) * clamp(H-iy,0,1) * …` | exact | **0** | **36 (`gather_*`)** |

Two separate findings:

1. **The bool chain silently miscompiles on the CPU delegate.** Same graph, correct on the GPU.
   Minimised to ~15 lines and filed as **`apple/coreai-torch#74`**. This also means **the CPU
   lane cannot be trusted as a parity reference**.
2. **Pure `clamp` arithmetic is exact for integer-valued coordinates** (these come from `floor()`
   and `floor()+1`) and avoids comparisons entirely. Prefer it.

> **The toy scale lied.** At 16² `clamp` reported **0** ANE rejections. At 128² decoder scale the
> same formulation reports **36**, all on `gather`. Scale changes ANE eligibility — echoing the
> Moebius finding that the compiler's split pass only engages at large spatial extents.
> **Never clear a graph for the ANE at toy resolution.**

## 5. The real blocker is `gather`

At decoder scale (64 ch, 128², 3 stacked DCNs, fp16) the cold-cache diagnostics name **only
`gather`** — 12 of them, 3 per bilinear corner × 4 corners:

```
failed: ANE I/O op can only do F16 MemRef <-> F32 Tensor cast   x36
ops: gather, gather_1 ... gather_11
```

Residency measurement, sustained 14 s with `macmon`:

| lane | gpu_freq | sys_power | rate | cold load |
|---|---|---|---|---|
| GPU | 1620 MHz | 110.9 W | **215.9/s** | 0.18 s |
| "ANE" | **1487 MHz — GPU busy** | 46.0 W | **39.0/s** | 3.95 s |

The ANE lane holds the GPU clock *high*, runs **5.5× slower** than the GPU lane, and pays a 3.95 s
specialization. That is a **partitioned graph** falling back for every scattered read — the worst
of both worlds.

**Conclusion: a scattered read (`gather` with int64 indices) is not ANE-resident.** Deformable
convolution is defined by scattered reads, so the whole family inherits this.

## 6. What this means for BiRefNet and its class

- **`deform_conv2d` is unblocked for CoreAI conversion** — approach A works, exactly, with no
  custom lowering. LibreYOLO's `blocked` row can be retired on conversion grounds.
- **It is NOT unblocked for the ANE.** A BiRefNet CoreAI port would run on the GPU, where it is
  competing directly with MLX — which removes the main reason to port it.
- Their suggested "encoder-only contract" route looks better than it did: the encoder has no
  deformable convs, so an encoder-only asset could be ANE-resident while the decoder stays
  elsewhere.

→ recorded in `mlx-vs-coreai-fit.md`.

## 7. Still OPEN

- Can the bilinear gather be expressed **without** `gather`? A dense one-hot matmul is
  algebraically possible but likely enormous; worth a bounded experiment.
- Does `TorchMetalKernel` help, or does it force GPU and moot the exercise? **Still the
  load-bearing unknown** from `custom-lowerings.md` — and now it matters less, since the graph is
  already GPU-bound.
- `register_torch_lowering` remains **unexercised**. Approach A won before B was needed. That is
  the right outcome for the port and a gap for the skill: write one deliberately on a case where
  approach A is unavailable.
