# ANE eligibility — what the Neural Engine will and won't accept

The GPU delegate is permissive; the ANE is not. Most of these only bite when you **explicitly
request** ANE, which is why they stay invisible until you start proving placement
(→ `placement-and-residency.md`).

---

## Hard rule: max tensor rank is 5

**MEASURED.** MPS-ANEC hard-rejects rank > 5.

The trap: `torch.export` decomposes an einsum with six distinct indices — e.g.
`'n m k u, b u v m -> b n k v'` — through **rank-6 reshapes**. The GPU delegate doesn't care, so
the graph looks fine until you ask for ANE, and then you get a hard raise out of
`load_function`.

**MEASURED again 2026-08-30 (SCUNet):** `einops.rearrange` used for window partitioning in a
Swin-style block produces rank-6 tensors just as multi-index einsums do — 678 ANE rejections and
an ANE lane **30× slower than the GPU**. → `case-scunet-window-attention.md`. Treat
`einops.rearrange` in any window-attention block as a rank-6 red flag and check
`graph_prescan()` before exporting.

**FIX GENERALISED AND PROVEN 2026-08-30 (SCUNet):** rank-6 from window attention is removable,
and doing so took a graph from **678 ANE rejections to 0** with residency proven on the GPU-idle
oracle. Three patterns, all exact in fp64:

1. **Window partition** — split one axis at a time and **merge the batch axis with each window
   axis as it is produced**, so no intermediate exceeds rank 5. Never materialise
   `b w1 w2 p1 p2 c`.
2. **Multi-index einsum** — fold the leading indices into one batch axis and use `bmm`
   (`'hbwpc,hbwqc->hbwpq'` → rank-3 batched matmul).
3. **Constant rank-6 tensors** (attention masks built with `torch.zeros(h,w,p,p,p,p)`) —
   precompute **eagerly on a warmup pass** so they enter the graph as constants.

Reusable implementation: `coreai-collection/recipes/scunet/rank5.py`.
**Caveat that matters: this fixes ELIGIBILITY, not accuracy or speed** — SCUNet's fp16 parity
moved only 37.78 → 39.12 dB and the ANE stayed 3.6× slower than the GPU lane.
→ `case-scunet-window-attention.md`.

**Fix:** fold multi-index einsums to batched matmul (contract axes merged, batch axes merged).
Algebraically exact — so *prove it*: verify `max abs diff == 0.0` in **fp64** before trusting it.

**Patching trap, MEASURED:** patch at the **module-global binding** of the einsum helper. A
`from x import _einsum` means patching the source module after import is a **no-op** — you must
rebind the *consumer* module's global.

**Gate the rewrite numerically inside the export script**: fp32 eager forward pre- vs post-patch,
hard-exit on divergence. Cheap, and it catches transcription slips at the only moment they are
catchable.

---

## Static shapes only

**MEASURED.** Static shapes are required for ANE residency. Tile/canvas geometry becomes a
**build-time property of the asset** — one executable per static shape.

This is the exact **inversion** of an MLX port's runtime-injectable geometry. Document the
inversion in both packages so nobody "fixes" either toward the other.

---

## Compositional compiler bugs exist

**MEASURED (macOS 27.0, CoreAICompiler 3600.79.1).** Two 64²-level `Transformer2DModel`s in one
graph break ANECCompiler's input-channel-split pass — *"failed create split by input channel,
graph is changed"* — even though **every component and every single-transformer composition
compiles**.

- The split pass only engages at large spatial extents; the same pair at 16²/32² is fine.
- **No graph-level dodge worked**: linearized projections, resnet separation, and forced
  region-breakers (rank-6 reshape roundtrip, fp32 bounce ×1.0000001) were all absorbed by the
  optimizer/segmenter.
- Filed upstream as `apple/coreai-models#138` with a public-weights repro. **Re-test per macOS
  update before assuming the door is still closed.**

---

## Value-dependent compiler bugs exist

**MEASURED, and this one invalidates a whole class of testing.** The two-transformer
ANECCompile failure **reproduces with trained weights and vanishes with same-architecture random
weights**. Four `[320]` LayerNorm affine vectors alone flip it; same-range synthetic values do
**not**.

Two consequences:

1. **Random-weight smoke tests cannot clear a graph for ANE.** Test with real weights.
2. **Build a "random control" by deepcopy-then-randomize**, never from config — diffusers'
   attention-class routing silently produced a *different architecture* from config and voided
   an entire verdict table.

---

## Scaffolding artifacts masquerade as eligibility failures

**MEASURED.** Keep-alive hacks like `+ x.sum()*0` fail the ANE compile **on their own**.

If a cumulative stage "fails" while its superset passes, that is impossible — non-monotonic
stage verdicts mean **your scaffolding is in the graph**, not that the model is at fault.
→ `debugging-methodology.md`.

---

## Known-hostile operators

| Op | Status | Note |
|---|---|---|
| rank-6 reshape (from 6-index einsum) | **MEASURED** hard reject | fold to batched matmul |
| `aten.as_strided` (from `AdaptiveAvgPool2d`) | **INHERITED** (LibreYOLO) — no lowering | primitive, no decomposition; `AdaptiveAvgPool2d(1)` ≡ spatial mean, exactly |
| `aten._upsample_bicubic2d_aa` | **INHERITED** (LibreYOLO) — no lowering | move the interpolation out of the graph, eagerly |
| `aten.grid_sampler_2d` | **INHERITED** (LibreYOLO) — no lowering | PyTorch core ships a reference decomposition; fold it into the decomp table |
| `deform_conv2d` (torchvision) | **INHERITED** (LibreYOLO) — no lowering | needs a custom lowering. **OPEN — never attempted by us** |

**INHERITED rows are leads, not verdicts.** Each needs its own measurement before it goes in a
recipe. See the `patch_nearest_upsample` case in `measurement-protocol.md` for why.

---

## OPEN questions

- Do these rank/op rules differ between macOS 27 point releases? Unmeasured.
- Is there a documented ANE op allowlist, or is stderr validation the only source of truth?
- What is the ANE's actual behaviour on grouped/depthwise convs at scale? We folded Conv3d depth
  into conv batch for Moebius and it helped, but the *why* is unconfirmed.
