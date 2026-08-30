# Case study — SCUNet, and what window attention actually costs on the ANE

Ported 2026-08-30 from our own `mlx-scunet-swift/oracle/` rig (original KAIR checkpoint,
upstream source, independent goldens). macOS 27.0 (26A5421a), M5 Max, `coreai-torch 0.4.1`,
fp16 static 128².

**Chosen to probe the class that blocked SwinIR upstream.** LibreYOLO's `swinir` row reads:

> The export process DIES rather than hangs, and the kill point moves between runs … Window
> attention unrolls into a very large number of small ops, so the converter's peak memory is the
> prime suspect **on a 16 GB machine**. Next steps: watch RSS during conversion.

We are on 128 GB, so that was directly testable. It was tested.

## Results

| | value |
|---|---|
| fp32 torch vs the oracle's golden | **inf dB (bit-exact)** |
| export | **succeeds** |
| **peak RSS, whole run** | **1.05 GB** |
| graph | **3,825** `call_function` nodes, **max tensor rank 6** |
| asset (fp16) | 38.4 MB, 0 convert-time ANE hits |
| GPU | **71.22 dB**, 8.217 ms, cold 1.42 s |
| **ANE** | **37.78 dB, 249.256 ms**, cold 5.54 s, **678 validation hits** |

Rejected ops on the ANE: `unsqueeze_1, unsqueeze_17, unsqueeze_23, unsqueeze_33, unsqueeze_39, …`

## Three findings

### 1. Memory is not the problem — at least not for this architecture

**MEASURED: 1.05 GB peak RSS** across build, capture, decomposition, `to_coreai` and
`save_asset`. Nothing here would trouble a 16 GB machine.

**Stated carefully: this is evidence about the window-attention CLASS, not a refutation of their
SwinIR observation.** SwinIR is a different model and we have not run it. But the prime suspect
for SwinIR — "window attention unrolls into many small ops, so peak memory kills it" — does not
reproduce on the nearest architecture we can test. **Re-measure SwinIR before inheriting that
diagnosis.** Their own note already warns that an earlier single-run conclusion there was
contradicted by the second run.

The other half of their characterisation *is* confirmed: **3,825 call_function nodes** is a very
large number of small ops for a 17.9M-param model.

### 2. rank 6 — from `einops.rearrange`, exactly as the eligibility rules predict

SCUNet's WMSA partitions into windows with `einops.rearrange`. That produces **rank-6** tensors,
and `ane-eligibility.md` already records the hard rule: **MPS-ANEC rejects rank > 5.** The
pre-flight caught it statically, before any load:

```
graph: 3825 call_function nodes, max tensor rank 6 (RANK>5 -> ANE will reject)
```

> **Practical: `einops.rearrange` in a window-attention block is a rank-6 red flag.** Grep for it
> before exporting; the static prescan in `scripts/ane_preflight.py` reports max rank without
> needing a load.

### 3. A partitioned graph is not "a bit slower" — here it is 30× slower

| lane | latency | vs the other |
|---|---|---|
| GPU | **8.217 ms** | — |
| "ANE" | **249.256 ms** | **30.3× SLOWER** |

This is the most extreme partition we have measured — worse than deformable conv's 5.5×
(`case-deformable-conv.md`). With 678 rejected ops the graph is being shuttled between units
constantly.

**And parity collapses with it: 37.78 dB on the ANE vs 71.22 dB on the GPU**, from the same
asset.

> **Reinforces the central routing rule.** Choosing CoreAI does not mean getting the ANE, and a
> CoreAI port that silently fails to reach it can be dramatically worse than the GPU path — or
> the MLX path — it replaced. Nothing raised; the numbers were simply 30× worse.

## Verdict for the fit journal

**Window-attention restoration models belong on the GPU — CoreAI-GPU or MLX — not the ANE.**
SCUNet converts cleanly and runs well on the CoreAI GPU lane (71.22 dB, 8.2 ms), so a CoreAI
port is viable; it just competes directly with MLX there, with no ANE differentiation to justify
it.

## Open — a possible fix, untried

The rank-6 tensors come from window partitioning, which is a pure reshuffle. The Moebius einsum
folding precedent (`ane-eligibility.md`) suggests a rank-5 formulation may exist: fold the two
window axes into one, keep batch and heads merged. If that lands under rank 6 **and** eliminates
the `unsqueeze` rejections, window attention could become ANE-eligible — which would matter well
beyond SCUNet (SwinIR, Swin, and the ViT family inherit it).

Worth a bounded attempt. Verify any rewrite with `max abs diff == 0.0` in fp64 before trusting
it, and re-measure **every** lane afterwards.
