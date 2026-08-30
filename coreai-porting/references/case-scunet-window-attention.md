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

## THE FIX WORKED — window attention CAN be made ANE-resident

**MEASURED 2026-08-30.** Three rank-6 sources removed, all exact:

| source | rewrite |
|---|---|
| `rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c')` | split one axis at a time, **merging the batch axis with each window axis as it is produced** so no intermediate exceeds rank 5 |
| `einsum('hbwpc,hbwqc->hbwpq')` — 6 distinct indices | fold `(h,b,w)` into one batch axis → plain `bmm` at **rank 3** |
| `generate_mask`'s `torch.zeros(h,w,p,p,p,p)` | constant for a static canvas — **precompute EAGERLY on a warmup pass** so rank 6 lives outside the traced graph |

**Exactness gates, all passed before export:** partition/merge round-trip and attention at
**max |Δ| = 0.0e+00 in fp64**, then the **whole model across all 28 WMSA modules at 0.000e+00**,
with the original forward restored cleanly afterwards.

### Result

| | baseline | **rank-5** |
|---|---|---|
| graph nodes | 3,825 | **2,789** (−27%) |
| max tensor rank | **6** | **5** |
| ANE validation hits | **678** | **0** |
| ANE latency | 249.9 ms | **28.8 ms** — **8.7× faster** |
| ANE parity | 37.78 dB | 39.12 dB |
| GPU latency / parity | 7.85 ms / 71.22 dB | 7.94 ms / 71.18 dB — **unchanged** |

**Residency PROVEN**, not inferred — sustained 12 s under `macmon`:

| lane | gpu_freq | sys_power | rate | mJ/inference |
|---|---|---|---|---|
| GPU | 1620 MHz | 86.6 W | 133.3/s | 567 |
| **ANE** | **338 MHz — idle clock** | 23.6 W | 23.5/s | 536 |

**This is the first graph we have taken from ANE-rejected to ANE-resident.**

### But the ANE is still the wrong lane for SCUNet — and that is the point

| axis | verdict |
|---|---|
| eligibility | **fixed** — 0 rejections, residency proven |
| speed vs GPU lane | still **3.6× slower** (28.8 vs 7.9 ms) |
| fp16 parity | **39.12 dB — still fails the 50 dB gate**, essentially unmoved from 37.78 |
| energy | a wash — 536 vs 567 mJ/inference (1.06×) |

> ### Eligibility and suitability are INDEPENDENT problems.
> We conflated them. Removing every rejection was entirely a **graph-shape** exercise and it
> worked completely. It moved parity by 1.3 dB — i.e. not at all. A model can be perfectly
> ANE-resident and still belong on the GPU.
>
> Practical consequence: **`ane_preflight` answering "clean" is necessary, not sufficient.**
> Always follow it with the parity and latency measurement.

### Why the unlock matters anyway

The rewrite is architecture-generic — it targets the window-partition and multi-index-einsum
patterns, not SCUNet. **SwinIR, Swin, and the ViT family all use the same shapes.** For a model
where the ANE *is* the right lane on the other axes, rank-6 is no longer a blocker.

Implementation: `coreai-collection/recipes/scunet/rank5.py` (`win_partition_r5`,
`win_merge_r5`, `attn_r5`) and `patch_r5.py` (reversible binding + eager mask precompute).

## Superseded — the original "open, untried" note

The rank-6 tensors come from window partitioning, which is a pure reshuffle. The Moebius einsum
folding precedent (`ane-eligibility.md`) suggests a rank-5 formulation may exist: fold the two
window axes into one, keep batch and heads merged. If that lands under rank 6 **and** eliminates
the `unsqueeze` rejections, window attention could become ANE-eligible — which would matter well
beyond SCUNet (SwinIR, Swin, and the ViT family inherit it).

Worth a bounded attempt. Verify any rewrite with `max abs diff == 0.0` in fp64 before trusting
it, and re-measure **every** lane afterwards.
