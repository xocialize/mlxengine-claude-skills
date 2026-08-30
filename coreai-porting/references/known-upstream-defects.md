# Known upstream defects — the failure-mode catalogue

Harvested from `apple/coreai-torch` issues, **2026-08-29** (23 issues at the time). All
**INHERITED** unless we reproduce them. Treated as a *hazard map*: these are failures other
people already paid for, and several would otherwise have cost us a port each.

**Re-harvest this list before each new port.** The tracker moves.

---

## Class 1 — SILENT WRONGNESS (the dangerous ones)

These produce numbers. The numbers are wrong. Nothing raises.

| Issue | Defect | Why it matters to us |
|---|---|---|
| **#49** | `AIProgram.optimize()` **removes broadcasting-significant axis moves and silently miscompiles** N×N distance expressions | **Our baseline recipe calls `optimize()` unconditionally.** See below. |
| **#10** | GPU delegate executes `aten.floor` / `trunc` / `ceil` as **identity**; `round` uses away-from-zero ties; `div(x,1,floor)` folds to identity | Any graph with rounding/flooring is suspect on the GPU lane. Grid/index math especially. |
| **#9** | Converter folds float→int→float cast round-trips, **dropping truncation semantics** (CPU too) | Quantise/dequantise patterns and index computations |
| **#11** | Runtime **clobbers an unrelated live tensor** when an int64-comparison bool-mask chain executes — explicitly the **deformable-attention sampler pattern** | **Directly hits the DETR families** (`rtdetr`, `dfine`, `deim`, `rfdetr`). Wave 3 must check this. |
| **#57** *(closed)* | `conv_transpose` lowered `output_padding` by padding the input, **silently producing an over-long output** | Fixed, but it is the shape of thing to look for in decoders |

### `optimize()` is not free, and our recipe assumed it was

Our documented recipe ends `program.optimize()` → `save_asset(...)` with no discussion. The
tracker shows `optimize()` has both **silently miscompiled** (#49) and **segfaulted** (#33,
closed).

**Revised guidance — MEASURE IT:** export **both** ways (optimized and un-optimized), parity-check
each, and compare size and latency on all three lanes. Per the A/B rule, `optimize()` is a
hypothesis like any other. Record which you shipped and why.

Note this also refines an inherited claim: LibreYOLO's `swinir` note warns *"Do NOT assume
`optimize()` is at fault"* after one run contradicted another. Both are true — do not *assume*
it, but do not exonerate it either. Test it.

---

## Class 2 — fp16 and ANE numerics

| Issue | Defect | Impact |
|---|---|---|
| **#51** | **[ANE] fp16 numerical discrepancy in MobileNetV3** (2D MatMul + Hardswish) | **`mobilenetv4` was slated as a Wave-1 "near-ideal ANE citizen." That assumption is now unsafe** — the sibling architecture has a known fp16/ANE discrepancy. Keep it in Wave 1, but expect the fp16 gate to be where the interest is, not a formality. |
| **#21** | `softplus`, `mish`, `logsumexp`, `logcumsumexp` **overflow in fp16 on ANE** — missing stable decompositions | Screen for these activations *before* exporting fp16. `mish` in particular appears in YOLO-family necks. |
| **#67** | **An fp16 asset aborts the process on ANE load** instead of returning an error | A crash at load is not necessarily *your* graph. → `debugging-methodology.md` on subprocess isolation. |
| **#5** | Open contribution proposal to add stable `softplus`/`mish`/`logsumexp` | Watch — it may close #21 |

Cross-reference our own MEASURED fp16 finding: subnormal `running_var` BatchNorm in fp16
(→ `precision.md`). The pattern is consistent — **fp16 on the ANE has a stability tax that fp32
never reveals**, which is precisely why an fp32-only support matrix cannot certify a model.

---

## Class 3 — process aborts (not exceptions)

These kill the interpreter. **Run conversions and loads in a subprocess** or lose every result
after the first.

| Issue | Trigger |
|---|---|
| **#68** | Zero-sized tensors on GPU/ANE — a 0-length split section, a width-0 output |
| **#58** | Depthwise `conv_transpose` when `output_padding > padding` (`explicit_padding = -1`) |
| **#8** | Converter aborts (`bad_optional_access`) on `aten.arange` with **float** start/end/step |
| **#6** | ANECompiler crash (`EXC_BAD_ACCESS`) at `AIModel.load` when `slice_update` begin/end are **runtime values** |
| **#2** | MPSGraph null-deref with 2+ `GatedDeltaUpdate` layers + attention with **dynamic KV context** |

#6 and #2 share a theme with our own findings: **runtime-valued indices and dynamic extents are
where the ANE stack breaks.** Static shapes are not merely an ANE *preference* — they are how
you stay out of this class entirely.

---

## Class 4 — performance and externalization

| Issue | Note |
|---|---|
| **#66** | `aten.topk` costs a **fixed 1.75 ms** on device when the rest of the graph is on the ANE — a single op can dominate. Detection heads use topk. |
| **#1** | `externalize`: SDPA submodule re-export **drops the upper bound on the key-length dim** with static query + dynamic KV |
| **#3** | No support for transposed conv3d |
| **#70** | Python 3.14 unsupported on macOS (3.13 landed in #4) |

---

## What this catalogue changes about our plan

1. **`optimize()` gets A/B'd on every port**, starting with Phase 1.
2. **`mobilenetv4`'s Wave-1 "easy win" status is withdrawn** — #51 makes its fp16/ANE lane the
   interesting part rather than a formality. That is *better* for the learning objective.
3. **Wave 3's DETR families inherit a named hazard** (#11, deformable-attention bool-mask
   tensor clobbering) to check for deliberately.
4. **Screen activations for `mish`/`softplus`/`logsumexp` before any fp16 export** (#21).
5. **Every conversion and load runs in a subprocess.** Five separate abort classes justify it.

---

## Reporting routing — MEASURED the hard way

| Defect type | Where it goes |
|---|---|
| `coreai-torch` Python layer (converter, lowerings, decomp table) | **`apple/coreai-torch` GitHub issues** — public repo since 2026-08-26, issues enabled |
| ANECompiler / MPSGraph internal errors | **Apple Feedback Assistant**, not GitHub |

Evidence: our `apple/coreai-models#138` (compositional ANECCompile failure) was **closed
`COMPLETED` on 2026-08-14 without a fix**, with the sole comment *"This looks like an upstream
bug. Can you file a report with the Feedback Assistant?"*

**So the Moebius ANE door is still shut**, and #138 being closed is not evidence otherwise. Any
compiler-level bug we find needs a Feedback Assistant report to go anywhere.

Our filings: **`apple/coreai-torch#73`** (avg_pool2d `count_include_pad` off-by-one, still live
in 0.4.2).
