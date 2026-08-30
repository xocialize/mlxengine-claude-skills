# Case study — resolving EoMT's `GuardOnDataDependentSymNode`

The second sealed port (2026-08-30). A **capture** failure, not a missing operator — a different
class from `deform_conv2d`, and one that cannot be dodged by decomposition.

**Outcome: LibreYOLO's `blocked` row for `eomt` can be retired. The graph captures and
converts.** Full ANE residency remains blocked on different ops.

---

## Sealing

EoMT ships in HF `transformers` (`EomtForUniversalSegmentation`), so the attempt used the
upstream model plus the CoreAI docs — no LibreYOLO export code. Their *diagnosis string* was
known going in (partial reconnaissance, recorded honestly), but not their fix.

**Random weights are legitimate here.** A `torch.export` guard failure is structural — it depends
on control flow, not values. The value-dependent-bug rule in `ane-eligibility.md` is about the
*compiler*; this fails long before the compiler is reached.

## The failure

```
GuardOnDataDependentSymNode:
Could not guard on data-dependent expression Eq(u0, 1) (unhinted: Eq(u0, 1)).
```

Reproduced independently. `torch.export` also prints
`Unable to find user code corresponding to {u0}` — so the symbol's origin is *not* in the
traceback.

## Finding the host read

The error text was useless; the **graph dump inside it** was not. Grep the dumped FX graph for
`item`:

```
item: "Sym(Eq(u0, 1))" = torch.ops.aten.item.default(ne)
```

and read the `# File:` comment immediately above it — the dump carries source attribution:

```python
# modeling_eomt.py:1119
self.training or self.attn_mask_probs[idx - self.num_hidden_layers + self.config.num_blocks] > 0
```

> **Generalisable technique:** for `GuardOnDataDependentSymNode`, don't read the exception —
> grep the embedded graph for `aten.item`, `aten.nonzero`, `_local_scalar_dense`, and read the
> `# File:` line above the hit. That is the host read, with its source location.

## Root cause

`self.attn_mask_probs = nn.Buffer(torch.ones(config.num_blocks))` — a **buffer**, hence a graph
input. Two places branch on its *value* from Python:

```python
# :1119
self.training or self.attn_mask_probs[...] > 0
# :1210, in _disable_attention_mask
if prob < 1:
```

Each read becomes `aten.item()` on a graph input → unbacked symbol `u0` → unresolvable branch.

> **This is a VALUE branch, not a shape problem.** Making the export canvas static does not fix
> it. That distinction matters: the reported diagnosis for this row suggested a static-shape fix
> "the same shape as the rfdetr `torch._assert`", and it is a different defect class.

## Why static resolution is correct, not a shortcut

`attn_mask_probs` gates a **training-time augmentation**: `_disable_attention_mask` randomly
disables attention to query tokens via `torch.rand(...) > prob`. At inference the buffer is
all-ones, so:

| site | value | resolves to |
|---|---|---|
| `:1119` `probs[i] > 0` | 1.0 | **True** — do the mask prediction |
| `:1210` `prob < 1` | 1.0 | **False** — augmentation is a no-op |

Both constant for a given checkpoint, so both belong outside the graph.

**Bonus safety property:** this also keeps `torch.rand` out of the graph. Exporting with any
`prob < 1` would bake **one random draw** into the asset permanently. The preparation therefore
*refuses* to run when any value is `< 1` rather than silently producing a poisoned asset.

## The fix

Swap the buffer for a plain Python list for the duration of capture, so both comparisons are
ordinary Python. Reversible, and validated before mutating anything.
Implementation: `coreai-collection/recipes/eomt/prepare.py`.

MEASURED: prepared eager graph is **bit-identical** to the original — max |Δ| = **0.000e+00**.

## Results

| dtype | asset | CPU | GPU | ANE | ANE rejections |
|---|---|---|---|---|---|
| fp32 | 9.7 MB | **134.22 dB** | 133.70 | 133.70 | 36 — `slice_scatter_1/3` |
| fp16 | 4.9 MB | 31.57 | **66.62** | **43.58** | 45 — + `scaled_dot_product_attention_2` |

- **fp32 proves the graph** at ~134 dB. Capture failure resolved, conversion works.
- **fp16 on the ANE is 43.58 dB — below the 50 dB gate**, in the investigate band. The GPU lane
  is fine at 66.62 dB, so this is an ANE-precision issue, not a graph issue.
- **CPU is again the worst fp16 lane** (31.57 dB), consistent with the Real-ESRGAN measurement
  and with `apple/coreai-torch#74`. **Never use the CPU lane as a reference.**

## What still blocks ANE residency

- **`slice_scatter`** (both dtypes) — from the in-place mask write
  `attention_mask[:, :nq, enc:] = interpolated_logits > 0`. An in-place scatter into a slice.
- **`scaled_dot_product_attention_2`** (fp16 only).

> **Tempering an earlier note.** `custom-lowerings.md` observed that SDPA is a *preserved
> composite* and suggested attention may be better supported than Moebius implied. MEASURED here:
> preserved-as-composite does **not** imply ANE-eligible — one SDPA instance is rejected in fp16
> while the same graph's other attention sites are not. Do not generalise from the op list.

## Chasing the fp16-on-ANE gap — and the observer effect that blocks it

**The gap:** fp32 **134 dB**, fp16-GPU **66.62 dB**, fp16-ANE **43.58 dB** on the *identical
asset*. Neither the graph nor fp16 itself — 23 dB between two lanes running the same bytes.

**Attempt 1 — the official comparator.** `coreai_torch.debugging.comparator.create_comparator_for_programs`
is the right instrument: it bisects the graph op by op **and takes `specialization_options`**, so it
can run the target on a chosen lane. Rough edges found: it lives in the submodule (nothing is
exported at `coreai_torch.debugging` package level), it is a **coroutine** (as is
`Comparator.compare`), and `Comparator.Status` has only `PASS`/`FAIL`/`UNKNOWN` — no `SKIP`.
It then failed executing the *source* ExportedProgram under torch 2.11:
`TypeError: 'int' object is not callable ... While executing %_guards_fn`. **Unresolved.**

**Attempt 2 — instrument the graph. THIS IS THE TRAP.**
Re-exporting with every layer's hidden state as an extra output **broke ANE compilation
outright**:

```
_ANECompiler : ANECCompile() FAILED
Compiler internal error: It has to be valid custom strides
- From PEFUSED_GOC Layer: … TERNARY_DYNAMIC_GOC: Pre-Scale: 1, ScaleBiasNegate: Y,
  ScaleBiasBroadcast: [ W:192 ]
```

The ANE lane then fell back — and the tell was unmissable once the numbers were in front of me:
**every ANE row equalled its GPU row to the last decimal.**

| output | CPU dB | GPU dB | ANE dB |
|---|---|---|---|
| h0 (embeddings) | 71.83 | 84.01 | **84.01** |
| h1 | 69.20 | 80.33 | **80.33** |
| h2 | 67.50 | 78.25 | **78.25** |
| h3 | 66.64 | 76.88 | **76.88** |
| h4 | 65.38 | 75.86 | **75.86** |
| h5 (final norm) | 64.69 | 75.60 | **75.60** |
| class_logits | 56.89 | 66.62 | **66.62** |
| mask_logits | 56.76 | 69.81 | **69.81** |

> ### The observer effect
> **Adding outputs to a graph can change its ANE eligibility.** Extra outputs alter fusion —
> here producing a `TERNARY_DYNAMIC_GOC` the compiler rejects — so an instrumented export is
> **not the same artifact** you are trying to debug. Any re-export (added outputs, truncation,
> a prefix model) has this problem.
>
> **Therefore the only faithful instrument for an ANE numerics question is one that reads
> intermediates out of the SAME compiled asset** — the inspector/comparator path. Making that
> work is a prerequisite, not a convenience.

**What the run did establish** (GPU lane, still useful):
- fp16 loss is **gradual**, not a single bad layer: 84.01 → 75.60 dB across six stages.
- The `predict` head costs the most in one step: 75.60 → 66.62 dB.
- **The CPU lane is 10–12 dB worse than the GPU at every single stage**, not just at the output —
  further confirmation it cannot serve as a reference (`coreai-torch#74`).

## Still OPEN

- Can the `slice_scatter` be rewritten as a masked `where`/`cat` to become ANE-eligible? The
  natural next experiment, and the same shape of fix as the validity-mask rewrite in
  `case-deformable-conv.md`.
- Why is only `scaled_dot_product_attention_2` rejected and not its siblings?
- **The fp16 ANE 43.58 dB is still not root-caused**, and the two obvious routes are blocked:
  the comparator fails on torch 2.11's `_guards_fn`, and re-exporting with instrumentation
  changes ANE eligibility (above). Next step is to unblock the comparator — that is the only
  faithful instrument. The `-1e9` mask-fill suspicion is **eliminated**: the `dtype` variant
  replaced it with `finfo(fp16).min` and parity was unchanged at 43.58 dB.
