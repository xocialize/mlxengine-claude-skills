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

## Still OPEN

- Can the `slice_scatter` be rewritten as a masked `where`/`cat` to become ANE-eligible? The
  natural next experiment, and the same shape of fix as the validity-mask rewrite in
  `case-deformable-conv.md`.
- Why is only `scaled_dot_product_attention_2` rejected and not its siblings?
- The fp16 ANE 43.58 dB has not been root-caused. Given `precision.md`, first suspects are
  normalisation statistics and the `-1e9` mask fill, which is **not fp16-representable**
  (fp16 max ≈ 65504) and will saturate to `-inf`.
