# Export recipes — capture, lowering, and graph preparation

---

## The baseline recipe

MEASURED-WORKING for both shipped ports (`coreai-core==1.0.0b2`, `coreai-torch==0.4.1`,
macOS 27.0). Run under `uv` in an isolated venv — torch is pinned to 2.11.x.

```python
import torch
from pathlib import Path
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(dummy,))       # dummy fixes the canvas
ep = ep.run_decompositions(get_decomp_table())
program = (TorchConverter()
           .add_exported_program(ep, input_names=["x"], output_names=["out"])
           .to_coreai())
program.optimize()
program.save_asset(Path("model_fp16_static128.aimodel"))
```

**Static shapes only** for ANE residency — the canvas the dummy fixes is the canvas the artifact
runs at, permanently. Encode dtype and canvas in the filename; the asset will reject mismatched
input dtypes and the caller needs to know from the name.

**A/B `optimize()`.** It has silently miscompiled (`coreai-torch#49`) and segfaulted (`#33`).
Export with and without it, parity-check both, compare size and all three lanes. Record which
you shipped and why. → `known-upstream-defects.md`.

**Screen activations before any fp16 export** — `softplus`, `mish`, `logsumexp`,
`logcumsumexp` overflow in fp16 on the ANE (`#21`).

**Export fp32 AND fp16.** The fp32 lane isolates "graph wrong" from "precision loss" in one
number → `precision.md`.

---

## Extending the decomposition table

When the converter has no lowering for an op but PyTorch core ships a reference decomposition,
fold it into the table rather than rewriting the model:

```python
from torch._decomp import get_decompositions
table = dict(get_decomp_table())
table.update(get_decompositions([torch.ops.aten.grid_sampler_2d]))
ep = ep.run_decompositions(table)
```

INHERITED (LibreYOLO) — this is how they lower deformable attention for the DETR families.
**Not yet independently measured by us.**

---

## Missing-lowering triage ladder

When lowering fails on an op, in order of preference:

1. **Is there a core decomposition?** → fold it into the decomp table (above). Cheapest, no
   model change, no numerics risk.
2. **Is the op algebraically equal to something simpler *in this configuration*?** → swap it.
   `AdaptiveAvgPool2d(1)` is *exactly* a spatial mean; any other output size is **not**, and a
   conversion error is better than a silent approximation. Gate the swap numerically.
3. **Can the computation move OUT of the graph?** → do it eagerly, before capture. LibreYOLO's
   RF-DETR position-embedding rebake is the model case: re-run the model's *own* baking path for
   the actual canvas so the interpolation happens ahead of capture, computing exactly what it
   computed before.
4. **Custom lowering** → `register_torch_lowering()`. **OPEN — never attempted by us.**
5. **Custom Metal kernel** → `TorchMetalKernel`. **OPEN — never attempted by us.**

**Warning on (3):** LibreYOLO's notes record that an earlier, *general* interception of
`F.interpolate` — replaying results by call order — silently changed outputs by up to **9.5e-01**
whenever the call sequence didn't line up. Reuse the model's own path; don't build a generic
replay.

---

## Graph preparation must be reversible

Any preparation that mutates the live model (folding BN, freezing anchor grids, baking
resolution-specific buffers) must:

- **Allocate and populate every replacement before mutating anything.** A failure on a later
  layer must leave earlier layers untouched.
- **Restore on the way out**, success or failure — an `ExitStack` of restore callbacks scoped
  around the capture.
- **Snapshot before a helper that mutates and returns nothing** — otherwise the caller's model
  silently keeps export-only state and misbehaves at other resolutions afterwards.

INHERITED (LibreYOLO's `_prepare_coreai_graph` / `_snapshot_rtdetr_static_eval`) and it is a
sound pattern. We have not yet needed it at that scale, but a port that mutates a shared model
object should follow it.

---

## Known preparation patterns, by cause

All rows below are **INHERITED from LibreYOLO and unverified by us** — they are leads for sealed
ports to rediscover and then grade against, not a recipe to apply blind.

| Cause | Pattern |
|---|---|
| Converter doesn't preserve Darknet's `eps`-after-`sqrt` BN | fold frozen inference BN into the preceding conv (algebraically exact) |
| `torch.export` makes unbacked symbols from live `h*w` anchor rebuilds | warm up with `export=False` so the cache genuinely fills, then freeze the grid as constants |
| No lowering for `_upsample_bicubic2d_aa` | re-bake position embeddings for the actual canvas, eagerly |
| No lowering for `as_strided` (from `AdaptiveAvgPool2d`) | exact spatial mean, **only** for output size 1 |
| No lowering for `grid_sampler_2d` | core decomposition, or a gather-based manual sampler |
| Upstream `avg_pool2d` arg-resolver off-by-one | normalize the node's arg tuple to full arity using documented defaults |

**One trap worth flagging even in an inherited list:** a warm-up forward run with `export=True`
can return early *without* populating the caches you are trying to freeze, so you freeze
construction-time garbage and get `IndexError: Dimension out of range` from a transpose. Warm up
with the export flag **off**.

---

## Per-family output contracts

Whatever wrapper defines the model's output tuple **must be shared between the conversion path
and the parity harness**. If the reference is built by a different code path than the artifact,
you are not measuring conversion — you are measuring the difference between two wrappers.

INHERITED (LibreYOLO) and clearly right: they reuse one `_wrap_coreai_contract` function in both
places specifically so the reference cannot accidentally retain a family's raw training outputs.

---

## OPEN

- `add_pytorch_module(..., externalize_modules=...)` — never used by us.
- `state_names` for mutable state / KV cache — never used by us.
- `set_static_shape_config()` for iOS — never used by us.
- AOT `xcrun coreai-build compile` — never run by us.
