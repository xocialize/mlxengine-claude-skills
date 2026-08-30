# Runtime and toolchain — sharp edges the docs don't state

For the Swift side see `Skill("coreai-swift-integration")`. This file is the Python runtime,
the toolchain, and the asset format.

---

## Toolchain versions

MEASURED-WORKING pin for both shipped ports: `coreai-core==1.0.0b2`, `coreai-torch==0.4.1`,
macOS 27.0 / Xcode 27 beta.

`coreai-torch` **pins torch to 2.11.x**. Use an isolated `uv` venv; do not install it into a
shared environment.

### The `avg_pool2d` off-by-one — MEASURED, still live in 0.4.2

**MEASURED 2026-08-29.** `coreai_torch/_aten_to_core.py` reads `count_include_pad` as:

```python
node.args[5] if len(node.args) > 4 and node.args[4] is not None else True
```

The guard tests element **4**; the read is element **5**. A node carrying exactly five arguments
passes the guard and then raises `IndexError: tuple index out of range`.

**Present in BOTH `0.4.1` and `0.4.2` (latest as of 2026-08-29).** Verified by source read *and*
by execution.

Minimal repro — an entirely ordinary module, not an exotic construction:

```python
nn.AvgPool2d(2, 2, 1, ceil_mode=True)     # -> 5-arg node -> IndexError
F.avg_pool2d(x, 2, 2, 1, True)            # -> 5-arg node -> IndexError
F.avg_pool2d(x, 2, 2, 1, True, False)     # -> 6-arg node -> converts fine
```

`ceil_mode=True` is what keeps element 4 in the exported graph. With default `ceil_mode`,
`torch.export` normalizes the trailing args away, the node carries 3, the guard fails cleanly and
the `else True` branch is correct — which is why this stays hidden until a model happens to use
`ceil_mode`.

**Workaround without patching upstream:** pass `count_include_pad` explicitly so the node carries
six arguments.

**The version-scoping trap.** LibreYOLO ships a shim for this, scoped to exactly
`_AFFECTED_COREAI_TORCH_VERSIONS = {"0.4.1"}`, that **declines silently on any other version**.
Since the defect is still present in 0.4.2, anyone who bumps the toolchain has the fix removed
with no error and no log line. If we adopt version-scoped shims, they must **log when they
decline**, not just return False.

**OPEN — worth reporting upstream.** We have a minimal public repro. Also worth telling LibreYOLO
their shim's version gate is now too narrow.

---

## The async surface — MEASURED, and not what it looks like

```python
model = await AIModel.load(path, specialization_options=opts)   # ASYNC
fn    = model.load_function(name)                               # *** SYNC ***
out   = await fn({"x": nd})                                     # ASYNC
```

`load_function` is **not a coroutine** — `await model.load_function(...)` raises
`TypeError: object InferenceFunction can't be used in 'await' expression`. Only `AIModel.load`
and `InferenceFunction.__call__` are async.

Two useful things visible in the real signatures:

```python
AIModel.load_function(function_name, intermediate_logger=None, profiler=None)
InferenceFunction.__call__(inputs=None, state=None)
```

- `load_function` takes the **debug hooks directly** — that is the seam for `IntermediateLogger`
  and `Profiler` (→ `debugging-methodology.md`).
- `__call__` takes a **`state` dict** — the mutable-state / KV-cache surface. **Still unexercised
  by us**, but it is here, not somewhere exotic.

## The library prints a banner to STDOUT

`coreai-torch 0.4.1: converting 1 program(s) to Core AI` goes to **stdout**, not stderr. Any
harness that parses JSON from a conversion subprocess must take the **last** line, or filter for
`^{`. This cost a debugging cycle.

## `Profiler` requires ALL THREE callbacks

**MEASURED.** `Profiler(...)` with only `on_log_event` leaves the interval hooks `None`, the
native side calls them anyway, and it surfaces as an opaque:

```
SystemError: <...> returned a result with an exception set
```

thrown from **`load_function`** — nowhere near the actual mistake. Pass all three callbacks.

---

## First-load E5RT specialization

**MEASURED**, scales with graph size:

| Model | Params | Cold first load | Warm |
|---|---|---|---|
| SRVGG | 1.4M | ~8 s | ~0.2 s |
| Moebius UNet (fp16) | 226M | **254 s** | 1.1 s load + 0.3 s first call |

OS-cached per (model × machine). Budget it at the prepare seam and warn the user the first time.
Never inside the first user-visible inference.

The same cache caches failure-then-fallback → `placement-and-residency.md`.

---

## The asset is a directory

`.aimodel` is a **directory**, not a file. `main.mlirb` is roughly checkpoint-sized —
**MEASURED: 431 MB fp16 / 862 MB fp32** for a 226M UNet.

- Gitignore the exports dir **before** the first `git add -A`.
- `save_asset` wants a `Path`, not a `str`.
- Metadata is stored as **strings** — structured values must be JSON-encoded on the way in.

---

## Output naming contract

CoreAI returns a **named dict**, and the key order matches neither the eager forward's tuple
order nor anything a caller can guess. INHERITED (LibreYOLO), and consistent with our own
experience of probing `"output"` then `"out"`:

- **Never pair CoreAI outputs with eager outputs positionally.**
- Declared output names are graph node names (`cat_33`, `silu_120`, …) and carry no semantic
  meaning — which is exactly why they must be **recorded in the asset metadata at export time**
  rather than re-derived later.

---

## Stateful models — `state_names` has a reference implementation upstream

**INHERITED, 2026-08-30, `apple/coreai-models`.** Our "never done" note on mutable state is now
backed by a working example rather than a doc mention.

`python/tests/_runner_infra/export/exporters/coreai_exporter.py` ships a `CoreaiStatefulExporter`
taking `state_names`, and `testing_utils.py` drives it for a KV cache:

```python
CoreaiStatefulExporter(
    input_names=("input_ids", "position_ids"),
    output_names=("logits",),
    state_names=(key_cache_swift_name, value_cache_swift_name),
)
```

Note the shape of the contract: **`k_cache` / `v_cache` are passed as inputs to the export but
declared as `state_names`, not as `input_names`** — the exporter subtracts the state names from
the reference inputs to derive the true inputs. At runtime, state is passed via
`InferenceFunction.__call__(inputs, state=...)`.

Still **unexercised by us**, but the pattern is no longer a guess. Read that file before the
first stateful port.

## The official export CLIs

`apple/coreai-models` exposes task-level entry points rather than expecting a hand-rolled script:

```bash
uv run coreai.llm.export       google/gemma-3-4b-it --compression none --compute-precision bfloat16
uv run coreai.diffusion.export Wan-AI/Wan2.1-T2V-1.3B-Diffusers --compression 4bit-asym
```

**Compression is wired into the official path** (`--compression none|8bit|4bit-asym`), not only
available standalone via `coreai-opt` (→ `compression.md`). For a supported family, check for a
recipe in `models/` before writing an export script.

## OPEN

- **AOT compilation (`xcrun coreai-build compile --platform ...`) has never been run by us.**
  Unknown: whether it changes E5RT cost, whether it pins placement, whether it changes numerics.
  All three matter and all three are guesses today.
- **iOS has never been targeted.** `set_static_shape_config()` and the iOS-vs-macOS compression
  preset split are unexercised.
- Whether `powermetrics --samplers ane` is a usable residency oracle. Untried.

---

## `fn.desc` — introspecting an asset without guessing  (MEASURED 2026-08-30)

Not in the docs, and not discoverable from `dir(fn)`, which shows exactly one attribute. The
`InferenceFunctionDescriptor` behind it is how you drive an arbitrary `.aimodel` without
hard-coding names or shapes:

```python
fn = model.load_function(next(iter(model.function_names)))
d  = fn.desc
d.input_names        # ['x']
d.output_names       # ['logits']
d.state_names        # []
d.name               # 'main'

idesc = d.input_descriptor('x')     # also d.output_descriptor(...), d.state_descriptor(...)
idesc.shape          # [1, 3, 224, 224]
idesc.dtype          # 'float16'
idesc.rank           # 4
idesc.storage_kind   # 'ioSurface'
```

Three things this buys:

1. **Generic harnesses.** Build the feed dict straight from the descriptors instead of assuming
   an input is called `x` — `scripts/residency.py` does exactly this.
2. **The #75 budget, computed from the asset.** `len(d.output_names)` is the divisor:
   `floor(16384 / n_outputs)` inferences before the process dies. You can print a model's own
   safe budget before running it.
3. **`storage_kind` names the exhausted resource.** Outputs report `ioSurface` — the same
   allocation whose 2^14 table `coreai-torch#75` exhausts.

`rank` is worth reading at load time too: rank > 5 is the ANE eligibility ceiling
(→ `ane-eligibility.md`).
