# Runtime and toolchain — sharp edges the docs don't state

For the Swift side see `Skill("coreai-swift-integration")`. This file is the Python runtime,
the toolchain, and the asset format.

---

## Toolchain versions

MEASURED-WORKING pin for both shipped ports: `coreai-core==1.0.0b2`, `coreai-torch==0.4.1`,
macOS 27.0 / Xcode 27 beta.

`coreai-torch` **pins torch to 2.11.x**. Use an isolated `uv` venv; do not install it into a
shared environment.

**Version-scoped shims are a trap worth knowing about.** LibreYOLO's `avg_pool2d` compat shim
(a real off-by-one in upstream's resolver — the guard tests `args[4]`, the read is `args[5]`) is
scoped to exactly `{"0.4.1"}` and **declines silently on any other version**. A toolchain bump
therefore removes the fix with no error. If we adopt version-scoped shims, they must *log* when
they decline, not just return False.

---

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

## OPEN

- **AOT compilation (`xcrun coreai-build compile --platform ...`) has never been run by us.**
  Unknown: whether it changes E5RT cost, whether it pins placement, whether it changes numerics.
  All three matter and all three are guesses today.
- **iOS has never been targeted.** `set_static_shape_config()` and the iOS-vs-macOS compression
  preset split are unexercised.
- Whether `powermetrics --samplers ane` is a usable residency oracle. Untried.
