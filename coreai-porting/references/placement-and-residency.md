# Placement and residency — proving where the graph actually ran

**Status: this is the area where we have the LEAST experience and the MOST to gain.** The ANE is
the reason CoreAI is interesting at all (it is the only way to reach it), and almost everything
below was learned by being wrong first.

Read this before any benchmark, any ANE claim, and any CoreAI-vs-MLX comparison.

---

## The core failure: a preference is not a placement

**MEASURED.** In Python, `AIModel.load(path)` with **no** `specialization_options` silently uses
default delegate placement. A benchmark script that parses `--compute` but forgets to build and
pass the options object produces identical-looking output with a false label.

This happened to us. A published "ANE" table was actually the GPU.

```python
# WRONG — the flag is inert, the label is a lie
model = await AIModel.load(path)

# RIGHT — see the three API traps below; use scripts/placement.py
from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
if not SpecializationOptions.is_supported():
    raise SystemExit("placement unsupported here — refusing to produce mislabeled data")
kind = ComputeUnitKind.neural_engine()          # CALL the factory — see trap 1
opts = SpecializationOptions.from_preferred_compute_unit_kind(kind)
assert str(opts.preferred_compute_unit_kind) == "Neural Engine"
model = await AIModel.load(path, specialization_options=opts)
print("actual:", opts.preferred_compute_unit_kind)   # off the object, not the argv string
```

**Use `scripts/placement.py`** — it wraps all of this, times the lanes, scans stderr, and
returns evidence instead of a claim.

---

## Three API traps in `SpecializationOptions`

**MEASURED 2026-08-29, macOS 27.0 (26A5421a), M5 Max, `coreai-core==1.0.0b2`.** Each of these
produces *plausible* wrong data rather than an obvious failure.

### Trap 1 — the compute-unit kinds are factories, not constants

`ComputeUnitKind.cpu` / `.gpu` / `.neural_engine` are **staticmethods**. You must call them:

```python
ComputeUnitKind.neural_engine()     # correct
ComputeUnitKind.neural_engine       # raises: Invalid ComputeUnitKind
```

Passing the uncalled attribute raises `RuntimeError: Invalid ComputeUnitKind in
preferred_compute_unit_kind`. That is loud — **but it is exactly the kind of exception a
broad `try/except` swallows before falling back to default placement**, at which point every
number in the run is mislabeled.

### Trap 2 — `available_kinds()` order is NON-DETERMINISTIC across processes

MEASURED over 6 consecutive processes:

```text
run 1: ['CPU', 'GPU', 'Neural Engine']
run 2: ['Neural Engine', 'CPU', 'GPU']
run 3: ['Neural Engine', 'GPU', 'CPU']
run 4: ['GPU', 'CPU', 'Neural Engine']
run 5: ['Neural Engine', 'GPU', 'CPU']
run 6: ['Neural Engine', 'GPU', 'CPU']
```

Stable *within* a process, shuffled *between* them — the signature of hash/pointer iteration
order.

> **`available_kinds()[0]` selects a different compute unit on every run.** A harness that
> indexes it produces silently mislabeled data that reads as *noise*, not as a bug — which
> defeats even a careful person who spot-checks once.

**Never index it. Select by name via the factories.**

### Trap 3 — you cannot forbid fallback off the ANE

`from_preferred_compute_unit_kind` sets a **preference**. MEASURED: `allowed_compute_unit_kinds`
remains **all three** for every preferred lane.

| Constructor | preferred | allowed |
|---|---|---|
| `default()` | `None` | all 3 |
| `from_preferred_compute_unit_kind(ane())` | Neural Engine | **all 3** |
| `cpu_only()` | `None` | CPU only |

`cpu_only()` is the **only** restriction primitive the API exposes. There is no
`neural_engine_only()`.

> **Silent fallback off the ANE is structurally unavoidable through this API.** That is not a
> bug in our harness — it is the contract. It is *why* the control lanes and the stderr scan
> below are the only defence, and why a completed run is never evidence of ANE execution.

### Also noted

`is_supported()` returned **True without `USE_OS_COREAI=1`**, contradicting its own docstring
("Returns True on macOS with env var `USE_OS_COREAI=1`"). The runtime module resolves to
`_coreai_runtime_os`, so the wheel appears to default to the OS framework on macOS 27. Treat the
docstring as stale; trust the returned value.

---

## Two failure modes when you request the ANE, and only one is loud

| Mode | Trigger | Symptom | Detectable? |
|---|---|---|---|
| **(a) Hard raise** | validation rejects the graph | `SystemError` out of `load_function` | Loud. Fine. |
| **(b) Silent fallback** | validation passes, **codegen fails** | `_ANECCompiler: ANECCompile() FAILED` on **stderr**, then GPU numbers with your "ANE" label | **Only via stderr + a control lane** |

MEASURED: mode (a) on rank-6 reshapes; mode (b) on the two-transformer compositional bug.

**A completed run is NOT evidence of ANE execution.** This is the single most important
sentence in this file.

---

## The control-lane protocol — non-negotiable

**Always run CPU as a control.** If CPU / GPU / ANE all return the same latency, the flag is
inert.

MEASURED (Moebius UNet, 226M): before the fix, all three lanes read ~205 ms — three agreeing
lanes is the tell. After passing the options object properly: CPU 601 ms, GPU 205 ms.

> **If your "ANE" latency equals your GPU latency, it IS the GPU.**

Two lanes with equal latency are the same lane. There is no exception to this that we have found.

---

## Residency oracles, ranked by trustworthiness

1. **`macmon` GPU-idle signature — the best oracle we have.** MEASURED: `gpu_freq` pinned at the
   338 MHz idle clock throughout a sustained inference means the GPU is not doing the work.
   Latency deltas are **not** an oracle; this is.
2. **stderr `ane_validation_message` warnings.** The best *diagnostic* evidence — they name the
   failing op, its rank, and the ORIGINAL Python source line via debug locations. They are buried
   in enormous MLIR blobs: filter with `grep -v "^warning: loc"` and grep `ane_validation_message`
   separately.
3. **stderr `ANECCompile` + control lanes.** The reliable detector for mode (b).
4. ~~Latency~~. Not an oracle. See above.

**OPEN:** we have not yet tried `powermetrics --samplers ane` or Instruments' Neural Engine
track as residency oracles. Both are plausible and would be stronger than the GPU-idle
inference. Worth trying on the next port — this is a named gap, not a settled answer.

---

## The specialization cache caches FAILURE

**MEASURED, and it cost us a whole diagnosis.** A first ANE attempt that prints
`ANECCompile() FAILED` and falls back to the GPU gets its **(GPU)** specialization cached under
that asset. Every subsequent load is fast, error-free, and still on the GPU.

Consequences:

- **The diagnostic evidence — validation warnings, compiler errors — exists ONLY on the first
  cold-cache attempt.** Capture raw stderr on that run or lose it.
- **A re-run with a warm cache tells you nothing.** To retry diagnosis, delete the asset's
  entries under `~/Library/Caches/coreai-cache/<os-build>/<bundle-id>/<hash>` (identify by
  mtime).
- Any benchmark harness must be able to run cold, and must say which mode it ran in.

---

## After ANY graph rewrite, re-measure EVERY lane

**MEASURED — we got this wrong.** A 4× "ANE" speedup on Moebius was really the *new graph's GPU
lane*; the elimination argument had compared it against the *old graph's* GPU number.

A rewrite changes every lane. Attribute a speedup to placement only after re-running all three.

Corollary worth keeping: **ANE-motivated rewrites can be GPU wins.** The rank-5-free λ
formulation (Conv3d depth folded into conv batch; einsums → broadcast/batched matmul) made the
MPSGraph **GPU** delegate 4.1× faster (205 → 50 ms). Keep placement-motivated fixes even when
the placement itself stays blocked.

---

## Crash forensics

**MEASURED.** A SIGABRT during specialization lands in MPSGraph/Metal *inside your process*
(`__assert_rtn` → `MPSGraphExecutable runMLIRModulePassesAndCommonInit`), with the assertion text
**only on stderr** — the `.ips` has no `asi` field. Capture stderr or lose it.

> A crash in a backend you don't believe you're using means your placement instrumentation is
> lying. Treat it as an instrumentation bug first, a model bug second.

---

## Checklist before claiming a placement

- [ ] Options object **built and passed** (not just parsed from argv)
- [ ] `is_supported()` checked, hard-exit on False
- [ ] Compute unit **echoed back off the runtime object**
- [ ] CPU control lane run, and it differs
- [ ] Cold-cache run done, raw stderr captured and searched for `ANECCompile`
- [ ] `ane_validation_message` lines extracted (or confirmed absent)
- [ ] GPU-idle-clock checked during sustained inference
- [ ] Every lane re-measured after the most recent graph rewrite
