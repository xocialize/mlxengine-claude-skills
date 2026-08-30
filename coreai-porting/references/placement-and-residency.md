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

# RIGHT — build it, pass it, then echo it back OFF THE RUNTIME OBJECT
opts = SpecializationOptions(preferred_compute_unit_kind=kind)
if not opts.is_supported():
    raise SystemExit("placement unsupported here — refusing to produce mislabeled data")
model = await AIModel.load(path, specialization_options=opts)
print("actual:", model.specialization_options.preferred_compute_unit_kind)  # not the argv string
```

`SpecializationOptions.is_supported()` gates on `USE_OS_COREAI` semantics (wheel installs on
macOS 27+ default to the OS framework). **Exit hard when False** — silent fallback produces
mislabeled data, which is worse than no data.

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
