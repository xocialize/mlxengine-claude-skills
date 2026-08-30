# Debugging methodology — when the error is opaque, redacted, or moves

CoreAI conversion errors are frequently `<private>`-redacted in the unified log and **empty in
the exception**. Guessing from error text alone found us *nothing*. Bisection found everything.

---

## The bisection ladder that worked

**MEASURED (Moebius, a full day's arc).** Each level took ~10 minutes:

1. **Per-block export + subprocess load**, with verdicts parsed from **stderr** — not from the
   exception, which was empty.
2. **Stage-split inside the failing module** — cumulative stages, each exported and loaded.
3. **Seam-split between passing modules** — find the composition that breaks.
4. **Minimal repro** — small enough to file upstream.

Run each level in a **subprocess**. A SIGABRT during specialization kills the host process
(→ `placement-and-residency.md`), so an in-process sweep loses every result after the first
crash.

---

## Non-monotonic verdicts mean your scaffolding is in the graph

**MEASURED.** A cumulative stage that "fails" while its **superset passes** is *impossible*. It
is not a compiler mystery — it means the harness is contributing ops.

Specifically: keep-alive hacks like `+ x.sum()*0`, inserted to stop the optimizer from pruning a
stage you want to measure, **fail the ANE compile on their own**.

> Before diagnosing the model, diagnose the harness. Any verdict pattern that violates
> monotonicity is a harness bug until proven otherwise.

---

## Capture stderr on the first attempt or lose the evidence

The specialization cache caches **failure-then-fallback**. After the first run, the diagnostics
are gone and every subsequent run looks clean. Details and the cache path →
`placement-and-residency.md`.

Practical shape:

```bash
python export_probe.py 2>stderr.txt
grep -v '^warning: loc' stderr.txt | grep -E 'ANECCompile|ane_validation_message|error'
```

The `ane_validation_message` lines name the failing op, its rank, **and the original Python
source line** via debug locations — the single most useful diagnostic the toolchain emits, and
it is buried in megabytes of MLIR `warning: loc(...)` noise.

---

## Build controls by deepcopy-then-randomize, never from config

**MEASURED, and it voided an entire verdict table.** When constructing a "same architecture,
random weights" control, build it by `deepcopy` of the real model followed by randomization.

Building from config silently produced a **different architecture** — diffusers' attention-class
routing selected a different attention implementation — so the control was not a control.

Related: value-dependent compiler bugs mean random-weight tests cannot clear a graph anyway.
→ `ane-eligibility.md`.

---

## Distinguish "hangs" from "dies"

**INHERITED (LibreYOLO, `swinir`), and a good diagnostic instinct worth keeping:** a process
whose **kill point moves between runs** is the signature of *memory exhaustion*, not a stuck
loop. One run reached "Step 3/3" before stopping; a later run of the same graph at the same
canvas died inside `to_coreai()`, both with a leaked-semaphore warning and no traceback.

Their own note also warns that an earlier single-run conclusion ("`optimize()` is at fault") was
**contradicted by the second run**. Two lessons:

1. Watch RSS during conversion before theorizing.
2. **A single run is not a diagnosis.** Neither for them nor for us.

---

## Gate rewrites numerically, inside the export script

Any algebraic rewrite (einsum folding, BN replacement, op substitution) gets a numeric gate in
the script itself: eager forward pre- vs post-patch, in fp32 (or fp64 for exactness claims),
**hard-exit on divergence**.

Cheap, and it catches transcription slips at the only moment they are catchable — before the
graph is captured and the evidence is gone.
