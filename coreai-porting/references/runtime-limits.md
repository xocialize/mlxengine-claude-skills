# Runtime limits — the 16,384-output-allocation ceiling

**MEASURED 2026-08-30.** macOS 27.0 (26A5421a), M5 Max 128 GB, `CoreAIRuntime 3600.83.2.14.1`,
`ANEServices 10.19`, `coreai-core==1.0.0b2`.

> ## A CoreAI process dies after exactly 2^14 = 16,384 output `NDArray` allocations.
>
> Not a slowdown, not an exception — an **uncatchable** Swift precondition
> (`EXC_BREAKPOINT` / SIGTRAP). Filed as **`apple/coreai-torch#75`**.
>
> ```
> max_inferences_per_process = floor(16384 / num_outputs) - (small fixed overhead)
> ```

This is the single most important operational fact we have found. It bounds every design that
assumes a long-lived inference process.

> ### ⚠️ Correction, 2026-08-30 — the original "~8,000 inferences" was wrong
>
> #75 was filed off a probe with **two** outputs, which dies at ~8,192. I recorded that as an
> *inference* ceiling. It is an **allocation** ceiling, and the divisor is the model's output
> count. A single-output model gets twice the budget; a four-output model gets a quarter.
> **The unit was wrong, not just the number** — and the wrong unit is the one that makes the
> limit look unpredictable across models. It is in fact exactly constant.

---

## The measurement

Run to failure, heartbeat fsync'd every iteration near the ceiling (the SIGTRAP is uncatchable,
so the count must already be on disk):

| model | outputs | last completed | × outputs |
|---|---:|---:|---:|
| `nout1` (synthetic) | 1 | 16,383 | **16,383** |
| `nout2` (synthetic) | 2 | 8,191 | **16,382** |
| `nout4` (synthetic) | 4 | 4,095 | **16,380** |
| `resnet18` fp16 224 | 1 | 16,381 | 16,381 |
| `mobilenetv4_conv_small` | 1 | 16,381 | 16,381 |

Warmup calls count. Varying warmup 0 / 3 / 25 on `resnet18` gives last-completed
16,381 / 16,378 / 16,356 — `warmup + counted = 16,381` every time. The budget is **per process**,
consumed from process start, not per loop.

Real models land 0–3 short of the synthetic ones; treat 16,384 as the hard cap and assume a small
model-dependent fixed overhead.

Reproduce: `coreai-collection/recipes/wave2/residency_check.py --to-failure --heartbeat <path>`.

## Two crash sites, one cause

```
CoreAIRuntime/NDArray+SharedStorage.swift:108: Fatal error: Failed to allocate storage for
NDArray with requirements: NDArrayDescriptor(… storageKind: ioSurface)

CoreAIRuntime/NDArray+Pool.swift:77: Fatal error: Failed to allocate storage for NDArray
with byteCount: 1572864, sk: ioSurface, st: float16
```

A **third** site appears on the small synthetic models — different `CoreAIRuntime` frame offsets,
and register `x22 = 0x4000` (**16,384**) live at the trap next to
`type metadata for DefaultStringInterpolation`. That is a capacity check building an interpolated
message around its own constant, and it is the best direct evidence that 16,384 is a table size
rather than an emergent resource limit.

Crash thread is always `libswiftCore _assertionFailure` → `CoreAIRuntime` ×6 →
`_coreai_runtime_os` → `completeTaskWithClosure`.

## What it is NOT — all measured, not assumed

| Hypothesis | Verdict |
|---|---|
| ANE-specific | **No.** The GPU lane dies at the identical count. |
| Caller retaining results | **No.** `del` + `gc.collect()` every 100 iterations change nothing. |
| Per-`AIModel` accumulation | **No.** Reloading every 2,000 calls dies at the same point — **and makes it worse** (below). |
| Memory pressure | **No.** ~308 MB writable at death on a 128 GB machine. |
| Model-specific | **No.** Constant across five models and two compute units. |
| Model-*size*-specific | **No.** A 3-conv toy and `resnet50` die at the same allocation count. |
| System-wide | **No.** Per-process; a fresh process gets a fresh 16,384. |

Everything points at a fixed-capacity table of IOSurface **handles**, not bytes.

## Reloading the model is a trap

The obvious mitigation — periodically rebuild the model to drain the pool — is strictly worse.
MEASURED from the crash report of a reload-every-2,000 run: **five live `ANEServicesThread`s**,
one per reload, none reaped, and `Neural Engine (reserved) 35.6M across 5 regions` against
`7296K / 1 region` for a single-load run. It still hits the same ceiling, and leaks a thread and
an NE region each time.

## What this means for design

1. **The budget is computable, so it is schedulable.** This is the practical difference the
   correction makes. "~8,000, cause unknown" forces a conservative guess; `16384 / n_outputs`
   lets a supervisor recycle a worker at a known safe margin (we use 80%).
2. **Output count is a design variable with a runtime cost.** Splitting a head into four separate
   outputs quarters the process lifetime. Prefer one concatenated output and slice host-side —
   the same rewrite is usually good for ANE residency anyway.
3. **Process recycling is the workaround** — shard across subprocesses, restart before the cap.
   Viable for batch and for request/response services behind a supervisor; still a real problem
   for anything stateful in-process.
4. **It still belongs in the fit journal**, but as a *cost*, not a *disqualifier*. → `mlx-vs-coreai-fit.md`.
5. **Budget it into any benchmark harness.** A 20,000-iteration single-output sweep will not
   complete. An 8-second sustained run at >2,000/s will not complete either — that is how this
   was rediscovered.

## How this was nearly missed — three methodology notes

**1. An endurance test that stops short of the failure point is a false negative.** The first
probe ran **3,000** iterations, reported "COMPLETED … without failure", and I recorded it as a
non-issue. The same underpowered-test error was then made twice more: a sustained run that
crashed only on the ANE lane became "ANE-specific" when the GPU run had simply not gone far
enough. **When a limit is suspected, run to failure or state the bound you actually tested.**

**2. A harness that does not check exit status will report on a corpse.** The residency sweep that
rediscovered this printed `338 MHz → RESIDENT` for two models whose processes had already died —
the GPU was idle because *nothing was running*. The blank throughput column was the only tell.
**Never derive a verdict from a measurement without first asserting the measured process
survived.**

**3. The unit of a limit is part of the finding.** Recording "~8,000 inferences" instead of
"16,384 allocations" hid a constant behind apparent per-model variance, and made the limit look
un-plannable when it is exactly predictable.
