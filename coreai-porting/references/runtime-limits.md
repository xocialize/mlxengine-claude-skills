# Runtime limits — the ~8,000-inference ceiling

**MEASURED 2026-08-30.** macOS 27.0 (26A5421a), M5 Max 128 GB, `CoreAIRuntime 3600.83.2.14.1`,
`ANEServices 10.19`, `coreai-core==1.0.0b2`.

> ## A CoreAI process dies after roughly 8,000 inferences.
>
> Not a slowdown, not an exception — an **uncatchable** Swift precondition
> (`EXC_BREAKPOINT` / SIGTRAP). Filed as **`apple/coreai-torch#75`**.

This is the single most important operational fact we have found. It bounds every design that
assumes a long-lived inference process.

---

## The failure

```
CoreAIRuntime/NDArray+SharedStorage.swift:108: Fatal error: Failed to allocate storage for
NDArray with requirements: NDArrayDescriptor(scalarType: float16, shape: [1, 20, 56, 56],
alignments: [1, 1, 1, 32, 1], …, storageKind: ioSurface)
```

```
CoreAIRuntime/NDArray+Pool.swift:77: Fatal error: Failed to allocate storage for NDArray
with byteCount: 1572864, sk: ioSurface, st: float16
```

Two sites, depending on output shape; same cause. Crash thread is always
`libswiftCore _assertionFailure` → `CoreAIRuntime` ×6 → `_coreai_runtime_os` →
`completeTaskWithClosure`.

## What it is NOT — all measured, not assumed

| Hypothesis | Verdict |
|---|---|
| ANE-specific | **No.** The GPU lane dies at the same count. |
| Caller retaining results | **No.** `del` and `gc.collect()` every 100 iterations change nothing. |
| Per-`AIModel` accumulation | **No.** Reloading every 2,000 calls dies at the same point — **and makes it worse** (below). |
| Memory pressure | **No.** ~308 MB writable at death on a 128 GB machine. |
| Model-specific | **No.** Two unrelated models, one and two outputs. |
| System-wide | **No.** Per-process; a fresh process gets a fresh ~8,000. |

Everything points at IOSurface **handles**, not bytes.

## Reloading the model is a trap

The obvious mitigation — periodically rebuild the model to drain the pool — is strictly worse.
MEASURED from the crash report of a reload-every-2,000 run: **five live `ANEServicesThread`s**,
one per reload, none reaped, and `Neural Engine (reserved) 35.6M across 5 regions` against
`7296K / 1 region` for a single-load run. It still hits the same ceiling, and leaks a thread and
an NE region each time.

## What this means for design

1. **A long-lived CoreAI inference service is not currently viable** on this runtime version. At
   ~1 ms/inference that is under 10 seconds of continuous work.
2. **The only workaround we have is process recycling** — shard across subprocesses and restart
   before ~8,000 calls. Viable for batch; not for a latency-sensitive or stateful service.
3. **This belongs in the fit journal.** For a workload doing sustained high-rate inference, the
   ANE's energy advantage is moot if the process cannot stay alive — MLX has no equivalent
   ceiling. → `mlx-vs-coreai-fit.md`.
4. **Budget it into any benchmark harness.** A 20,000-iteration sweep will not complete.

## How this was nearly missed — a methodology note

The first probe ran **3,000** iterations, reported "COMPLETED … without failure", and I recorded
it as a non-issue. The threshold is ~8,000. **An endurance test that stops short of the failure
point is a false negative**, and it read as a clean result.

Worse, the same underpowered-test error was made twice in a row: the first sustained run crashed
only on the ANE lane, and I wrote "ANE-specific" — when the GPU run had simply not gone far
enough. **When a limit is suspected, run to failure or state the bound you actually tested.**
