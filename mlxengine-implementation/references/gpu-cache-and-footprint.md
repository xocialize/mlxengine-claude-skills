# GPU buffer cache & runtime footprint (consumer side)

## The symptom

The app's memory (Activity Monitor / Xcode gauge) **grows stepwise with every
inference interaction and is never released** — e.g. a chat companion stepping
+GBs per turn to 40+ GB. It looks like a leak; it is not. Instruments shows no
Swift/ObjC leak, because the "leak" is MLX's Metal buffer-recycling pool.

## Why it happens

MLX recycles every freed GPU buffer into a cache pool for reuse. **By default
the pool is effectively unbounded and never returns memory to the OS.** Any app
that runs varied workloads keeps ratcheting the pool because each run allocates
new buffer *shapes* that don't match the pooled ones:

- LLM turns with growing context (every prefill is a new shape; vocab-sized
  logit buffers are ~hundreds of MB each),
- packages that run each request on a fresh session (new KV cache per turn),
- multiple capabilities per interaction (llm + embed + tts each with their own
  tensor shapes).

**RESOLVED at engine ≥ 0.21.0 (2026-07-05, N5):** `MLXServeEngine` now owns this
policy. Constructing an engine applies a bounded `MLX.Memory.cacheLimit` by
default (`GPUCacheConfiguration.automatic` = min(2 GB, 5% of the governor
budget)); `.bytes(_)` fixes the cap, `.unmanaged` opts out. The engine also
exposes `trimCaches()`, optional `trimAfterEvict`/`trimEveryRuns` knobs, and
`gpuPoolSnapshot()` telemetry (active/cache/peak/effective limit) so apps need
no MLX import for observability. Precedence: last-write-wins on the
process-global setting — a host writing `cacheLimit` after engine construction
overrides the engine; a pre-construction host write is superseded (use
`.unmanaged` to keep it). **On engines ≥ 0.21.0, DELETE any app-side
`MLX.GPU.set(cacheLimit:)` workaround** — it's redundant (before engine init)
or an intentional override (after). Spec: `mlx-engine-swift/docs/architecture.md`
R-MEM-2.

## The consumer-side fix (only for engines < 0.21.0)

Bound the pool once, early (before the first `prepare`):

```swift
import MLX

// Keeps hot-path buffer reuse; everything beyond the cap returns to the OS.
MLX.GPU.set(cacheLimit: 2 * 1024 * 1024 * 1024)   // 2 GB is a good default
```

Rules of thumb (they also guide `GPUCacheConfiguration.bytes(_)` sizing):
- 2 GB suits chat-scale apps (LLM ≤ 9B + TTS + embeddings). Media apps doing
  video/diffusion may want more — size it to a fraction of the *transient*
  working set, not the weights (weights are `active`, not `cache`).
- `cacheLimit: 0` disables recycling entirely — correct for one-shot batch
  tools, too slow for interactive loops.
- For a burst you never repeat (e.g. one huge decode), `MLX.GPU.clearCache()`
  afterwards drops the pool immediately.

## Verifying (and telling MLX growth from a real leak)

Log this after each interaction (cheap):

```swift
let phys: Int64 = { /* task_info(TASK_VM_INFO).phys_footprint */ }()
GPU.activeMemory   // live tensors (weights + in-flight)
GPU.cacheMemory    // the recycling pool — should saturate at cacheLimit
GPU.peakMemory     // high-water mark
```

Interpretation:
- `phys ≈ baseline + active + cache`. After the fix, `cache` saturates at the
  limit and `phys` plateaus.
- `cache` flat but `phys` climbing → NOT MLX tensors; look at app-side
  retention, other allocators, or (RealityKit hosts) per-frame component
  reconstruction (`SkeletalPose` rebuild leaks — mutate `jointTransforms` in
  place instead).
- `active` climbing across turns → something retains tensors (a package
  keeping sessions/KV alive; an app array of `MLXArray`s). That IS a leak.

A working reference: MLXCompanion's `_Logging/MemoryDiagnostics.swift`
(phys_footprint + the three GPU numbers, one os_log line per turn phase).

## Related

- The governor's budget (`.forDevice(_, fraction:)`) governs *admission* of
  package weights — it does not constrain the buffer pool. On a 128 GB box a
  0.7 fraction leaves the pool ~90 GB of headroom to ratchet through.
- Package-level follow-up (same N5 entry): per-request fresh sessions re-prefill
  the whole history each turn; prompt/KV cache reuse would cut latency and the
  transient peak, but needs an explicit lifetime policy or it becomes the
  retention bug this page is about.
