# Package efficiency checklist (the library-revisit sweep)

What every port — new, and especially the revisit sweep across the accumulated library — does to be a
good memory citizen. Informed by a study of ComfyUI's model management, translated to **Apple unified
memory**. The one big non-transfer: ComfyUI's `LowVramPatch` streams weight modules GPU↔CPU because a
discrete GPU has a scarce separate VRAM pool. On unified memory, "offload to CPU" frees nothing — so we
do **not** port manual module offloading. The Apple-correct wins are below.

## 1. Declare the split footprint (persistent + transient)

Measure and declare BOTH halves (see `memory-harness.md`):
- `QuantFootprint.residentBytes` — persistent weights (the resident floor).
- `QuantFootprint.peakActivationBytes` — the transient activation peak (worstPeak − floor).

The engine reserves a single transient across residents (serialized inference), so an honest split
directly buys co-residency. A `0` transient means the engine can't reserve for you — measure it. For
same-quant modes, declare per-mode via `FootprintConfigured.residentBytesHint` + `peakActivationBytesHint`.

## 2. Load weights mmap/lazy — never force a full eager copy

This is the Apple analog of ComfyUI's partial loading. MLX loads safetensors **lazily** and the OS
mmaps the file, so weights page into unified memory on demand and clean pages can be dropped under
pressure — for free, no module-shuffling machinery.

- **Do** load via MLX's safetensors loading and let lazy eval materialize per-use.
- **Don't** eagerly walk the weight dict casting/copying everything up front (e.g.
  `weights.mapValues { $0.asType(.float32) }` across the whole model, or building a second full dict) —
  that materializes the entire model into a copy and defeats mmap. Cast lazily, at use.
- Verify: after `load()` + `clearCache()`, the resident floor should track the on-disk weight bytes of
  the *selected* variant, not a multiple of it.

## 3. Default to per-stage load → use → evict for multi-component models

A multi-component pipeline (text/encoder → DiT/backbone → VAE/decoder) should hold only the **current
stage** resident, not all stages at once. Generalize the Wan T5 pattern: load encoder → encode → free →
load DiT → denoise → free → load decoder → decode. The peak is then `max(stage)`, which is exactly what
you declare as the footprint — and it's the unified-memory equivalent of "only load the node you need."

- Free a stage with the ref drop + `GPU.clearCache()` before loading the next.
- This makes the declared `peakActivationBytes` honest (it's one stage's peak, not the sum).
- Optional flag to keep a hot stage resident on big-RAM machines (the heavy-tier refinement), but the
  default is evict-between-stages.

## 4. Opt into BudgetAware for memory-adaptive dtype (where it earns it)

If the port has a dtype lever that trades quality for memory (e.g. umT5 fp32→bf16→fp8, fp32-vs-bf16
DiT), conform the configuration to `BudgetAware`. The engine stamps `availableBudgetBytes` (the real
headroom this model is loading into, after eviction) right before `load()`, so `load()` can pick a
lighter dtype when tight — ComfyUI's `unet_dtype`-by-free-memory idea. Don't hardcode the heaviest
dtype if a lighter one is acceptable under pressure; don't override the stamp.

## Sweep order (current)

**LTX first** (Wan is deferred for a dedicated deep-dive), then the most-consumed packages (image
capabilities, Qwen LLM/TTS), then the optimizer family (BiRefNet / Real-ESRGAN / NAFNet / SigLIP2 —
which benefit most from #1 + the engine's single-transient reserve when chained). Wan last.

## Writing the adoption brief (the per-package work order)

The sweep runs one package at a time via a **self-contained `EFFICIENCY-ADOPTION.md`** committed in the
package repo, so a session-specific agent can execute it cold. Use the same shape every time (template:
`ltx-2-mlx-swift/EFFICIENCY-ADOPTION.md`):

1. **At-a-glance** — wrapper/core, capability, component list, why it's a sweep target.
2. **Engine dependency status** — current `Package.swift` pin + resolved version; is a `swift package
   update` to the latest engine enough, or a manifest edit?
3. **Audit vs. the four levers** — a table (lever · state 🟢/🟡/❌ · finding · priority) backed by
   `file:line` evidence.
4. **Prioritized tasks (P0…Pn)** — each with effort, concrete change, and the measured data if it exists.
5. **Already good — don't regress.**
6. **Definition of done** (below) + a validation note.

Keep findings grounded in `file:line`; rank by effort × value; be explicit about deferrals and *why*.

## Definition of done (per package)

- [ ] Split footprint measured + declared (`residentBytes` + `peakActivationBytes`; per-mode hints if needed).
- [ ] Weights load mmap/lazy; resident floor ≈ selected-variant on-disk bytes (no full eager copy).
- [ ] Multi-component: per-stage load→use→evict; declared peak = `max(stage)`.
- [ ] BudgetAware adopted iff the port has a real dtype/quality lever; otherwise skip (don't add ceremony).
- [ ] Re-run the memory harness; commit the measured numbers with the input envelope as a comment.
- [ ] Update the package's **Eff** (and any changed Val/Eng) cell in `mlx-engine-swift/docs/model-registry.md`
      — the registry's Eff column *is* the sweep tracker; keep it current in the same change.
- [ ] **`unload()` must `MLX.Memory.clearCache()`** (not just drop refs). Dropping the model ref leaves its
      activation/weight buffers in MLX's pool, so `phys_footprint` doesn't fall and `engine.evict` / R-MEM-1
      can't reclaim — process RSS then grows monotonically across model switches and OOMs on smaller Macs.
      (Found by the image app's acceptance run; fixed in BiRefNet first — apply to every package's `unload()`.)
- [ ] **Verify the unload by ATTRIBUTION, not by `phys_footprint` alone.** Print all three after
      `unload()`: `Memory.activeMemory`, `Memory.cacheMemory`, and process `phys_footprint`. The
      combination is what tells you which world you're in, and `phys` on its own is *systematically
      misleading* here:
      | MLX active | MLX cache | phys | verdict |
      |---|---|---|---|
      | ~0 | ~0 | still high | **allocator/Metal page retention — NOT a package leak.** Nothing more to fix. |
      | ~0 | high | high | `clearCache()` missing or ineffective — the bullet above |
      | high | — | high | a real retained reference (a cached prefix, a memoized tensor, a closure capture) |
      Audio8-TTS measured active 0 MB / cache 0 MB with ~10 GB of `phys` still resident, i.e. row 1 —
      a clean unload that *looks* exactly like a leak if you only watch RSS. Gate on
      `activeMemory ≈ 0`, and treat residual `phys` as informational. Without this split, a correct
      package gets a leak hunt and a leaking one gets waved through on a lucky-looking RSS graph.

## Gotchas & measurement (validated on the LTX-2.3 run, 2026-06-30)

The first sweep target proved the approach (LTX bf16: engine charge 84 → **40 GB**, peak −31 GB;
activation **dtype-independent** ~12–15 GB across bf16/int8/int4 — the co-residency premise, confirmed).
Three traps cost the most time — pre-empt them:

- **Async stage loaders "send" the non-Sendable pipeline off `@InferenceActor`.** Making the pipeline
  methods `async` (because a stage's loader is async) trips Swift 6 region isolation — the
  `@InferenceActor`-isolated, non-Sendable pipeline can't cross the hop. Fix: have the async methods
  **inherit the caller's isolation** with `isolated (any Actor)? = #isolation` (or `isolated` on the
  pipeline param). This is the canonical per-stage-eviction fix; expect it on any package whose staged
  loaders are async.
- **Two-repo `Package.resolved`.** A package's own `swift package update` (P0) does **not** update the
  consuming **app workspace's** `Package.resolved` — the app keeps resolving the old engine and won't
  compile a 1.14 manifest until you bump the workspace pin too. The workspace `Package.resolved` is
  unversioned (xcodebuild regenerates it), so there's nothing to commit there — just re-resolve the app.
- **Measure via the app autorun, not a CLI bench (for heavy packages).** A full-pipeline CLI mem-bench
  trips the GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`) on heavy video models on this beta
  OS, even with file prewarm — it lacks the engine's `WeightPrewarmer` + governor. Use **component-scale
  CLI gates** (1 step / 256²) for parity, and the **app headless autorun** (engine prewarm + governor) as
  the footprint-measurement surface. Budget ~40 min wall-clock of cold per-quant runs (I/O-dominated off
  external volumes). The split *declaration* is trivial once the numbers land — measurement is the cost.

## Measurement findings worth knowing (from the sweep)

- **Flat footprints can UNDER-declare, not just over-reserve.** A pre-split flat `residentBytes` was often
  set to the *weight* size or a guess, which can sit *below* the real activation peak (NAFNet fp16 0.6 GB
  vs ~2.0 GB measured; Real-ESRGAN 1.0 GB vs ~2.2 GB). So the split is a **correctness** fix as much as an
  efficiency one — measure the peak, don't trust the existing number.
- **Tiled ops are tile-bounded, not input-area-driven.** Real-ESRGAN's 1024² input peaked *below* its
  512² run — activation tracks the tile working set, not the image area. Measure at the tile size that
  dominates, and note the tile size as the activation driver in the brief.
- **Autoregressive (LLM) transient is NOT (just) the KV-cache — measure it, don't derive it.** For
  Qwen3.5 the analytic KV-cache was bit-exact (12,288 B/token, ~96 MB @ 8k) but the *measured* peak was
  **~20× larger and prefill-scratch-dominated** — because it's a **hybrid linear/full-attention** model
  (GatedDeltaNet chunked-scan over the prompt dominates; only 1-in-4 layers grow a context KV-cache).
  Declaring the analytic KV-cache alone would have under-reserved ~20×. So: declare `peakActivationBytes`
  from a **measured** long-prompt run at a documented **maxTokens envelope** (Qwen used 2048 — a realistic
  chat window; 8k prefill scratch was ~7 GB even at 0.8B, pathological). Keep the analytic KV-cache as the
  *persisted* cache size and the basis for any context-cap BudgetAware lever, but never as the transient.
  The transient scales with **context**, not a fixed peak — note the envelope like a resolution envelope.

- **Declare from in-app `phys_footprint`, not a smoke's MLX working-set peak — they differ ~2.7×.**
  BiRefNet's per-package `birefnet-smoke` reported MLX-peak (fast 4.9 / best 18.3 GB); the in-app harness
  reading true process `phys_footprint` showed **fast 13.7 / best 47.8 GB** — the gap is the MLX buffer
  cache + process overhead the MLX-peak metric omits. **The engine (R-MEM-1, admission) compares against
  `phys_footprint`**, so that's the authoritative basis: a smoke-measured `peakActivationBytes` can
  under-declare the real admission cost ~3× (BiRefNet's `best` guard was 2.4× too low → it falsely admitted
  on 32 GB Macs). Prefer the **in-app (or `xcodebuild`-built) phys_footprint** measurement; treat a bare
  `swift`-smoke MLX-peak as a low estimate and re-baseline against phys when an app/autorun exists.
  **Instrument:** read both numbers with the shared **`MLXProfiling`** (`MetalToolBox/PROD/mlx-profiling`,
  `MLX_PROFILE=1`) — its `[MLXPROF]` rows report MLX pool (`active`/`cache`/`peak`) AND OS `phys_footprint`
  side by side, so the ~2.7× gap above is visible in one run, and its ⚠PAGING flag catches an activation
  peak that would OOM on a smaller tier. Use it here (split + evict verification), at the manifest
  measurement (`memory-harness.md`), and for watchdog triage (`swift-port-parity.md`) — one instrument, not
  three.

- **Measure the resident floor POST-LOAD, not post-run — `clearCache()` frees pool buffers but NOT
  referenced arrays.** The in-app harness first read the floor *after the run* (target still resident),
  assuming post-run phys ≈ weights. But a model whose live graph retains run intermediates holds them as
  *referenced* `MLXArray`s, and `MLX.GPU.clearCache()` only reclaims *unreferenced* pool buffers — so the
  post-run floor over-reads (NAFNet + DDColor both stuck at an identical ~5.4 GB while their weights are
  MB-scale) and `activation = peak − floor` collapses toward 0. Fix (MLXEngineTestKit 0.17.0): read the
  floor **post-load, pre-run** (= true weights resident), and read the resident *again* post-run into a
  separate `retainedAfterRunBytes` (`retain=` in the SPLIT line) so post-run retention surfaces as its own
  signal instead of inflating the floor. A non-trivial `retain` that is **flat** across repeated resident
  runs = a bounded working buffer (benign — freed on evict); one that **climbs per request** = a real
  retention leak in the model graph (drop the retained refs in the forward / don't cache intermediates as
  instance state). Tiled forwards (Real-ESRGAN) tend to read ~0 retain; full-res un-tiled forwards (NAFNet)
  retain more. Note: the engine's evict path still frees it all — this is a *measurement* correctness fix
  plus a leak-detector, not an admission bug.

## App-side counterpart

These are package-author tasks. The **app** has its own seams to make them engage (surface the reserve,
pick variants via the activation-aware admissibility, don't fight `BudgetAware`) — see the
`mlxengine-implementation` skill, topic 3 (memory).
