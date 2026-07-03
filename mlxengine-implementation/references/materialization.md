# First-run weight materialization (auto-materialize, engine ≥ 0.19.0)

The app's side of "the user picked a models folder and pressed Generate on a fresh machine."
Engine v0.19.0 made first-run downloads a **package responsibility with an engine-level contract**
— the app never hand-downloads weights, never knows repo ids, and never threads directories.

## The contract (who does what)

| Piece | Owner | What it does |
|---|---|---|
| `ModelStorable` | engine stamps it | WHERE weights live: the store root the app chose |
| `WeightSourcing` (v0.19.0) | package config declares | WHAT would be fetched: `WeightSource{role, repo, revision, globs}` + `missingWeightSources(storeRoot:)` |
| `load()` | package executes | downloads missing sources into the store layout (`<root>/<org>/<name>/…`), forwarding progress via `WeightDownloadProgress` |
| folder pick, progress UI, routing | **the app (you)** | everything below |

## The golden first-run flow

```swift
// 1. Store BEFORE registering (see model-store.md — the step apps skip).
await engine.useModelStore(ModelStore(root: storage.resolvedModelsDirectory))

// 2. Register a DIR-LESS configuration. No paths: quant/profile only.
//    (Explicit directories remain the dev-mode escape hatch — they never touch the network.)
let id = try await engine.register(MLXLTX2Package.registration,
                                   configuration: LTX2Configuration(quant: .int8, profile: .standard64))

// 3. Route the user: will prepare() download?
if engine.needsDownload(.textToVideo) {
    // show the download consent / storage UI first (sizes, disk space, license)
}

// 4. prepare() behind the progress affordance — the monitor surfaces the REAL phases:
//    .downloading(fraction:bytesPerSecond:) → .prewarming → .loading → .ready
try await engine.prepare(.textToVideo, package: id)
// ModelStateView(monitor: engine.preparation, capability: .textToVideo) renders all of it.
```

Notes on the phases:
- **Multi-source packages are normal** (LTX materializes three: components + text-encoder +
  quantized transformer). The fraction is monotonic across the whole materialization — source
  *i* of *n* spans `[i/n, (i+1)/n)` — so one progress bar is correct; don't build per-repo bars.
- `needsDownload` is a heuristic (prewarm paths exist → no; else marker-absent → yes). Bundled
  packages read `true` pre-marker but skip `.downloading`; treat it as "route to download UI,"
  not as a byte estimate.
- Quant-aware downloads: a conforming package's declaration EXCLUDES files that quant doesn't
  need (LTX's int8/int4 configs skip the 35 GB bf16 transformer). If a first run pulls
  obviously-too-much, that's a package declaration bug — file it, don't work around it.

## Verifying a package's first-run (the MAT gate)

Before an app trusts a package's fresh-machine behavior:

1. **Offline conformance** (package author's suite runs it; you can too):
   `MLXServeConformance.MaterializationConformance.check(freshConfiguration:satisfiedConfiguration:)`
   — MAT-1 ModelStorable · MAT-2 declares sources · MAT-3 role/repo hygiene · MAT-4 honest
   fresh-machine missing set · MAT-5 explicit paths satisfy. No network. (Package authors: this is
   a `mlx-swift-integration` conformance-gate item now.)
2. **Live measurement** from the area testing app (`MLXEngineTestKit`):
   ```swift
   let run = try await MaterializationBench.run(engine: engine, capability: .textToVideo,
                                                package: id, configuration: cfg,
                                                sourceRepo: manifest.provenance.sourceRepo,
                                                storeRoot: emptyFolder)
   print(run.logLine)   // "[MAT] pkg=… bytes=… secs=… downloadPhase=yes events=… marker=yes peak=…"
   ```
   `downloadPhase=NO` or `events=0` on a run that clearly downloaded ⇒ the package downloads
   **silently** (not forwarding `WeightDownloadProgress`) — a conformance smell; users would see a
   dead spinner. The `[MAT]` line is the registry's Val evidence for first-run behavior.

## Pitfalls

- **Registering before `useModelStore`** — the config never gets the root stamped; a dir-less
  package then throws (good ones) or downloads into its private cache (the classic
  duplicate-weights failure). Order is golden-path step 2 for a reason.
- **A dir-less config with NO store** — conforming packages throw a configuration error naming
  both options (explicit dirs or a store). If you see it, you skipped the folder pick.
- **Testing "first run" against a non-empty folder** — `missingWeightSources` probes the store
  layout; leftovers from another app run make the test lie. Use a genuinely empty temp folder
  (that's what `MaterializationBench` is for).
- **Gated HF repos** — package downloaders use environment token detection; a gated source needs
  `HF_TOKEN` (or the HF CLI token file) present in the app's environment, and the account must
  have accepted the repo terms once on the website.
- **First-launch prewarm is a no-op** — nothing on disk yet; the engine's `WeightPrewarmer` pays
  off from the SECOND cold launch (conforming packages resolve `prewarmPaths` against the store
  layout). Don't chase "prewarm did nothing" on run one.

Cross-refs: [model-store.md](model-store.md) (the folder-grant trick), [progress-and-errors.md](progress-and-errors.md)
(phase → UI mapping). First conforming package + reference implementation: `MLXLTX2`
(`LTX2Configuration` WeightSourcing + `WeightMaterializer`, ltx-2-mlx-swift `7ae7aed`).
