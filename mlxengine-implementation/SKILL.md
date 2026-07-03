---
name: mlxengine-implementation
description: How to correctly BUILD AN APP THAT CONSUMES MLXEngine (the Xocialize MLXServeCore/MLXServeEngine coordinator) on Apple Silicon — the consumer side, distinct from porting a model INTO the engine. Use whenever wiring MLXEngine into a SwiftUI/AppKit app: standing up the engine, pointing it at the shared model store / weight downloader via a security-scoped folder pick (so downloads land in one sandbox-friendly location instead of each package's private cache), first-run auto-materialization (dir-less WeightSourcing configs, needsDownload routing, the MAT gate), surfacing download/first-load progress, reusing the provided settings UI (EngineSettingsView / ModelStorageSettingsView / ModelStateView), the register → prepare → run → evict lifecycle from the app's side, memory-budget/admissibility wiring, capability routing & multi-package selection (PackageID), sandbox entitlements, and the offline-vs-live metallib boundary. Trigger whenever someone says the app "skips the downloader", "downloads weights twice", "models don't show in settings", "where do weights go", "integrate MLXEngine into my app", "engine added but not used right", or is building any app on top of mlx-engine-swift. This is a LIVING skill — extend it as new consumer-side gaps surface. Complements `mlx-swift-integration` (which builds packages INTO the engine); do NOT use this for porting models or authoring a ModelPackage.
---

# Implementing MLXEngine in a consuming app

This is the **consumer-side** companion to `mlx-swift-integration`. That skill is for people building a
`ModelPackage` to plug *into* the engine. This skill is for people building an **app that drives the
engine** — wiring it up so it behaves the way the engine was designed to behave.

```
Build a package INTO the engine ──► mlx-swift-integration   (ModelPackage, manifest, C0–C13, parity)
Build an APP that USES the engine ─► THIS SKILL              (engine lifecycle, model store, UI, sandbox)
```

The recurring failure mode this skill exists to prevent: an app adds the `mlx-engine-swift` SPM, calls
`register`/`run`, and **silently skips the engine's coordination layer** — the shared model store, the
progress affordance, the memory governor's signals, the settings UI. It "works" in the demo and then
re-downloads multi-GB weights per app, can't see its own models in a sandbox, and shows a dead spinner
on first launch. The engine already solves all of that; the app just has to opt in.

## The golden path — what every consuming app wires (in this order)

The order matters. Each step depends on the one before it.

1. **Own one engine instance** for the app's lifetime (a `@MainActor` app model holds it). Build it with
   a real memory budget: `MLXServeEngine(governor: .forDevice(.current(), fraction: 0.7))`.
2. **Stand up the model store BEFORE registering anything.** Create a `ModelStorageModel` (from
   `MLXEngineUI`), let the user pick a models folder once, and call
   `engine.useModelStore(ModelStore(root: storage.resolvedModelsDirectory))`. `useModelStore` only
   affects packages registered *after* it — so this happens before step 3. **This is the step apps skip;
   see [references/model-store.md](references/model-store.md) for the full why + wiring.**
3. **Register packages** (their `ModelStorable` configs get the store root stamped automatically).
4. **Surface preparation** by binding `MLXEngineUI.ModelStateView(monitor: engine.preparation,
   capability: .textToImage)` so the user sees `downloading (fraction) → loading → ready` instead of an
   indeterminate spinner.
5. **`prepare` before first `run`** so the heavy download/load happens behind the progress UI, not inside
   the user's first request.
6. **Reuse the settings UI** — `EngineSettingsView(storage:)` already gives you the storage panel (disk
   used / models installed / free space) and web-retrieval settings. Don't rebuild it.

## Topic roadmap (living skill)

Extend this as consumer-side gaps surface. Each topic, once written, gets a `references/<topic>.md`.

| # | Topic | Status |
|---|---|---|
| 1 | **Model store & weight downloader** (shared sandbox-friendly location via security-scoped folder pick) | ✅ [references/model-store.md](references/model-store.md) |
| 2 | **Engine lifecycle & ownership** in an app (who holds it, `@InferenceActor`, register → prepare → run → evict) | ✅ [references/lifecycle.md](references/lifecycle.md) |
| 3 | **Memory budget & admissibility** from the app's side (budget fraction, `admissibility` surveys, footprint hints, R-MEM-1 real-pressure) | ✅ [references/memory.md](references/memory.md) |
| 4 | **Capability routing & multi-package selection** (`PackageID`, `setDefault`, per-request backer; MLXVLM link-shadowing pitfall when combining packages) | ✅ [references/routing.md](references/routing.md) |
| 5 | **Progress & error UX** (`PreparationMonitor` phases, `EngineError` handling, license/eligibility gates) | ✅ [references/progress-and-errors.md](references/progress-and-errors.md) |
| 6 | **Sandbox & entitlements** checklist; offline-CLI vs live-Xcode metallib boundary | ✅ [references/sandbox-and-metallib.md](references/sandbox-and-metallib.md) |
| 7 | **Smoke-testing** an integration in-app | ✅ [references/smoke-testing.md](references/smoke-testing.md) |
| 8 | **First-run weight materialization** (engine ≥ 0.19.0: dir-less `WeightSourcing` configs auto-download via the store; `needsDownload` routing, multi-source progress semantics, MAT gate + `MaterializationBench` measurement, first-run pitfalls) | ✅ [references/materialization.md](references/materialization.md) |

> When you add a topic, write the deep guidance in a `references/` file and add a one-row pointer here —
> keep this SKILL.md the lean hub. Source of truth for the engine API is `mlx-engine-swift/` itself
> (`Sources/MLXServeCore`, `Sources/MLXToolKit`, `Sources/MLXEngineUI`); verify against it, since this
> skill describes a moving target.

## Start here

For the first and most common gap — apps skipping the provided downloader / model store — read
**[references/model-store.md](references/model-store.md)**. It covers why the shared store exists (the
sandbox folder-grant trick), the exact wiring, the entitlements, and the anti-pattern to recognize.
