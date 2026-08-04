---
name: mlxengine-implementation
description: How to correctly BUILD AN APP THAT CONSUMES MLXEngine (the Xocialize MLXServeCore/MLXServeEngine coordinator) on Apple Silicon — the consumer side, distinct from porting a model INTO the engine. Use whenever wiring MLXEngine into a SwiftUI/AppKit app: standing up the engine, pointing it at the shared model store / weight downloader via a security-scoped folder pick (so downloads land in one sandbox-friendly location instead of each package's private cache), first-run auto-materialization (dir-less WeightSourcing configs, needsDownload routing, the MAT gate), surfacing download/first-load progress, reusing the provided settings UI (EngineSettingsView / ModelStorageSettingsView / ModelStateView), the register → prepare → run → evict lifecycle from the app's side, memory-budget/admissibility wiring, capability routing & multi-package selection (PackageID), sandbox entitlements, and the offline-vs-live metallib boundary. Trigger whenever someone says the app "skips the downloader", "downloads weights twice", "models don't show in settings", "where do weights go", "integrate MLXEngine into my app", "engine added but not used right", the app's "memory keeps growing every turn / looks like a leak / never releases" (the MLX buffer-pool ratchet — engine ≥0.21.0 bounds it by default), or is building any app on top of mlx-engine-swift. This is a LIVING skill — extend it as new consumer-side gaps surface. Complements `mlx-swift-integration` (which builds packages INTO the engine); do NOT use this for porting models or authoring a ModelPackage.
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

## When you land a change IN `mlx-engine-swift` itself — install the push guard

Most consumer work never touches the engine repo. But app work regularly *does* reach into it — a
contract bump because the app needs a capability or a request field, a registry row for a package the
app is first to drive, a `Val 🟡 → ✅` flip after in-app validation (which the registry expects **in the
same change** as the validation), or a docs fix. The moment you push there, one gate applies:

```bash
git config core.hooksPath Scripts/git-hooks   # once per engine clone
```

It runs the docs gates before anything reaches origin (`--no-verify` bypasses when you mean it). The
CI job is `status-block-lint`, and it asserts the README status block against source and tags:
capability count vs `Capability`'s cases, contract version vs `ContractVersion.current`, and the
quoted tag vs the newest reachable from HEAD.

⚠️ **Why it needs a hook at all.** Engine work lands by *direct push to main*, so the workflow's
`pull_request` trigger never fires and the push trigger only reports after the fact. In 2026-08 the
lint sat red for **8 pushes across 2.5 days and three tagged releases** before anyone acted. Branch
protection is deliberately not set (it would block that same workflow), so the hook is the only thing
in the path.

**The consumer-side trap specifically:** an app validates a package in-app and flips its `Val` flag,
or is the first consumer of a brand-new capability — and the registry row is the step that gets
dropped, because the app-side work *feels* finished when the app works. A missing row is not caught
directly; it shows up as the capability count going stale. Add the row first, then recount the
package numbers **from the registry** rather than incrementing them — and note the lint deliberately
does not check those two counts, so green CI ≠ a current status block.

## Topic roadmap (living skill)

Extend this as consumer-side gaps surface. Each topic, once written, gets a `references/<topic>.md`.

| # | Topic | Status |
|---|---|---|
| 1 | **Model store & weight downloader** (shared sandbox-friendly location via security-scoped folder pick; flat `models--org--name` layout since contract 1.22 — migrate hand-staged stores by rename; hardlinks-not-symlinks for sandbox staging) | ✅ [references/model-store.md](references/model-store.md) |
| 2 | **Engine lifecycle & ownership** in an app (who holds it, `@InferenceActor`, register → prepare → run → evict) | ✅ [references/lifecycle.md](references/lifecycle.md) |
| 3 | **Memory budget & admissibility** from the app's side (budget fraction, `admissibility` surveys, footprint hints, R-MEM-1 real-pressure) | ✅ [references/memory.md](references/memory.md) |
| 4 | **Capability routing & multi-package selection** (`PackageID`, `setDefault`, per-request backer; MLXVLM link-shadowing pitfall when combining packages) | ✅ [references/routing.md](references/routing.md) |
| 5 | **Progress & error UX** (`PreparationMonitor` phases, `EngineError` handling, license/eligibility gates) | ✅ [references/progress-and-errors.md](references/progress-and-errors.md) |
| 6 | **Sandbox & entitlements** checklist; offline-CLI vs live-Xcode metallib boundary | ✅ [references/sandbox-and-metallib.md](references/sandbox-and-metallib.md) |
| 7 | **Smoke-testing** an integration in-app | ✅ [references/smoke-testing.md](references/smoke-testing.md) |
| 8 | **First-run weight materialization** (engine ≥ 0.19.0: dir-less `WeightSourcing` configs auto-download via the store; `needsDownload` routing, multi-source progress semantics, MAT gate + `MaterializationBench` measurement, first-run pitfalls) | ✅ [references/materialization.md](references/materialization.md) |
| 9 | **GPU buffer cache & runtime footprint** (stepwise never-released growth per interaction = unbounded MLX buffer pool; engine ≥ 0.21.0 bounds it BY DEFAULT at `MLXServeEngine` construction — `GPUCacheConfiguration`, `trimCaches()`, `gpuPoolSnapshot()` telemetry — so DELETE app-side `GPU.set(cacheLimit:)` workarounds there; the app-init write remains only for engines < 0.21.0; phys vs active/cache/peak diagnostics, leak-vs-pool triage) | ✅ [references/gpu-cache-and-footprint.md](references/gpu-cache-and-footprint.md) |
| 10 | **Field issue log** — consumer-side catches & resolutions from real apps (capability-overlap routing, quant-suffixed repos vs staged weights, pre-MAT packages, settings-rebind hook, model-tier prompt adherence; Mage Demo: un-wired materialization executors, sibling-package co-residency doubling the conditioner, in-app [MAT]/[RUN] metrics collection; MageVL Demo: UI controls bound to local state when the value is package CONFIGURATION, stale peakActivationBytes silently shrinking admissibility, measuring a sandboxed app headlessly; two engine instances — one shown, one driven — mimicking a permissions failure; in-process sweeps under-reading activation; a bench that skipped a production step reproducing a stale footprint forever; Media Upscale: headless sandbox denial's progressively-flaky signature + the `ENABLE_APP_SANDBOX=NO` scratch-DerivedData build, and forcing Xcode to pick up a just-pushed tag). **Append an entry in the same change that fixes a catch.** | ✅ [references/field-issues.md](references/field-issues.md) |
| 11 | **Realtime-audio apps** (mic in / speech out): the Swift 6 `installTap` isolation trap that SIGTRAPs on the first buffer, half-duplex turn-taking + the playback drain signal, energy VAD that doesn't fight the user, `@InferenceActor` serialization across STT→LLM→TTS, `[TURN]` latency phases, and the code-only AppKit shell (weak delegate, activation policy, Spaces, mic entitlements) | ✅ [references/realtime-audio-apps.md](references/realtime-audio-apps.md) |

> When you add a topic, write the deep guidance in a `references/` file and add a one-row pointer here —
> keep this SKILL.md the lean hub. Source of truth for the engine API is `mlx-engine-swift/` itself
> (`Sources/MLXServeCore`, `Sources/MLXToolKit`, `Sources/MLXEngineUI`); verify against it, since this
> skill describes a moving target.

## Start here

For the first and most common gap — apps skipping the provided downloader / model store — read
**[references/model-store.md](references/model-store.md)**. It covers why the shared store exists (the
sandbox folder-grant trick), the exact wiring, the entitlements, and the anti-pattern to recognize.
