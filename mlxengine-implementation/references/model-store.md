# Topic 1 — The model store & weight downloader

## The gap this fixes

A consuming app adds `mlx-engine-swift`, registers a package, calls `run`, and it works — but every
package downloaded its weights into **its own native downloader's default cache** (mlx-swift-lm,
mlx-audio, swift-transformers each have one), not a place the app controls. Consequences:

- **Re-downloads.** Two apps (or the same app re-installed / sandbox-relocated) each pull the same
  multi-GB weights again. Nothing is shared.
- **Sandbox blindness.** Under the App Sandbox a package's default cache is inside the app *container*
  (or an unwritable `~/.cache`), so models silently land somewhere the user can't point other tools at,
  and the storage panel shows nothing.
- **Dead first-run UX.** Without the progress affordance the first `run` blocks on a silent multi-GB
  download.

The engine already solves this with the **model store**. The app's whole job is to (a) let the user pick
one folder and (b) hand that folder to the engine *before registering packages*. Everything else —
stamping the path onto each package, writing the per-model marker, computing disk-used — the engine does.

## Why a user-picked folder (the sandbox folder-grant trick)

This is the key insight that makes the design worth adopting, not a nuisance step:

> In a sandboxed macOS app, the user choosing a folder in an `NSOpenPanel` **grants the app
> security-scoped access to that folder and its entire subtree** — including locations *outside* the
> app container (an external volume, a shared `~/Models`, etc.). Persisted as an **app-scope bookmark**,
> that grant survives relaunch. So one folder pick buys a **shared, app-external, sandbox-legal** models
> location with no broad filesystem entitlement.

That's why the engine routes downloads through a *user-selected* folder instead of just using a default
cache: the selection is simultaneously the "where" and the "permission to write there outside the
sandbox." Skipping it throws that away.

`MLXEngineUI.ModelStorageModel` implements this end to end (folder pick → `startAccessingSecurityScopedResource`
→ `bookmarkData(options: .withSecurityScope)` → `UserDefaults` → resolve-on-init). The app does not
hand-roll bookmarks.

## The pieces (what's already provided)

| Type | Module | Role |
|---|---|---|
| `ModelStore(root: URL?)` | `MLXToolKit` | Foundation-only store root + `mlx-package.json` marker. No HF dependency. |
| `MLXServeEngine.useModelStore(_:)` | `MLXServeCore` | Point the engine at a store root. **Affects packages registered AFTER the call.** Stamps `ModelStorable.modelsRootDirectory` onto each config and writes the marker after a successful `load()`. |
| `ModelStorable { var modelsRootDirectory: URL? }` | `MLXToolKit` | A package config opts in; the engine sets it from the store root. (Configs that don't conform fall back to the default cache — that's the silent-skip case.) |
| `ModelStorageModel` | `MLXEngineUI` | `@Observable` security-scoped bookmark manager: `chooseFolder()`, `apply()`, `reset()`, `resolvedModelsDirectory`, `refresh()`, `status`. Resolves a saved bookmark on `init` (survives relaunch). |
| `ModelStorageSettingsView` / `EngineSettingsView(storage:)` | `MLXEngineUI` | Drop-in settings UI: pick folder, Apply/Reset, disk-used / models-installed / free-space. |
| `ModelStateView(monitor:)` | `MLXEngineUI` | Binds to `engine.preparation` (`PreparationMonitor`) → shows `downloading(fraction) → loading → ready`. |

## Wiring (the canonical flow)

```swift
import MLXServeCore
import MLXToolKit
import MLXEngineUI

@MainActor
final class AppEngine {
    let engine = MLXServeEngine(governor: .forDevice(.current(), fraction: 0.7))
    let storage = ModelStorageModel()          // resolves a saved bookmark on init (survives relaunch)

    /// Call ONCE at startup, and again whenever the user applies a new folder — BEFORE registering.
    /// `useModelStore` is actor-isolated on the engine, so it's awaited.
    func bindModelStore() async {
        await engine.useModelStore(ModelStore(root: storage.resolvedModelsDirectory))
    }

    func registerEverything() async throws {
        await bindModelStore()                  // store first…
        try await engine.register(MyPackage.registration, configuration: MyConfig()) // …then packages
    }
}
```

UI side — reuse the provided views; don't rebuild them:

```swift
// Settings: folder picker + Apply/Reset + disk metrics (+ web retrieval).
EngineSettingsView(storage: appEngine.storage)

// First-load/download affordance anywhere a model is about to be used (capability is required).
ModelStateView(monitor: appEngine.engine.preparation, capability: .textToImage, title: "My Model")
```

Then `prepare` behind the progress UI before the first `run`:

```swift
try await engine.prepare(.textToImage)   // download + load happen here, under ModelStateView
let resp = try await engine.run(myRequest)
```

After a model materializes, call `storage.refresh()` so Disk Used / Models Installed update.

## Ordering rules that bite if you get them wrong

- **`useModelStore` before `register`.** It only stamps packages registered *after* it. Register-then-set
  leaves those packages on their default cache. If the user changes the folder later, re-`useModelStore`
  and re-register (or restart) — already-resident packages keep the old root until evicted.
- **Hold the `ModelStorageModel` alive.** The security-scoped access lives on that instance
  (`accessedURL`). If it deinits, the grant drops and writes under the root start failing. Keep it on the
  long-lived app model, not a transient view.
- **Pass `resolvedModelsDirectory`, not `appliedPath`.** The resolved URL carries the active access
  scope; the string path doesn't. A `nil` resolved directory means no folder chosen yet → the store is
  empty and packages use their default cache (acceptable fallback, but the panel won't track them).
- **Absolute out-of-store snapshot paths vs. the stamped root.** The engine stamps
  `modelsRootDirectory` (the store root) onto *every* `ModelStorable` config at register. A package that
  also exposes an explicit **absolute** local snapshot path (`snapshotPath`, for staged-on-disk weights —
  NC/gated weights, a hand-placed `dist/` folder) must have its `load()` **honor the absolute path over
  the stamped root**. The trap: a naive `modelsRootDirectory.appending(snapshotPath)` prepends the store
  root to an already-absolute path → broken resolution. The safe shapes are either resolve the
  `snapshotPath` directly (`URL(fileURLWithPath:)`, ignoring the stamp when it's absolute — Qwen-Image-Edit,
  ERNIE) or branch on `snapshotPath.hasPrefix("/")` and prefer it (Anima's fix, shipped v0.1.1). This is a
  *package*-side resolution bug, but it only surfaces once an app sets a store — so verify it when wiring
  any absolute-snapshot package into a store-backed app; flag a broken one back to `mlx-swift-integration`.

## Entitlements (sandboxed apps)

Add to the app's `.entitlements` — without these the bookmark can't be created/resolved and the grant
won't persist (the picker still works for the session, but every relaunch re-prompts):

```xml
<key>com.apple.security.files.user-selected.read-write</key> <true/>
<key>com.apple.security.files.bookmarks.app-scope</key>      <true/>
```

## Recognizing the anti-pattern (review checklist)

When an app "skips the downloader," you'll see one or more of:

- [ ] No `ModelStorageModel` anywhere; no `engine.useModelStore(...)` call.
- [ ] `register(...)` called before any `useModelStore(...)` (or it's never called).
- [ ] The app builds its own `NSOpenPanel` / bookmark code instead of reusing `ModelStorageModel`.
- [ ] No `ModelStateView` — a custom spinner (or nothing) during first load.
- [ ] Settings has a hand-rolled storage panel instead of `EngineSettingsView` / `ModelStorageSettingsView`.
- [ ] Missing the two sandbox entitlements above.
- [ ] Package configs don't conform to `ModelStorable`, so even with a store set the root is never applied
      (this one is on the *package*, not the app — flag it back to `mlx-swift-integration`).

The fix for the app side is always the same short list: own one engine, stand up `ModelStorageModel`,
`useModelStore` before `register`, bind `ModelStateView`, reuse `EngineSettingsView`, add the entitlements.

## Verify against source

This describes a moving target. Ground-truth files:
`mlx-engine-swift/Sources/MLXToolKit/ModelStore.swift`,
`Sources/MLXServeCore/MLXServeEngine.swift` (`useModelStore`, the register-time stamp + marker write),
`Sources/MLXToolKit/Preparation.swift` (`PreparationMonitor`, `PreparePhase`),
`Sources/MLXEngineUI/{ModelStorageSettingsView,EngineSettingsView,ModelStateView}.swift`.
