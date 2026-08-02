# Topic 2 — Engine lifecycle & ownership in an app

## The mental model: the engine is the coordinator, not a model

`MLXServeEngine` is an `actor` that owns the *coordination* around models — the resident set, the memory
governor, capability routing, the model store, and the preparation monitor. Packages do inference; the
engine instantiates, holds, drives, and evicts them. This **inversion of control** (C13) is the whole
value proposition: the app never constructs or calls a `ModelPackage` directly — it hands registrations
to the engine and talks to the engine.

Two consequences shape every app:

1. **There is exactly one engine.** Two engines = two memory governors that don't know about each
   other, duplicate residency, and a split model store. The budget only means something if one governor
   sees all the residents.
2. **You don't manage threads.** Inference is serialized on the `@InferenceActor` global actor;
   `ModelPackage` lifecycle methods are isolated to it. The app just `await`s the engine's async API.

## Ownership — where the engine lives

Hold the engine (and the `ModelStorageModel` from topic 1) on a **long-lived `@MainActor` app model**,
injected into the view tree — never in a transient view's `@State`, and never re-created per request.

```swift
@MainActor @Observable
final class AppEngine {
    let engine = MLXServeEngine(governor: .forDevice(.current(), fraction: 0.7))
    let storage = ModelStorageModel()

    func bootstrap() async throws {
        await engine.useModelStore(ModelStore(root: storage.resolvedModelsDirectory)) // topic 1: store FIRST
        try await registerAll()                                                        // then register
    }
}
```

```swift
@main struct MyApp: App {
    @State private var app = AppEngine()
    var body: some Scene {
        WindowGroup { RootView().environment(app) }
            .task { try? await app.bootstrap() }
    }
}
```

**⚠️ Own the engine on `@MainActor` — but do NOT drive generation from a `@MainActor` task.**
Field-proven pitfall (LTX proving ground, 2026-07-02): a run flow living on the view model's
`@MainActor` works in normal use (inference itself hops to `@InferenceActor`), but during **app
termination the main-actor executor is starved** — a cancelled main-actor task can never resume to
finish its teardown, so a graceful quit-drain hangs regardless of cancellation checkpoints, and
`.terminateLater` deadlocks outright (the modal terminate run-loop starves the main queue too).
The app was forced into a bounded-drain + `_exit(0)` backstop. Run generation in a detached or
non-main-actor task (`Task.detached { try await engine.run(...) }`, hop back to `@MainActor` only
to publish UI state); keep a bounded-drain + `_exit` backstop in `applicationShouldTerminate`
anyway — it's the right insurance for any phase that can't unwind in time.

The engine's `init` is synchronous and `nonisolated`, so it's fine as a stored property. Its
`preparation` monitor is a `nonisolated let` — bind UI to it directly without `await`.

## The lifecycle: register → prepare → run → evict

### register — cheap, eager, at startup
`register(_:configuration:id:)` runs the **two-layer license gate** and the **C10 device-eligibility**
check *immediately* (before any instance exists), records the registration, and **defers construction**.
It does *not* load weights. So register everything you might use at startup — it's just gating +
bookkeeping — and let admission/eviction manage what's actually resident.

```swift
let id = try await engine.register(MyPackage.registration, configuration: MyConfig())
```

- Returns a `PackageID` (defaults to the first surface name; pass an explicit `id` to register the same
  package twice, e.g. bf16 vs int4 variants).
- **Multi-package per capability:** registering a second backer for a capability *adds* it and makes it
  the new default ("last registration wins routing"). One registration serving N capabilities is
  constructed once and shared. (Routing detail → topic 4.)
- Re-registering an existing `id` replaces the entry and evicts any stale resident.
- Throws `.licenseRejected(layer)` or `.ineligible(dimension)` — handle at the boundary (→ topic 5).

### prepare — load behind the progress UI
`prepare(_:package:)` constructs the instance, runs **memory admission** (evicting idle LRU residents to
make headroom; honoring the R-MEM-1 real-pressure trigger), prewarms/downloads weights, and loads. Do
this **before the first `run`**, behind `ModelStateView`, so the heavy download/load isn't hidden inside
the user's first request.

```swift
try await engine.prepare(.textToImage)   // download + load happen here, under the progress affordance
```

Throws `.exceedsMemoryBudget(required:budget:)` when even an empty budget can't fit the working set.

### run — lazy-admits if needed, then executes
`run(_:package:)` resolves the capability's package (the default, or a caller-named backer), lazily
constructs + loads it if it isn't resident (same admission path as `prepare`), runs on the
`@InferenceActor`, and marks it most-recently-used (LRU bookkeeping).

```swift
let resp = try await engine.run(myRequest) as! MyResponse
```

`run` works without a prior `prepare` (it admits on demand) — but then the first call eats the load
time. `prepare` exists precisely to move that cost behind the UI.

### evict — free the working set, keep the registration
`evict(_:package:)` / `evict(package:)` calls the package's `unload()` and releases its budget; the
**registration stays**, so it can be re-admitted later. Use it to deliberately free memory.

```swift
await engine.evict(.textToImage)
```

**Important limitation (current):** eviction only reclaims *idle* residents. A **running** inference
can't be stopped — `evict` won't interrupt it and packages don't yet yield between units (the
cooperative-cancellation / mid-run-preemption work, roadmap item 3.4). Don't design UX that assumes
"cancel" frees a model mid-run.

## Observability (what to bind UI to)

- `engine.preparation` — `PreparationMonitor` (phase per capability/package) → drives `ModelStateView`.
- `engine.memory` — a `MemorySnapshot` (budget / resident / available / `underPressure`, plus
  `realResidentBytes` / `underRealPressure` from the R-MEM-1 work). (Memory detail → topic 3.)
- `engine.residentPackages` — `[PackageID: UInt64]` of what's loaded and its charge.
- `engine.needsDownload(_:package:)` — best-effort "will this still hit the network?" to route the user
  into the download UI first.

## Ordering & ownership rules that bite

- **One engine, app-lifetime.** Per-view or per-request engines fragment the governor and the store.
- **Store before register** (topic 1) — `useModelStore` only stamps packages registered after it.
- **Register early, load lazily.** Registering is cheap and front-loads license/eligibility errors to a
  point where you can surface them calmly, instead of at first use.
- **Go through the engine.** Never call `package.load()` / `package.run()` directly — that bypasses
  admission, the governor, routing, and the store stamp (C13). If you're holding a `ModelPackage`
  instance, you've already gone wrong.
- **`prepare` before first `run`** for anything heavy, so first use isn't a silent stall.

## Recognizing the anti-pattern (review checklist)

- [ ] An engine constructed inside a view, a request handler, or re-created on navigation.
- [ ] Multiple `MLXServeEngine()` instances in the app.
- [ ] Direct `MyPackage(configuration:).load()` / `.run()` calls instead of `engine.register` + `engine.run`.
- [ ] No `prepare` — every capability loads inside its first `run`, so first use stalls.
- [ ] Registration happening lazily right before a run (misses upfront gating; fragile store ordering).
- [ ] Treating `evict` / a Cancel button as a way to stop a *running* job (it isn't, yet).

## Verify against source

`mlx-engine-swift/Sources/MLXServeCore/MLXServeEngine.swift` (`register` / `prepare` / `run` / `evict` /
`resident` / `makeHeadroom` / `memory` / `residentPackages` / `needsDownload`),
`Sources/MLXToolKit/ModelPackage.swift` (the protocol + `@InferenceActor` isolation),
`Sources/MLXToolKit/InferenceActor.swift`, `Sources/MLXToolKit/Preparation.swift`.
