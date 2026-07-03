# Topic 7 — Smoke-testing an integration in-app

You can't prove an MLXEngine integration with `swift test` alone — that validates the *coordinator*
(offline, mocks) but never the *metallib* path (topic 6). So integration confidence comes in two layers:
the wiring offline, and one real register → prepare → run → evict cycle in the app.

## Layer 1 — the coordinator, offline (SPM)

The engine's own logic — registration, license/eligibility gating, routing, admission, eviction, the
memory governor — is pure Swift and tests with **mock packages** (no MLX) via `swift test`. If your app
adds any logic *around* the engine (a model picker, a budget chooser, store wiring), cover it here with
mocks. This is fast and runs in CI. (The engine ships `MLXGovernorTests` as the reference pattern.)

## Layer 2 — one real cycle, in the app

The only thing that proves the metallib + weights + sandbox seam is loading and running a real model
inside the app build. Keep a tiny, console-only smoke you can trigger on demand — the engine's own
`AppManager.runLLMSmokeTest()` is the template:

```swift
func smoke() async {
    do {
        await engine.useModelStore(ModelStore(root: storage.resolvedModelsDirectory))  // topic 1
        try await engine.register(MyPackage.registration, configuration: MyConfig())
        try await engine.prepare(.textToImage)                 // download + load (watch ModelStateView)
        let resp = try await engine.run(myMinimalRequest)      // assert non-empty / valid output
        print("[SMOKE] ok:", summarize(resp))
        await engine.evict(.textToImage)                       // confirm budget returns to 0
        print("[SMOKE] memory after evict:", await engine.memory.residentBytes)
    } catch {
        print("[SMOKE] FAILED:", error)                        // EngineError tells you which gate (topic 5)
    }
}
```

What this one cycle actually verifies — the things unit tests can't:

- **Store wiring** — after the run, the per-package marker (`mlx-package.json`) exists under the store
  root and `storage.refresh()` shows Models Installed +1 / Disk Used up. If not, the store wasn't bound
  before register (topic 1).
- **Progress** — `ModelStateView` moved through `downloading → loading → ready` (not a dead spinner).
- **Eviction** — `engine.memory.residentBytes` returns to 0 after `evict` (the package's `unload()`
  actually releases).
- **Re-admission** — preparing a second, larger model evicts the first (watch `residentPackages`).

## A launch-time capability sanity marker (free, no loading)

Log what the machine can admit at startup using `admissibility` (topic 3) — it loads nothing but catches
"this device/budget can't run what we registered" before the user ever taps Run:

```swift
for cfg in candidateConfigs {
    let a = await engine.admissibility(for: MyPackage.manifest.requirements, configuration: cfg)
    print("[CAPABILITY] \(cfg) → admissible=\(a.admissible) now=\(a.admissibleNow) footprint=\(a.footprint)")
}
```

## A reusable in-app validation harness

For more than a console print, mirror the engine's `ValidationView` pattern: a small screen that, per
registered model, does register → prepare → run → evict and reports **timing + peak memory + a visible
result**. It's the fastest way to eyeball a new package end-to-end and to measure the real footprint
(which feeds the package's manifest — back to `mlx-swift-integration`). Build it once; reuse across the
app's models.

### The shared harness: `MLXEngineTestKit`

These seams are implemented as a reusable, opt-in library product in the engine repo —
**`MLXEngineTestKit`** (SwiftUI + `MLXServeCore`/`MLXToolKit` only; not in the shipping `MLXEngineUI`).
Adopt it in each category testing app instead of re-implementing: `ValidationHarness.run(...)` (the
register→prepare(timed)→run(timed)→capture flow + sampler + heartbeat + grants), `EngineMemoryView`
(split + `transientReserveBytes`), `AdmissibilityTiers`/`AdmissibilityTierView` (the 16/32/64 GB tier
check), `PhaseTrace` (phase-attributed peak), `HeadlessAutorun` (env-driven scriptable run). Model-store
grant is already in `MLXEngineUI` (`ModelStorageModel`). It's lean by design — extend per-package only
when a package needs a bespoke seam, and promote it back if it generalizes.

### Per-category testing-app harness — the required seams (checklist)

Each top-level category should have a testing app, and they tend to drift to *different* levels of
completeness. The efficiency sweep is also a survey of these gaps — audit every category app against the
same seams (distilled from the LTX-2.3 run, where several were missing):

- [ ] **Split footprint readout** — show the engine's accounting, not just one peak: **resident floor**
      (declared `residentBytes`, ≈ phys right after `run()` returns with `clearCache()`) **and activation**
      (peak − floor). Without it you can't see whether the declared split matches reality — the whole point
      of the 1.14 work. Emit a machine-readable line too (e.g. `[<app>] SPLIT floor=… peak=… act=…`) for
      headless capture.
- [ ] **`transientReserveBytes` row** — surface `engine.memory.transientReserveBytes` so the co-residency
      win (one shared activation reserve) is visible; otherwise the headline 1.14 benefit is invisible in-app.
- [ ] **Admissibility / tier seam** — a control to evaluate variants via `admissibility(for:configuration:)`
      against simulated smaller budgets (16/32 GB tiers), so "does int4 fit a 16 GB Mac?" has a surface. On a
      big-RAM dev box nothing is inadmissible, so this is the only way to catch tier regressions.
- [ ] **Phase-tagged memory trace** — sample phys per pipeline phase (encode / denoise / decode), not one
      number, so a peak can be *attributed* to a stage. This is what visually proves per-stage eviction
      ("Gemma gone before the denoise peak").
- [ ] **`ModelStateView` + real model-store grant** — exercise the actual security-scoped folder pick +
      download path (topics 1, 5), not hardcoded `/Volumes` paths. Research weights often bypass this, so
      the store/grant seam goes untested per category.
- [ ] **Headless autorun** — an env-driven (`<APP>_AUTORUN=1` …) GUI-less single generation that
      self-terminates, so validation/measurement is scriptable via `xcodebuild` (the reliable measurement
      surface for heavy packages — the CLI bench trips the GPU watchdog).

Promote the ones every category needs (split readout, `transientReserveBytes`, phase trace) into the
**shared** testing harness rather than re-adding them per app.

> **Field evidence (2026-06-30):** audited across the LTX *video* app and the *image* app — **both** had
> model-store grant PRESENT but split-readout PARTIAL and `transientReserveBytes` / admissibility-tier /
> phase-tagged-trace / headless-autorun MISSING. The pattern recurs per category, so these belong in a
> shared harness, not re-implemented per app. (The image app also only registered one capability — check
> each category app actually *registers* its packages, not just that the harness exists.)
>
> **Prerequisite gap (deeper than the seams):** a category app must first be *cleanly stood up* before
> any seam applies — the image app (`MLXEngineImage`) was found with an **empty
> `.xcworkspace/contents.xcworkspacedata` and no project package references** (it never cleanly built).
> Establish the app's package wiring (local-path refs to the engine + the area's packages, products
> linked, a SwiftUI host for the validation view) as step zero. Don't let a "wire in the harness" task
> silently balloon into a shell rewrite — if the app isn't stood up, that's its own scoped task.

## A headless real-run option

If you need GPU inference outside the GUI (CI-ish), build an **engine-driven CLI through `xcodebuild`**
(not `swift run`) so it links the metallib — the engine's own `RunTrellis2Engine` / `RunX` targets are
the pattern: construct the engine, register, prepare, run, write the artifact, evict. Same code path the
app uses, no UI.

## Recognizing the anti-pattern (review checklist)

- [ ] "Integration tested" means only `swift test` passed — the metallib path was never exercised.
- [ ] No in-app smoke; the first real load happens in front of the user with no fallback.
- [ ] Smoke checks the run succeeded but not that the **store marker** appeared or that **evict freed**
      the budget — so a silently-misconfigured store passes.
- [ ] No launch-time admissibility log, so device/budget mismatches surface only at first use.

## Verify against source

The engineering CLAUDE.md "Build & verify" + "Offline (CLI) vs live (Xcode app)" sections,
`mlx-engine-swift/Tests/MLXServeCoreTests/MLXGovernorTests.swift` (mock-package coordinator tests),
`Sources/MLXServeCore/MLXServeEngine.swift` (`memory`, `residentPackages`, `admissibility`), and the
test-app `ValidationView` / `RunX` engine-driven CLI patterns.
