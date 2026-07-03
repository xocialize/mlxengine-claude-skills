# Topic 5 — Progress & error UX

The engine already models the two things first-time users stumble on — "why is it hanging" (it's
downloading) and "why did it refuse" (license/device/memory). The app's job is to *surface* the engine's
signals, not invent its own.

## Progress: bind, don't build

`MLXServeEngine.preparation` is an `@Observable PreparationMonitor` tracking a `PreparePhase` per
capability/package. Bind `MLXEngineUI.ModelStateView` to it and you get the whole affordance for free —
determinate download bar with %+speed, indeterminate load spinner, "first load is heavy, one-time"
captions, and a failure row that shows the reason on hover:

```swift
ModelStateView(monitor: engine.preparation, capability: .imageRestore, title: "NAFNet · SIDD")
```

The phases (from `PreparePhase`): `idle → registering → prewarming(fraction) → downloading(fraction,
bytesPerSecond) → loading → ready`, or `failed(reason)`. You rarely read these directly — `ModelStateView`
renders them — but if you build a custom HUD, those are the states to cover. Don't collapse them into one
spinner; the value is telling "downloading 2 GB" apart from "loading onto the GPU."

Route the user to the download UI *before* a heavy first use with `engine.needsDownload(_:package:)` (a
best-effort "will this still hit the network?").

## Errors: every `EngineError` maps to a clear user story

`register` / `prepare` / `run` throw `EngineError`. Handle each at the boundary — most are
*actionable*, not crashes:

| `EngineError` | Meaning | What the app should do |
|---|---|---|
| `.licenseRejected(LicenseGateResult)` | The license policy didn't admit this package | See "license policy" below — often a deliberate engine-config choice, not a bug |
| `.ineligible(DeviceEligibility)` | C10: missing backend / chip below floor / OS too old | "This Mac can't run X." Pre-filter with `admissibility` so it never gets offered |
| `.exceedsMemoryBudget(required:budget:)` | A single working set is larger than the whole budget | Offer a lighter variant; raise the budget fraction only deliberately (topic 3) |
| `.noPackage(Capability)` | Nothing registered for that capability | Programming error — did `register` run (and after `useModelStore`)? |
| `.unknownPackage(Capability, PackageID)` | That id doesn't back the capability | Programming error — check the id / routing (topic 4) |

`LicenseGateResult` names the failing layer: `.rejectedWeight(license)` vs `.rejectedPortCode(license)`
— surface which, so the user/dev knows whether it's the model weights or the port code that's
incompatible. `DeviceEligibility` names the failing dimension (`.missingBackend`, `.chipBelowFloor`,
`.osBelowMinimum`).

The `.failed(reason)` preparation phase carries the same error text — `ModelStateView` already shows it,
so an error during `prepare` surfaces in the strip without extra code.

## License policy is an engine-construction decision (the consumer gotcha)

The engine defaults to `LicensePolicy.permissiveOnly` — it will **refuse to register** a package whose
weight or port license isn't on the permissive allowlist. That's correct for a shippable, commercial
app. But it bites when you try to consume an **eval / non-commercial** model and `register` throws
`.licenseRejected` — that's the policy working as designed, not a defect.

If the app *intends* to use an acknowledged-eval model, opt in **explicitly at construction**:

```swift
MLXServeEngine(policy: .permissiveOrAcknowledged)   // permissive + curated eval-acknowledged licenses
// .permissiveOnly (default) · .permissiveOrAcknowledged · .any
```

Choosing the policy is a product/legal decision the app owner makes once — don't reach for `.any` to make
an error go away. If a license is rejected under `.permissiveOnly`, the honest question is "are we allowed
to ship this model," not "how do I bypass the gate."

## Recognizing the anti-pattern (review checklist)

- [ ] A custom indeterminate spinner instead of `ModelStateView` — no download %, no "one-time" framing.
- [ ] `try?`-swallowing `EngineError` so failures vanish into a dead button.
- [ ] Treating `.ineligible` / `.exceedsMemoryBudget` as crashes instead of pre-filtering with
      `admissibility` (topic 3) and offering an alternative.
- [ ] Flipping the engine to `.any` to silence a `.licenseRejected` instead of making a deliberate
      policy choice.

## Verify against source

`mlx-engine-swift/Sources/MLXToolKit/Preparation.swift` (`PreparePhase`, `PreparationMonitor`),
`Sources/MLXEngineUI/ModelStateView.swift`, `Sources/MLXServeCore/MLXServeEngine.swift` (`EngineError`,
`needsDownload`), `Sources/MLXToolKit/License.swift` (`LicensePolicy`, `LicenseGateResult`,
`DeviceEligibility`).
