# Topic 3 — Memory budget & admissibility from the app's side

The engine does the hard part — charging footprints, evicting under pressure, reclaiming on real
memory. The app's job is smaller and specific: **set a sane budget, pick variants the device can
actually run, and surface what's resident**. This topic is the consumer view of the memory-governance
work (config-aware footprint + R-MEM-1) — what an app touches, not the internals.

## What the app controls vs. what the engine handles

| App's job | Engine's job |
|---|---|
| Choose the **budget fraction** at engine construction | Charge each registration's footprint; track resident bytes |
| **Pick variants** (quant / mode) per package config | Resolve the right footprint for the chosen variant |
| **Survey admissibility** before offering/loading a variant | Evict idle LRU at admission to make headroom |
| **Bind a memory view** to show residency/pressure | Reclaim idle residents on **real** memory pressure (R-MEM-1) |
| Optionally **evict** to free memory deliberately | Reject a single working set that can't fit the budget |

## Setting the budget

The governor's budget is a fraction of total unified memory:

```swift
let engine = MLXServeEngine(governor: .forDevice(.current(), fraction: 0.7))
```

- **0.7 (default)** leaves headroom for the OS and other apps — the right default for a shipping consumer
  app.
- Raise it (e.g. 0.85) only for a controlled, single-purpose "pro"/validation context where the user
  expects the app to dominate memory (this is what the engine's own proving-ground used so heavy editing
  models admit). It's a deliberate product decision, not a free win — a too-high fraction trades OS
  headroom for admitting one more model.
- The budget is a *declared-byte* ceiling. Real memory is governed separately by the R-MEM-1 trigger
  (below), so a high fraction doesn't mean the app will actually stack past the physical ceiling.

## Picking variants — let the package's footprints do the work

Most packages ship multiple variants. The app picks one via the package's configuration; the engine
charges the matching footprint automatically:

- **`QuantConfigured`** — the config exposes `quant`; the engine charges that quant's declared
  `QuantFootprint` (e.g. bf16 vs int4).
- **`FootprintConfigured`** — for variants that share a quant but differ a lot in working set (the
  BiRefNet `fast`@1024 ≈ 4.9 GB vs `best`@2048 ≈ 18.3 GB case, both fp16), the config supplies
  `residentBytesHint` (persistent weights) and `peakActivationBytesHint` (the transient activation peak)
  so the same-quant modes are charged correctly.

Footprints are now a **split**: persistent weights (resident the whole time) + a transient activation
peak (live only during inference). The app doesn't compute either — the package author measures and
declares them (that's `mlx-swift-integration`'s job). What the app *should* do is **not offer a variant
the device can't run** — which is what admissibility is for. Note admissibility's `footprint` now reports
the variant's *own peak* (persistent + transient), so a "will it fit" check already counts activation.

The payoff the app sees: because the engine reserves a single shared activation peak, **more models stay
co-resident** than before — a chained pipeline (e.g. the optimizer's IQA → restore → upscale) can keep
its stages warm without each one's scratch being double-counted. You generally get this for free; just
don't assume the *old* "weights+activation per model" math when sizing what fits.

## Admissibility — the "will it fit?" survey (no loading)

`admissibility(...)` answers, without constructing or loading anything, whether a package's requirements
fit this device (C10 eligibility) and memory budget. It's the seam for a **model-manager / variant
picker** that greys out what won't run.

```swift
public struct Admissibility {
    let eligibility: DeviceEligibility   // backends / chip tier / OS
    let footprint: UInt64                // bytes that would be charged
    let fitsBudget: Bool                 // could load (possibly after evicting idle residents)
    let fitsAvailable: Bool              // could load right now, no eviction
    var admissible: Bool                 // eligible && fitsBudget
    var admissibleNow: Bool              // eligible && fitsAvailable
}
```

Three overloads, pick by what you're asking:

```swift
// 1. Variant-agnostic survey — "can this machine run this package at all?" (largest-that-fits)
let a = await engine.admissibility(for: MyPackage.manifest.requirements)

// 2. The SELECTED variant — reads the config's quant + footprint hint exactly like register would.
//    Use this to rank/grey-out concrete variants in a picker.
let best = await engine.admissibility(for: MyPackage.manifest.requirements,
                                      configuration: MyConfig(mode: .best))   // FootprintConfigured + QuantConfigured

// 3. Explicit quant/hint when you don't have a config object handy.
let int4 = await engine.admissibility(for: reqs, quant: .int4, hint: nil)
```

Pattern — a variant picker that only offers what fits:

```swift
let candidates = [MyConfig(mode: .fast), MyConfig(mode: .best)]
for cfg in candidates {
    let adm = await engine.admissibility(for: MyPackage.manifest.requirements, configuration: cfg)
    // enable the row iff adm.admissible; show adm.footprint; mark adm.admissibleNow as "loads instantly"
}
```

Surveying first is how you avoid the bad UX of offering a variant, loading it, and only then throwing
`EngineError.exceedsMemoryBudget` in the user's face. **Pre-check with admissibility; let prepare/run be
the happy path.**

## Reading the memory state (the HUD)

`engine.memory` returns a `MemorySnapshot` — bind a small indicator to it:

```swift
let m = await engine.memory
// m.budgetBytes                                            — the governor's ceiling (fraction × unified)
// m.residentBytes                                          — Σ persistent weights of residents
// m.transientReserveBytes                                  — the ONE activation peak reserved (serialized)
// m.availableBytes                                         — budget − residentBytes − transientReserve
// m.residents: [Capability: UInt64]                        — what's loaded, per capability
// m.underPressure: Bool                                    — declared resident ≥ high-watermark
// m.realResidentBytes: UInt64?                             — actual phys_footprint (nil if unavailable)
// m.underRealPressure: Bool                                — actual footprint over the watermark (R-MEM-1)
```

`engine.residentPackages` gives the package-keyed view (`[PackageID: UInt64]`). Note the engine reserves
**one** activation peak across all residents (inference is serialized — only one model runs at a time),
not one per model, so `residentBytes + transientReserveBytes` is the accounted peak and `availableBytes`
already nets both out. Showing real footprint + `underRealPressure` is more honest than declared bytes
alone — it's the number Activity Monitor shows.

## BudgetAware — let memory-adaptive packages adapt (don't fight them)

Some packages have a dtype lever that trades quality for memory (fp8 vs bf16 vs fp32). Those conform
their configuration to `BudgetAware`, and the engine stamps `availableBudgetBytes` — the real headroom
the model is loading into, computed *after* admission/eviction — right before `load()`, so the package
can pick a lighter dtype when memory is tight. App-side conformance:

- **Set up the config and let the engine stamp it.** The app constructs the package's configuration as
  usual; it does **not** set `availableBudgetBytes` (the engine owns that value). Leave it `nil`.
- **Don't hardcode a dtype that defeats the adaptation.** If a package is `BudgetAware`, forcing its
  heaviest dtype via config removes its ability to fit under pressure. Expose the user's *quality
  intent* (e.g. a quality/balanced/fast preference) and let the package resolve dtype from intent +
  the stamped budget, rather than pinning a precision in the app.
- This is purely opt-in per package — non-`BudgetAware` packages need nothing from the app here.

## mmap & the model-store volume

Ports load weights mmap/lazy (the Apple analog of partial loading — the OS pages weights into unified
memory on demand). The app influences this through **where the model store points** (topic 1): a fast
local volume keeps paging cheap; a slow/external/network volume makes first-use and any re-paging
sluggish. When you let the user choose the models folder, a fast internal location is the good default.

## R-MEM-1: declared bytes are a floor, not a cap

The engine now reads the process's real `phys_footprint` on the admission path and evicts idle LRU
residents when actual memory is over the high-watermark — even if declared footprints summed under
budget. App-level implications:

- **Co-residency is opt-in and pressure-bounded.** Two heavy models stay co-resident only while they
  *genuinely* fit. If you deliberately hold two backers resident (the multi-package path), the engine may
  still reclaim the idle one under true pressure. Don't assume "declared bytes fit ⇒ both stay loaded."
- **Idle is the unit of reclaim.** Only idle residents are evicted; a running inference is never
  interrupted (and still can't be cancelled — roadmap 3.4). So real-pressure reclaim is invisible to an
  in-flight request.
- **Surface it.** If your app keeps several models warm, reading `underRealPressure` lets you tell the
  user "memory is tight, models may reload on demand" instead of them being surprised by a reload pause.

## Recognizing the anti-pattern (review checklist)

- [ ] A hardcoded budget fraction at 0.9+ "to fit the big model," starving the OS — or no governor at
      all (defaulting blindly) when the app is memory-sensitive.
- [ ] A variant picker that offers quants/modes without consulting `admissibility` — users pick something
      the device can't run, then hit `exceedsMemoryBudget`.
- [ ] Catching `exceedsMemoryBudget` as the *primary* fit check instead of surveying first.
- [ ] A memory HUD that shows only declared `residentBytes` and ignores `transientReserveBytes` /
      `realResidentBytes` / `underRealPressure` (looks fine while the process is actually thrashing).
- [ ] Assuming two heavy models stay co-resident because their declared footprints sum under budget.
- [ ] Setting `availableBudgetBytes` on a `BudgetAware` config from the app (the engine owns it), or
      hardcoding a package's heaviest dtype so it can't adapt under pressure.
- [ ] Pointing the model store at a slow/external volume, making mmap paging sluggish on every use.

## Verify against source

`mlx-engine-swift/Sources/MLXServeCore/MemoryGovernor.swift` (`forDevice`, `footprint(for:quant:hint:)`,
`MemorySnapshot`), `Sources/MLXServeCore/MLXServeEngine.swift` (`admissibility` overloads, `memory`,
`residentPackages`, `makeHeadroom` real-pressure pass), `Sources/MLXServeCore/HostMemory.swift`,
`Sources/MLXToolKit/{RequirementsManifest,PackageConfiguration}.swift` (`QuantFootprint` incl.
`peakActivationBytes`, `QuantConfigured`, `FootprintConfigured` incl. `peakActivationBytesHint`,
`BudgetAware`). See also the engine `docs/architecture.md` R-MEM-1 spec and the package-author
counterpart in the `mlx-swift-integration` skill (`references/package-efficiency.md`).
