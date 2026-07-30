# Memory Harness — empirical `QuantFootprint.residentBytes`

Produces the empirical footprint every variant declares in its `RequirementsManifest`
(`footprints: [QuantFootprint(quant:residentBytes:)]` in the live Gen-B contract — earlier drafts
called this value `minUnifiedMemory`), which `MemoryGovernor` uses to admit/deny *before* a load.

> **Status of the probe (reconciled 2026-06-12; instrument named 2026-07-01):** there is **no
> `MemoryProbe` / `WorkingSetProbe` type in the live engine yet** — it remains a planned harness. The
> shipped practice is the same METHODOLOGY done manually, and the **shipped instrument for the measurement
> is now `MLXProfiling`** (`MetalToolBox/PROD/mlx-profiling`, `mlx-swift`-only, `MLX_PROFILE=1`): its
> `[MLXPROF]` rows read the authoritative `task_vm_info.phys_footprint` live around the pipeline's own
> `eval` — exactly the number to declare `QuantFootprint` from. Measure at the input envelope (prefer the
> in-app / `xcodebuild`-built path — a bare `swift`-smoke MLX-peak under-reads phys ~2.7×; see
> `package-efficiency.md`), record the number + conditions as a comment next to the `QuantFootprint`
> (e.g. Lens: "Measured (1024², 20 steps, bf16 DiT + dense bf16 encoder + fp32 VAE): peak ~61 GB" → 62 GB
> declared), and log it in APP-VALIDATION.md. Use `phys_footprint`, not `Memory.peakMemory` (cumulative
> allocations, misleads under a `cacheLimit` cap). Build the typed probe when the variant count justifies
> it; the methodology below is what it must implement, and `MLXProfiling` is what it wraps.

> This is the "empirical footprint" conformance item. Gen A numbered it C13; in the current Gen-B
> gate C13 is inversion of control, so **reconcile the number against `conformance.md`** — the
> methodology below is what matters and is stable regardless of the label.

## The rule: measure peak active, not weight size

The admission-relevant quantity is the **peak active unified memory during a real forward pass** —
resident weights *plus* the activation high-water — measured at the **largest input the variant
accepts** (longest audio, max sequence, highest resolution × frames × diffusion steps). Weight file
size is only the floor; activations, KV-cache growth, attention scratch, and VAE/decoder buffers are
frequently the larger term and scale with input. Admitting on weight size alone is how you OOM
mid-inference.

Two ceilings matter; the report records both:

- **`physicalMemory`** — total unified RAM.
- **`recommendedWorkingSet`** (`GPU.deviceInfo().maxRecommendedWorkingSetSize`) — Metal's soft
  ceiling. Crossing it is when allocations start getting paged or the process is jetsam'd. **Admit
  against this, not physical RAM.** An `exceedsRecommendedWorkingSet` flag is a hard "won't run
  cleanly on this device" signal — exactly the case where the variant must be ruled ineligible for
  that tier (`DeviceProfile.eligibility` returns false / the variant is dropped).

## Methodology (what the probe does, and why)

1. **Resident floor** — load, force realization with one warmup of the largest workload,
   `clearCache()`, then read `activeMemory`. With activations freed back to the cache,
   active ≈ weights resident.
2. **Per-workload peak** — for each input size: warm up (compile size-specific kernels, unmeasured),
   then for each measured iteration `clearCache()` + reset peak (`Memory.peakMemory = 0`, which
   rebases peak to current active, i.e. weights) + run with **forced eval** + read `peakMemory`.
   Worst case across iterations is the admission input.
3. **Recommendation** — `worstPeak × (1 + headroom) + fixedOverhead`. Defaults: 20% multiplicative
   (fragmentation, OS pressure, co-resident packages) + 256 MB additive (cold metallib-compile
   scratch, framework, mmap page cache). Tune per modality; diffusion and video want more headroom
   than autoregressive text.

**Forced eval is non-negotiable.** MLX is lazy — an unrealized graph allocates nothing. The workload
closure must call `eval(...)` / `asyncEval(...)` on outputs, or better, call the real `run` inference
path so the measured graph is identical to production.

## Declare the split: persistent weights + transient activation (contract 1.14.0)

The harness already produces both halves — declare them **separately**, don't collapse to one number:

- **`QuantFootprint.residentBytes` = the resident floor** (step 1: weights resident after `clearCache()`),
  plus a small fixed overhead. This is what stays resident the whole time the model is loaded.
- **`QuantFootprint.peakActivationBytes` = worstPeak − resident floor** (step 2 minus step 1), plus the
  multiplicative headroom. This is the *transient* activation scratch, live only during a forward pass.

Why split: inference is serialized on `@InferenceActor`, so the engine reserves **one** activation peak
across all residents (`Σ residentBytes + max(peakActivationBytes)`), not one per model. Declaring the
split lets two models co-reside on the weights while sharing a single activation reserve — strictly more
co-residency than the old "weights+activation as one number" floor, at equal safety. Undeclared
`peakActivationBytes` defaults to 0 (the reactive R-MEM-1 `phys_footprint` trigger still catches
overflow) — but a 0 there means the engine can't reserve for you, so **measure and declare it**.

For same-quant multi-mode variants (e.g. BiRefNet `fast`@1024 vs `best`@2048, both fp16, very different
activation), declare the per-mode split via `FootprintConfigured.residentBytesHint` +
`peakActivationBytesHint` on the configuration — the `QuantFootprint` (keyed on quant) can't tell modes
apart, the hints can.

## Portability of the numbers

`peakActive` is largely **machine-independent** for a given variant + input — it's the graph's
allocation, not the host's. Measure on one representative machine (e.g. an M-series Max) and trust the
peak across tiers. What *is* machine-specific: `recommendedWorkingSet` and cache growth (the buffer
pool scales with `recommendedMaxWorkingSetSize`). That's why admission compares the (portable)
measured `minUnifiedMemory` against the (local) `recommendedWorkingSet` probed at runtime via
`DeviceProfile`.

## How it feeds admission at runtime

- **`DeviceProfile.eligibility(for:)`** compares the variant's declared footprint against the live
  `recommendedWorkingSet` and the `MemoryGovernor` budget; an ineligible variant is dropped for that
  tier rather than admitted and OOM'd.
- **`MemoryGovernor` admission** (budget ≈ 0.7× unified memory) holds back loads whose
  `minUnifiedMemory` would push the working set past the watermark, and **LRU-evicts idle residents**
  to fit a new working set; it rejects a footprint larger than the whole budget.
- **Footprint is now variant-aware AND split (shipped 1.13/1.14).** `MemoryGovernor` charges the
  *selected* variant's footprint via `QuantConfigured` (the config's quant) and `FootprintConfigured`
  (per-mode hints), not the largest-that-fits survey — so a multi-variant manifest no longer
  over-reserves to the max variant. Residency is `Σ residentBytes`; the activation reserve is a single
  `max(peakActivationBytes)` across residents (serialized inference). The old "largest-that-fits"
  behavior remains only as the fallback for a config that opts into neither. So: declare per-quant
  `QuantFootprint`s with the split, and for same-quant modes add the `FootprintConfigured` hints.
- **Cache policy** — the buffer pool can grow to GBs uncapped. Set `Memory.cacheLimit` from the
  budget so cache doesn't crowd out a second resident model. The report's per-workload `cacheAfter`
  tells you a sane ceiling (often far smaller than peak).
- **Wired memory** — current mlx-swift exposes a `WiredMemoryManager` + ticket admission (`withWiredLimit`
  is deprecated to a no-op). Weights are the natural thing to wire (keep non-pageable); route weight
  residency through a wired ticket sized at `weightsResident` so the `MemoryGovernor` watermark ladder
  and MLX's own admission agree rather than fight.

## Usage — wire it as a gated bench (one per variant)

```swift
import XCTest
import MLXToolKit
@testable import SwiftRoFormerCore

final class AudioSeparationMemoryReportTests: XCTestCase {
    func testMeasure_roformer_f16() async throws {
        let variant = AudioSeparationPackage.variant(id: "roformer-kim-vocal-2")
        // Only measure where this variant is actually admissible on the CI box.
        try XCTSkipUnless(DeviceProfile.current.eligibility(for: variant))

        var model: RoFormerSession!
        let report = try await MemoryProbe().measure(
            modelID: variant.modelID, quant: "f16",
            load: { model = try await RoFormer.load(variant) },        // realizes on first run
            workloads: [
                ("audio=60s",  { _ = try await model.separate(.silence(seconds: 60),  eval: true) }),
                ("audio=300s", { _ = try await model.separate(.silence(seconds: 300), eval: true) }),
                ("audio=600s", { _ = try await model.separate(.silence(seconds: 600), eval: true) }), // envelope max
            ])

        print(report.prettyReport)
        // Persist next to the port for audit + codegen into the variant matrix.
        let url = URL(fileURLWithPath: "Bench/Reports/\(report.modelID)-f16.json")
        try JSONEncoder().encode(report).write(to: url)

        XCTAssertFalse(report.exceedsRecommendedWorkingSet, "variant no longer fits its target tier")
    }
}
```

Use synthetic worst-case inputs (`.silence(seconds:)`, zeroed tensors at max shape) so the peak
reflects the **input envelope**, not whatever sample was lying around — peak is a function of shape,
not content.

**Live MLX tests must be XCTest, not swift-testing.** The SPM metallib workaround (colocating the
`Cmlx` bundle in `.build/debug/`) only works for XCTest; swift-testing suites run in a separate helper
process away from `.build/debug`, so the metallib is never found. Write gated live suites as XCTest.

## Definition of done (memory)

Every vended variant has a recorded measurement (today: the manifest-comment + APP-VALIDATION
entry; with the probe: a committed `Bench/Reports/<modelID>-<quant>.json`) whose recommended value
populates that variant's `QuantFootprint.residentBytes`, measured at the input envelope with
`exceedsRecommendedWorkingSet == false` on every tier the variant's eligibility claims to support.
Re-run when the model, quant, or input envelope changes.

## If activation scales with an input dimension, don't sample the envelope — remove the growth (Audio8, 2026-07-30)

`peakActivationBytes` was declared wrong **three times running** on one package — 5.00 → 7.20 →
9.50 GB — and each revision failed for the identical reason, which is what makes it worth writing
down. The transient was linear in generated frames (`≈ 1824 + 14.2 × frames` MB), and every
declaration came from a *sample* that stopped short of the cap the package itself permits:

| | basis | missed |
|---|---|---|
| 5.00 GB | one 9.2 s utterance | anything longer |
| 7.20 GB | corpus sweep, longest 15.7 s | the default `maxFrames` cap |
| 9.50 GB | the fitted model AT the cap | nothing — but only because the model was fitted |

The escalation is the tell. Each fix was "measure more", and each one was overtaken. Three rules,
in increasing order of value:

1. **Sampling cannot bound a growing envelope.** If activation depends on an input dimension,
   *fit the relationship* (a handful of points across a 50× range gave ±4%) rather than reporting
   a maximum you happened to observe.
2. **Declare at the parameter cap YOUR package permits by default**, not at the largest input you
   tried. A caller raising `maxFrames` is opting out; a caller using the default is not.
3. **Better: make the envelope constant.** Windowing the decode made it flat — 3.4–3.9 GB at 64,
   128, 224 and 1035 frames alike — and the declaration dropped to 4.20 GB. That is worth more
   than the 5.3 GB saved: `MemoryGovernor` reserves this number process-wide, so a *bounded*
   figure is one the real envelope cannot exceed, whereas a fitted one is still a bet on caller
   behaviour. Measured cost: ~14% throughput (median RTF 0.94 → 1.07).

**Measure it where it ships.** The CLI harness read 3.40 GB for the same work the sandboxed app
read 4.25 GB — the app carries allocation the harness does not. Declare from the in-app number; a
headless smoke under-reads. (See also `field-issues.md` on Xcode pinning a published tag: an app
"validating" a fix can silently be running the previous release, and the tell is the app
reporting the *pre-fix* number.)
