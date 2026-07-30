# Stage 1 — Swift-MLX Port Conformance (harness-ready `ModelPackage`s)

**Goal:** every Python-MLX → Swift port arrives at the engine already carrying its own capability
surface and accurate requirements, with zero engine-side glue. The integration cost is paid once
(here) instead of once-per-package.

> **Vocabulary note.** This file uses the current (Gen-B) `MLXToolKit` contract — `ModelPackage`,
> `PackageManifest`, `Capability`, `Quant`, `PackageRegistration`. An earlier design expressed the
> same craft as `MLXToolProvider` / `MLXTool` / `ToolDescriptor` with a per-port MCP-server target;
> that vocabulary is **superseded** and intentionally not used here. The reasoning below (the
> dependency-inversion seam, variant-matrix-as-data, parity discipline) carries over unchanged.

## The dependency-inversion seam

A standalone SPM can't expose itself to the engine without coupling them. So one tiny,
dependency-light package — **`MLXToolKit`** (the contracts) — is imported by *both* sides:

```
MLXToolKit  (protocols + value types only; depends on nothing but stdlib + Foundation)
   ▲                         ▲
   │ conforms                │ consumes
<Name>-swift  ───────────►  MLXEngine (MLXServeCore / MLXServeEngine, MemoryGovernor, MLXServeConformance)
```

The engine never depends on a port. Each port depends only on `MLXToolKit`. The engine discovers
ports through the contract via `PackageRegistration`. The engine's own internal topology is three
targets — **`MLXToolKit`** (contracts) / **`MLXServeCore`** (the coordinator) / **`MLXServeConformance`**
(the C0–C13 harness) — but that is the *engine's* structure; a port is just a `-swift` SPM that
conforms.

## 1. Per-port package topology

A port is **one `-swift` SPM**. Its internal shape depends on the path:

| | Path A (reuse `mlx-swift-lm`) | Path B (novel architecture) |
|---|---|---|
| Ported model code (`Core`) | none — reuse `MLXLLM` / `MLXVLM` | a `<Name>Core` module: layers, weight init, `load(plan:) -> Session`, inference on the session |
| `ModelPackage` conformer | wraps the runtime loader | wraps `<Name>Core` |
| Default product | MCP-free, transport-free | MCP-free, transport-free |

Rules that hold for both:

- **`-swift` suffix on the package/repo name ONLY** (mirrors `mlx-engine-swift`). The module/product
  stays clean PascalCase (`MLXQwenLLM`, not `mlx-qwen-llm-swift`). The Python port keeps the bare name.
- The package **depends on `MLXToolKit`**; never fold a package into `mlx-engine-swift`.
- If a `Core` exists, it has **no** `MLXToolKit` / engine import — it is the only thing you actually
  *port*; everything else is declaration. Keeping `Core` clean makes it reusable, unit-testable in
  isolation, and free of harness churn.
- **No transport in the default product.** Linking a port into the engine must never pull an MCP SDK
  or any transport; the engine decides transport. Enforce by asserting on the resolved dependency
  graph in CI.

## 2. The `MLXToolKit` surface you conform to

You implement against these; the engine satisfies the host-side halves (construction, loading,
admission, device/memory probing). Treat the exact type names below as the *roles* — reconcile
against the live `MLXToolKit` source, since the contract version grows (see C-version below).

- **`ModelPackage`** — the discovery + conformance unit. Exactly one per package. The live protocol:
  `associatedtype Configuration: PackageConfiguration` (C9 init-time config),
  `nonisolated static var manifest`, `nonisolated init(configuration:)` (cheap — no weights),
  `load()` (no-arg; residency), `unload()`, `run(_:)`. There is **no static `registration`
  requirement** — registrations are built at the call site via `PackageRegistration.of(Type.self)`.
- **`PackageManifest`** — declares, once: `contractVersion`, `license` (a `LicenseDeclaration` with
  BOTH `weightLicense` and `portCodeLicense`), `provenance` (`sourceRepo`/`revision`/`tier`),
  the `RequirementsManifest`, `specialties`, and `surfaces: [ToolDescriptor]`. **`capabilities` is
  DERIVED from `surfaces`** — there is no separate capability list and no `variants:` field; the
  variant axis is realized via the `Configuration` + multiple `QuantFootprint`s.
- **`RequirementsManifest`** — load-bearing: `footprints: [QuantFootprint(quant:residentBytes:)]`,
  `requiredBackends: Set<Backend>`, `os: OSRequirement(minMacOS:)`, `chipFloor: ChipTier?`. The
  engine's admission gates on it (`DeviceProfile.eligibility(for:)` + `MemoryGovernor` budget). Set
  each `residentBytes` from a real measurement (see `memory-harness.md`), not a guess — an
  inaccurate value gets the package rejected or mis-budgeted.
- **`Capability`** and **`Quant`** — **additive enums**; the engine grows cases over time. A new case
  makes an older package's exhaustive `switch` non-exhaustive → build break. Handle the new case or
  `@unknown default` (C12 discipline; see lessons file).
- **`CapabilityRequest` / `CapabilityResponse`** — the canonical per-capability I/O. `any ModelPackage`
  / `any CapabilityRequest` / `any CapabilityResponse` are `Sendable`, so they cross the engine actor
  cleanly.
- **`LicensePolicy`** — `.permissiveOnly`; `LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted`.
- **`WeightSourcing` / `WeightSource`** (engine ≥ 0.19.0) — the `Configuration` declares every
  fresh-machine network source (`WeightSource{role, repo, revision, matching}`) and computes the
  still-missing subset via `missingWeightSources(storeRoot:)`. Complements `ModelStorable` (WHERE
  weights go) with WHAT would be fetched; drives the MAT gate and the ENGINE's pre-load
  materialization (contract 1.24: the engine executes, `load()` just loads) — full requirements
  in section 4.

## 3. Per-port conformance checklist (C0–C13)

Run this for each port. Numbering follows the authoritative Gen-B gate (see
`references/engine-contract.md` for the full C0–C13 summary;
`~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/conformance.md` is ground truth). Items
marked "(craft)" are workflow discipline that isn't a numbered C item.

- **Topology (craft, unnumbered).** `-swift` SPM, depends on `MLXToolKit`; default product
  transport-free (CI-asserted); any `Core` has no `MLXToolKit`/engine import.
- **Contract version (C0).** The package's manifest declares the `contractVersion` it targets so
  the engine can refuse/quarantine an incompatible package.
- **Package entry point (C1).** Exactly one `ModelPackage`, registered via
  `PackageRegistration.of(Self.self)` at the call site; ≥1 capability from the canonical enum.
  One model with several surfaces stays **one** package and shares a loaded session.
- **Lifecycle is host-owned (C13).** The engine constructs, loads, drives, and evicts. **No internal
  weight singleton, no self-caching.** The conformer is
  `@InferenceActor final class … : ModelPackage` (class-level isolation makes it `Sendable` and
  satisfies the actor-isolated `load`/`unload`/`run`); `init`, `manifest`, and `registration` are
  `nonisolated`. The session may be non-`Sendable` as long as it never escapes its `perform { … }`
  closure.
- **Variant matrix as declared data.** Every selectable configuration is declared with a
  `minUnifiedMemory` so admission can admit/deny *before* load. Compute the repo id from
  **size × quant**; don't hardcode one checkpoint string.
- **License, twice (C7 + C8).** The manifest's `license` (weight, **C7**) is gated by
  `.permissiveOnly` at registration; the **port-code** license (**C8**) is gated independently. A
  non-permissive weight cannot be admitted even if one gate has a bug. The allowlist grows
  (`LicenseRef-FunASR-Model`, `CC-BY-4.0`, …) — check current `LicensePolicy`.
- **Canonical schema + hand-tuned descriptions (C2/C3/C11).** I/O matches the capability's
  canonical schema and artifact (serialized round-trip form, C3); the `ToolDescriptor` summary and
  parameter descriptions are written for the model audience — underspecified schemas measurably
  degrade selection (C11 introspection).
- **Invocation contract.** `run(_:)` dispatches on `request.capability`, downcasts to the canonical
  request, honors `Task.checkCancellation()`, and throws `PackageError.unsupportedCapability` /
  `.notLoaded` appropriately. Tool-level failures are returned, not thrown across the boundary.
- **Sendable + accurate requirements (C7).** The `ModelPackage` is `Sendable`. Requirements are
  declared accurately per variant (footprint, backends, chip, OS).
- **Asset resolution (craft, unnumbered).** Weights/tokenizer/config resolve from the
  `Configuration` (and the engine-stamped `modelsRootDirectory` for `ModelStorable` configs) —
  no hardcoded per-machine paths in the package.
- **Weight sourcing + auto-materialization (MAT gate, engine ≥ 0.19.0; engine-executed since
  contract 1.24).** The `Configuration` conforms to `WeightSourcing` (quant-aware globs); the
  ENGINE downloads the missing sources pre-`load()` (opt out via `SelfMaterializing` only for
  non-HF hosts / wrappers that fetch internally — those forward progress via
  `WeightDownloadProgress` themselves), `prewarmPaths` resolves the store layout, and the
  package's own conformance suite runs `MaterializationConformance.check` (MAT-1..5, offline).
  Full requirements in section 4.
- **Device eligibility (C10).** `DeviceProfile.eligibility(for:)` — required backends present, chip ≥
  floor, OS ≥ min — gates admission at `register`. A variant that can't run on a tier must not claim
  it.
- **Dual-mode purity (C0/C10).** No transport in the default product (restated; it's the load-bearing
  isolation property).
- **Forward-compat discipline (C12).** `@unknown default` on every `Capability`/`Quant` switch —
  the enums are additive and WILL grow (1.1.0 added int5/int6; 1.2.0 added imageEdit).
- **Parity + descriptor tests (craft; gate inputs).** Ship (a) an **elementwise parity** test vs
  the reference on a fixture (cosine ≈ 0.999 — *not* just norm/energy sanity; see the
  silent-failure class in the lessons file), gated to admissible tiers, and (b) a check that
  schemas are well-formed. The PyTorch→Python-MLX parity is `mlx-porting`-skill territory; the
  **Python-MLX→Swift parity is this skill's Stage 1** — workflow and numerics doctrine in
  `swift-port-parity.md` (phase gates, key contracts, CPU-stream fixtures, CLI gate modes); the
  conformance gate consumes the result.
- **Empirical footprint (memory item; feeds C10).** Every variant ships a recorded measurement
  whose value populates its `QuantFootprint.residentBytes`. See `memory-harness.md`. (Gen A
  numbered this C13; in Gen B C13 is runtime-governance cooperation / inversion of control.)

## 4. Weight sourcing & auto-materialization (engine ≥ 0.19.0 — the MAT gate)

First-run weight downloads are a **declaration responsibility with an engine-executed contract**:
the app only picks a models folder; the package declares WHAT it would fetch on a fresh machine
(v0.19.0, `ee65087`), and **since contract 1.24 the ENGINE executes the download itself** —
`resident()` materializes the missing sources into the store BEFORE `load()`
(`MLXServeCore.WeightMaterializer`: chunk-delegate streaming, 8-way ranged chunks for xet-backed
files ≥ 64 MB, progress through `WeightDownloadProgress`), so `load()` just loads. A package
ships its own executor ONLY by conforming to `SelfMaterializing` (non-HF hosts, wrappers whose
runtime fetches internally); a legacy package that still self-materializes in `load()` stays
correct — its own missing-check runs after the engine's pass and finds nothing left. Every new
package conforms **born-clean** — same adoption pattern as the 1.14 efficiency contract: build it
into Stage 2 steps 3–4, never as a later retrofit. Explicit-directory configs remain the dev-mode
escape hatch and **never touch the network** (DEV_ARCHIVE flows stay untouched).

Four requirements:

1. **Declare — conform the `Configuration` to `MLXToolKit.WeightSourcing`.** `weightSources`
   lists every fresh-machine network source as a `WeightSource(role:repo:revision:matching:)` —
   `role` is a stable per-source tag ("components", "text-encoder", "transformer-int8") keying
   progress/measurement/UI; `repo` is `org/name`; `matching` globs select files within the repo
   (nil/empty = whole snapshot). Multi-source packages are normal. **Quant-tiered configs must
   EXCLUDE files their quant doesn't need from the globs** — MLXLTX2's int8/int4 configs exclude
   the 35 GB bf16 transformer from the components glob and add a `transformer-<quant>` source
   instead. `missingWeightSources(storeRoot:)` computes the still-missing subset: honor the
   configuration's **explicit local paths first**, then probe the ModelStore layout
   (`<root>/<org>/<name>/…`); nil store + no explicit paths ⇒ everything missing.
   **BUNDLED-WEIGHTS packages declare the other vocabulary (engine ≥ 0.24.0, contract 1.17):**
   a package whose checkpoints are vendored in the SPM resource bundle (Real-ESRGAN's ~2 MB
   SRVGGNetCompact checkpoints) conforms to `BundledWeightSourcing` *instead of*
   `WeightSourcing` — `bundledWeightSources` lists `BundledWeightSource(role:url:)` with the
   resolved bundle URL (declare the `Bundle.module` lookup result directly; `nil` = lookup
   failed, the gate fails that role). Do NOT declare a network `WeightSource` for bundled
   weights — it would either report a dishonest missing set or force a pointless download.
   Hybrid (bundled + network) packages conform to both; the role namespace is shared. A
   bundled-only config whose sources all resolve makes `engine.needsDownload` read `false` —
   the sanctioned signal (the pre-1.17 always-present-`prewarmPaths` workaround is retired;
   `WeightPrewarming` keeps its real job, cold-start page-in).
2. **Execute — the ENGINE materializes; `load()` just loads (contract 1.24).** When explicit
   dirs are nil and a store root is set, `resident()` downloads the missing sources into the
   FLAT store layout (`models--<org>--<name>/<path>`, the fleet convention — the MS-2 default
   probe accepts it alongside the hub-client snapshot layout) before the package is even
   constructed, with byte-accurate progress surfaced through `PreparationMonitor`. `load()`'s
   only job is loading from the store-resolved view of the configuration (MLXLTX2's
   `resolved(storeRoot:)` maps nil dirs onto the store; explicit dirs always win). **Opt-out:**
   conform the `Configuration` to `SelfMaterializing` when the generic executor can't do the
   job (non-HF hosts, Path-A wrappers whose runtime downloads internally) — then `load()`
   executes its own download and MUST forward progress via
   `WeightDownloadProgress.report(fraction:bytesPerSecond:)`; a download the monitor can't see
   is a conformance smell (dead spinner; the live `MaterializationRun` bench flags it as
   `downloadPhase=NO`), and sources should map source *i* of *n* onto fraction
   `[i/n, (i+1)/n)` so one progress bar stays monotonic. Do NOT ship a new per-package
   `WeightMaterializer` copy — that pattern is retired (the engine's executor absorbed the
   mage-flow reference implementation, cf45682/6faa4cb).
3. **Prove — MAT-1..5 in the package's own conformance suite.** Next to the C0–C13 gate tests, run
   `MLXServeConformance.MaterializationConformance.check(freshConfiguration:satisfiedConfiguration:)`
   — offline, no network, no weights: **MAT-1** ModelStorable · **MAT-2** non-empty source
   declaration (network `WeightSourcing` and/or bundled `BundledWeightSourcing`) · **MAT-3**
   role/repo hygiene (roles unique across BOTH vocabularies, `org/name` repos) · **MAT-4**
   fresh-machine posture, each subset by its own rule (network sources: nil store ⇒ ALL missing;
   bundled sources: ALL resolve to existing files — a stripped resource fails in the suite, not
   at the user's first inference) · **MAT-5** explicit paths satisfy (tiny probe files in a temp
   dir make the satisfied config; bundled-only packages have nothing to satisfy). Assert
   `report.passed` with `report.summary` as the failure message, and run it **per selectable
   quant tier** (or per bundled variant) — the declaration changes with the selection.
4. **Prewarm — `prewarmPaths` resolves the store.** For nil-dir configs, `WeightPrewarming`'s
   `prewarmPaths` must resolve against the store layout so the engine's `WeightPrewarmer` pays off
   from the SECOND cold launch on downloaded weights (first launch is a no-op — nothing on disk
   yet; missing paths skip).

**Reference implementations:** network — MLXLTX2 (`~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift`,
`7ae7aed`) — the `LTX2Configuration` `WeightSourcing` extension (+ `resolved(storeRoot:)`) and
`Tests/MLXLTX2Tests/MaterializationTests.swift` (its `Sources/MLXLTX2/WeightMaterializer.swift`
is the retired per-package executor pattern — since contract 1.24 the engine's
`MLXServeCore/WeightMaterializer.swift` is the executor; don't copy the package-side one).
Bundled — mlx-realesrgan-swift v0.4.1 (`~/Development/mlxengine-image/PROD/mlx-realesrgan-swift`) —
the `RealESRGANConfiguration` `BundledWeightSourcing` extension over
`SRVGGNetCompact_Playback.Variant.bundledWeightsURL` and its full-gate-per-variant
`MaterializationTests`.
The consumer/app side — folder pick, `needsDownload` routing, progress UI, and the live
`MaterializationBench` measurement whose `[MAT]` logLine is the registry's Val evidence for
first-run behavior — is documented in the `mlxengine-implementation` skill's
`references/materialization.md`.

## 5. Cancellation honoring (engine ≥ 0.27.0 — the CAN gate)

Cancellation honoring is a **package obligation with an executable gate** — the adjunct to C13's
cancellation convention the same way the MAT gate (§4) is to `WeightSourcing`. Every new package
conforms **born cancel-clean** in Stage 2 step 4, never as a later retrofit (the Qwen3-TTS Talker
loop shipped with no checkpoints and nothing caught it — this gate is why that can't recur).

The engine's final run-lifecycle semantics (V1–V3, engine 0.26.0) the package codes against:
**both** cancellation lanes arrive as the same `CancellationError` — a user cancel (the app
cancelling the `Task` wrapping `engine.run()`) which the engine surfaces to the caller unchanged
(classified `.cancelled`, not failed), and a governor preemption which the engine recognizes as
its own doing and **requeues**. The package cannot and must not tell them apart: checkpoint,
unwind cleanly, rethrow unchanged.

Three requirements:

1. **Checkpoint — `try Task.checkCancellation()` first, then at every natural yield point.**
   The FIRST act of `run()` is a cancellation check — before `notLoaded` validation, before
   dispatch (this entry checkpoint is what the offline gate exercises). Then per denoise step,
   per VAE-decode chunk, per generated token/frame, per encoder layer — the LTX-proven
   placements. Rethrow the `CancellationError` **unchanged**: wrapping it in a package error
   ("laundering") breaks both the engine's lane disambiguation and the caller's `.cancelled`
   classification. Report `RunProgress` (contract 1.18) at the same seams — per-step progress
   is accepted as observable evidence of the cadence in CAN-3, and it feeds the governor's
   preserve-nearly-done policy.
2. **Prove — CAN-1..3 in the package's own conformance suite.** Next to the C0–C13 + MAT tests:
   `await CancellationConformance.checkRun(package:request:)` with a stub/smallest configuration
   (construction is cheap per C13; the pre-cancelled form throws at the entry checkpoint before
   weights are touched, so it is offline-safe for every package) — **CAN-1** pre-cancelled
   `run()` surfaces `CancellationError` · **CAN-2** the outcome is cancelled-not-failed in the
   capability's canonical shape (error unwrapped; a partial response, if returned, carries
   `FinishReason.cancelled` — prove via `classifiesCancelled:`). Then
   `CancellationConformance.checkCadence(manifest:posture:)` — **CAN-3** a long-run-implied
   manifest (video/audio generation capability, or ≥ 2 GB declared `peakActivationBytes`)
   declares its checkpoint cadence (`.cadence([...(phase:unit:reportsRunProgress:)])`); genuinely
   sub-second packages declare `.subSecondRuns(reason:)` instead. The declaration in the suite IS
   the document of record for the cadence.
3. **Measure — the live `[CAN]` bench in the consuming app.** `MLXEngineTestKit.CancellationBench`
   cancels at T seconds into a real run through the sanctioned user seam, measures time-to-throw
   (one MLX eval is the substrate floor; LTX: 1.08 s steady-state, 16–21 s in a compile-heavy
   first step), captures the V2 phase that absorbed the latency, and proves the clean
   cancel→re-run recovery. Xcode-app harness only (metallib boundary). This replaces bespoke
   per-app harnesses (LTX's `LTX_CANCEL_TEST`/`LTX_CANCEL_AFTER`/`LTX_CANCEL_RERUN` levers) —
   don't build new ones.

> ⚠ **A green CAN-1..3 proves the ENTRY checkpoint and NOTHING ELSE.** The offline harness
> cancels *before* `run()` begins, so a package whose mid-run cadence is entirely missing — no
> per-frame check, no per-step check, the rollout running happily to completion after a cancel —
> still passes CAN-1 and CAN-2. CAN-3 only checks that you *declared* a cadence; the declaration
> is unverified prose. So the gate's own documentation of record can be a claim nobody tested.
>
> **Don't wait for the app harness to find that out — a live mid-run probe is a CLI gate mode**
> (`swift run … --cancel`), and the CLI lane does real GPU inference (see swift-port-parity.md,
> "Where gates can actually run"). Shape: start a deliberately long run, `Task.cancel()` after a
> second or two, assert on **both** axes —
>
> - **type**: the surfaced error is `CancellationError`, *unwrapped* (laundering into a package
>   error is the CAN-2 failure the offline gate does catch, but only at the entry seam);
> - **latency**: time-to-throw ≈ one unit of the declared cadence, not the full run. Audio8-TTS
>   cancelled at t=1.500 s and threw at **1.53 s** — ~30 ms ≈ one frame, against >20 s
>   uncancelled. That 30 ms is the evidence; without it "cadence: generate/frame" is a guess.
>
> The app-harness `[CAN]` bench (item 3) is still the richer instrument — phase attribution,
> re-run recovery. This is the cheap thing that stops a broken cadence shipping to it.

## 5b. Inference mode (engine ≥ 0.36.0 / contract 1.27.0 — the C14 INF gate)

**`MLXNN.Module.training` defaults to `true`.** In that state `BatchNorm.callAsFunction`
normalizes by the CURRENT batch's statistics *and overwrites* the checkpoint's
`running_mean`/`running_var` on every forward (`MLXNN/Normalization.swift`: the
`if self.training, let runningMean, let runningVar` branch). A port that never calls
`model.train(false)` therefore runs inference on per-image statistics, never reads the trained
statistics at all, and **drifts across successive calls on one instance**.

This one already lived in `swift-port-parity.md` as a bullet — and a PROD package shipped it
anyway, for months, through in-app validation. That is the lesson: **as prose it did not hold; it
needed a gate.** Hence C14.

**Why it survives everything else you run.** It passes the key contract (the frozen running-stat
keys load fine — freezing affects `trainableParameters()`, not `update`). It passes offline
conformance, which never runs a kernel. It passes an eyeball: a matte still looks like a matte, a
denoise still looks like an image. In `mlx-birefnet-swift` the PROD tier over-segmented by **68%**
(foreground fraction 0.42 vs a PyTorch oracle's 0.25) with e2e logits cosine **0.264**, and every
human who looked at it said "fine."

**The diagnostic signature** — worth memorizing, because it localizes in one read:

> the LayerNorm/RMSNorm parts of the graph are bit-clean while everything downstream of the first
> BatchNorm diverges.

A transformer encoder at cosine 1.0000000 feeding a conv decoder at cosine 0.62 is this bug, not a
layer-translation bug. Corollaries that also point here: the divergence is **dtype-independent**
(fp16 ≈ fp32), and patching a weird value *inside* `running_var` changes **nothing** — which is
itself proof the running stats are never read.

Three requirements:

1. **Set eval mode at the single construction choke point**, not per call site. Find the one place
   every load path funnels through (the pipeline/wrapper initializer) and call `model.train(false)`
   there, so `fromPretrained`, convenience helpers, direct init, the package wrapper and the CLI
   gates all inherit it. Per-forward calls are the anti-pattern: the next entry point added won't
   have one.
2. **Expose the loaded graph to the gate.** A `var inferenceModeGraphs: [String: MLXNN.Module?]`
   seam keyed by tier/role (reached via `@testable`, with the `InferenceModeInspectable`
   conformance living in the *test* target so the shipping target takes no dependency on the
   conformance library). Key it by role, not by hardcoded tier names, or a package that later
   grows a checkpoint family silently stops covering the new one.
3. **Assert the gate can fail.** The test that matters is not "the loaded model reports
   `training == false`" — it is that a **freshly constructed** model FAILS INF-1 and passes only
   after going through the choke point. Without that, deleting the fix leaves the suite green.

Only nets carrying running statistics are affected (`BatchNorm`, `InstanceNorm` with
`track_running_stats`). Dropout is identity-safe. Pure LayerNorm/RMSNorm transformer stacks are
immune — which is exactly why LLM/VLM ports never surfaced this and conv/audio ports did.

## 6. Worked example — a `ModelPackage` conformer

Built on the real Gen-B signatures. The model graph stays in `<Name>Core` (Path B) or is the reused
`MLXLLM`/`MLXVLM` loader (Path A); the conformer below is the thin declaration layer — aim for this
ratio.

```swift
// Real Gen-B signatures — the shape every shipped image-era package uses
// (LensT2IPackage, QwenImageEditPackage). Core stays in its own target.
import MLXToolKit
import QwenImageEdit   // Path B: your ported Core. Path A: import MLXLLM and reuse its loader.

/// Init-time configuration (C9): where weights live + generation defaults.
public struct QwenImageEditConfiguration: PackageConfiguration, ModelStorable {
    public var snapshotPath: String
    public var defaultSteps: Int
    public var modelsRootDirectory: URL?   // ModelStorable — the engine stamps the store root
    // … memberwise init + CodingKeys
}

@InferenceActor
public final class QwenImageEditPackage: ModelPackage {
    public typealias Configuration = QwenImageEditConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit), // C7+C8
            provenance: Provenance(sourceRepo: "Qwen/Qwen-Image-Edit-2511",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // residentBytes from a real measured run (see memory-harness.md)
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 60_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max),
            specialties: [],
            surfaces: [IEditContract.descriptor(name: "qwen-image-edit",
                                                summary: "…hand-tuned for the model audience…")])
    }

    private let configuration: Configuration
    private var generator: QwenImageEditGenerator?      // host-owned lifecycle; no global singleton

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {                   // engine calls this; package never self-loads
        guard generator == nil else { return }          // idempotent
        generator = try await … // build Core components from configuration.snapshotPath
    }

    public func unload() async { generator = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .imageEdit, let edit = request as? IEditRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
        let (pixels, w, h) = try generator.generate(/* canonical request fields */)
        try Task.checkCancellation()
        return IEditResponse(image: Image(format: .png, data: encodePNG(pixels), width: w, height: h))
    }
}

// Registration happens at the call site:
//   try await engine.register(PackageRegistration.of(QwenImageEditPackage.self),
//                             configuration: QwenImageEditConfiguration())
```

What *didn't* move into the conformer: the RoFormer/Demucs graph stays in `*Core`. The conformer is
~40 lines of declaration + dispatch. That ratio is the target for every port.

## 7. Discovery & "dynamic" loading on Apple platforms (set expectations)

There is **no `dlopen` of arbitrary signed SPMs** under sandbox/notarization. "Loaded as needed" is
realized as three independent axes, not runtime plugin loading:

1. **Link-time set.** The host links the ports it ships and registers their `PackageRegistration`s.
   `ModelFactoryRegistry` finds runtime factories via `NSClassFromString` trampolines, so **linking
   the product is enough** — no manual registration — and it auto-dispatches by `config.json`.
2. **Capability gating (runtime).** Admission omits any package whose `DeviceProfile.eligibility` or
   license gate fails on *this* machine, so an orchestrator only sees packages that can run here.
3. **Weight laziness (runtime).** No weights load at registration; `prepare`/`run` triggers a lazy
   load, and `MemoryGovernor` LRU-evicts idle residents under pressure.

True out-of-process / third-party tools are a future transport concern handled by the engine's
bridge with the same manifests — never by changing `Core` or the conformer.

## 8. Definition of done (per port)

A port is harness-ready when: it's a `-swift` SPM with a transport-free default product; exactly one
`ModelPackage` declares the manifest and registration; lifecycle is host-owned with no internal
caching; every variant declares `license` + `minUnifiedMemory` + assets; the license gate is
two-layer; `run(_:)` dispatches on capability and never throws non-`PackageError` across the
boundary — with ONE sanctioned exception: `CancellationError` rethrown **unchanged** from its
cooperative checkpoints; the `Configuration` declares `WeightSourcing` and the package's suite
passes the offline **MAT-1..5 gate** (section 4), the offline **CAN-1..3 gate** (section 5), and —
for any net carrying BatchNorm/running statistics — the **C14 INF gate** (section 5b), with a test
that proves a freshly constructed model FAILS it and passes only via the construction choke point;
and **elementwise parity + a committed memory report** pass on at
least one admissible tier. At that point Stage 2 integration is a single
`register(registration, configuration)` call into `MLXServeEngine`.
