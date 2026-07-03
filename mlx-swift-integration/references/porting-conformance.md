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

## 4. Worked example — a `ModelPackage` conformer

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

## 5. Discovery & "dynamic" loading on Apple platforms (set expectations)

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

## 6. Definition of done (per port)

A port is harness-ready when: it's a `-swift` SPM with a transport-free default product; exactly one
`ModelPackage` declares the manifest and registration; lifecycle is host-owned with no internal
caching; every variant declares `license` + `minUnifiedMemory` + assets; the license gate is
two-layer; `run(_:)` dispatches on capability and never throws non-`PackageError` across the
boundary; and **elementwise parity + a committed memory report** pass on at least one admissible
tier. At that point Stage 2 integration is a single `register(registration, configuration)` call
into `MLXServeEngine`.
