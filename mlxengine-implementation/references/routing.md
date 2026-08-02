# Topic 4 — Capability routing & multi-package selection

## One capability, possibly many backers

A `Capability` (e.g. `textToImage`) can be backed by **several registered packages** at once — Lens AND
ERNIE-Turbo both serving `textToImage`, or the same package registered twice at different quants. This is
"modularity on top of the engine": the app decides which modules it wants per capability, and routes per
request. The engine keys residents by **package**, so backers co-reside (subject to memory + R-MEM-1) and
one registration serving N capabilities is constructed once and shared.

## PackageID — the engine-side identity

`register` returns a `PackageID`. It defaults to the manifest's first surface name (`"lens-t2i"`,
`"qwen-image-edit"`), falling back to `provenance.sourceRepo`. Pass an explicit `id` to register the
**same package twice** — the canonical "bf16 vs int4 of one model" case:

```swift
let quality = try await engine.register(MyT2I.registration,
                                        configuration: MyConfig(quant: .bf16), id: "myt2i-bf16")
let fast    = try await engine.register(MyT2I.registration,
                                        configuration: MyConfig(quant: .int4), id: "myt2i-int4")
```

`PackageID` is `ExpressibleByStringLiteral`, so `"myt2i-int4"` works wherever a `PackageID` is expected.

## Routing API

```swift
engine.packages(for: .textToImage)       // [PackageID] — every backer, in registration order
engine.defaultPackage(for: .textToImage) // PackageID? — where the capability routes by default
try engine.setDefault(fast, for: .textToImage)  // re-point routing WITHOUT re-registering
```

- **Registering a backer makes it the new default** ("last registration wins routing" — preserves the
  historical swap flow). So if you register quality then fast, `fast` is the default until you say
  otherwise.
- **`setDefault` is how you switch the active variant** — not re-registering. It throws
  `.unknownPackage(capability, id)` if `id` isn't a backer of that capability.

## Per-request / per-prepare selection

Both `run` and `prepare` take an optional `package:` to target a specific backer regardless of the
default — the seam for an A/B toggle or a "use the fast one just this once":

```swift
let resp = try await engine.run(request, package: fast)   // this call uses int4…
try await engine.prepare(.textToImage, package: quality)  // …while quality is warmed for the next
```

Omitting `package:` uses the capability's current default.

## Eviction is package-scoped too

```swift
await engine.evict(.textToImage, package: fast)  // evict that backer; the default stays resident
await engine.evict(package: fast)                // evict by id regardless of capability routing
await engine.evict(.textToImage)                 // evict the capability's default
```

## App patterns

- **Quality/Fast toggle** — register both variants once at startup; a UI switch calls
  `setDefault(_:for:)`. No re-registration, no reload churn beyond admission.
- **Model picker** — list `packages(for:)`, show each via its `manifest(for:)` (provenance, footprint,
  `admissibility(for:configuration:)` from topic 3), let the user pick the default.
- **A/B compare** — keep both resident, drive each call with `run(_, package:)`, compare outputs.

## PITFALL — linking MLXVLM anywhere in the app shadows text architectures for ALL packages

Combining packages in one app has a **link-time** hazard the engine cannot protect you from:
mlx-swift-lm's `ModelFactoryRegistry` is **process-global** and probes `MLXVLM.TrampolineModelFactory`
BEFORE `MLXLLM` (`Libraries/MLXLMCommon/ModelFactory.swift:480` via `NSClassFromString`). Architectures
registered in BOTH factories — e.g. `"gemma3"`: `VLMModelFactory.swift:91` (multimodal Gemma3) vs
`LLMModelFactory.swift:31` (`Gemma3TextModel`) — resolve to the **VLM variant** once MLXVLM is linked
anywhere in the process.

This fires at **link time, even if the MLXVLM-linking package is never registered or run**. Real
incident (2026-07-01, BRIDGE-LTX-003 in `~/Development/mlxengine-video-ltx/AGENT_BRIDGE.md`):
registering `mlx-qwen-llm-swift` (which links MLXVLM to serve VL checkpoints) into the LTX app made
LTX's own `GemmaEncoder` auto-dispatch load multimodal Gemma3 and fatal-error on its `Gemma3TextModel`
cast.

What the app integrator does about it:

- **Audit before combining**: before adding any package to an app that already loads text models by
  architecture name (or vice versa), check whether the newcomer's dependency tree pulls `MLXVLM`
  (`swift package show-dependencies`, or grep its `Package.swift` products).
- **Prefer host-model closures from utility kits**: kits like `prompt-enhance-kit-swift` (v0.3.0's
  `generate(system:user:)` overload + README warning) let the HOST supply the generate function, so the
  kit never obliges linking a model package at all.
- **Report brittle packages upstream**: a package that auto-dispatches via `#huggingFaceLoadModel…` and
  then `fatalError`s on a failed downcast (instead of throwing a typed error) turns this silent
  shadowing into a crash — file it against the package (mitigation belongs there, see the
  `mlx-swift-integration` skill's `integration-lessons.md`).

## Recognizing the anti-pattern (review checklist)

- [ ] Re-registering a package to "switch to int4" instead of registering both once and `setDefault`-ing.
- [ ] Assuming one package per capability — hardcoding a single backer, so a second module can't be added.
- [ ] Relying on registration order implicitly instead of calling `setDefault` to make routing explicit.
- [ ] Building a model picker that re-registers on every selection (churns residency) rather than
      switching the default.
- [ ] Adding a package that links MLXVLM to an app whose other packages load text models by architecture
      name, without auditing the shared-name collision (`gemma3`, `qwen3`, …) — see the MLXVLM-shadowing
      pitfall above.

## Verify against source

`mlx-engine-swift/Sources/MLXServeCore/MLXServeEngine.swift` (`PackageID`, `register`, `packages(for:)`,
`defaultPackage(for:)`, `setDefault`, `resolve`, `run`/`prepare`/`evict` `package:` params,
`manifest(for:)`).

## Field catch (2026-07-09, ModelSheet Studio): capability overlap rots default routing

Packages GAIN surfaces across versions — z-image-swift v0.2.0 added an `imageEdit` (img2img)
surface, so an app that registered Z-Image after klein silently swapped its edit backer and
`prepare(.imageEdit)` tried to load the wrong snapshot (`unreadableSnapshot`). Registration-order
tricks ("register X last so it defaults") are load-bearing on the *current* pin's surface list and
break when a pin moves. In any app registering ≥2 image packages: capture the `PackageID` from
every `register(...)` and pass it explicitly to `run`/`prepare`/`needsDownload`/`ModelStateView`
(`package:` takes the ID's `rawValue` string in the UI views). Reserve capability-default routing
for single-backer capabilities.
