# Engine contract & conformance review (folded from the `mlx-engine` skill)

MLXEngine (**MLXServeCore** / `MLXServeEngine`) is the Xocialize-owned, community-released
runtime **coordinator** for on-device Apple Silicon inference. It does **not** do inference —
**packages** do. The engine's job is the coordination *around* packages: admission + queuing,
model residency + memory governance, serialized execution, the two-layer license gate, and a
registry presenting one **common way to engage every model**. This file is the conformance-review
half of the skill: contract design, the C0–C13 gate, and reviewer judgment calls.

Ground-truth docs (the repo-owned spec; these WIN over this file when they disagree):
- `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/architecture.md` — authoritative spec.
- `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/capability-contract.md` — per-capability
  canonical schemas.
- `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/conformance.md` — C0–C13 in full
  (pass/fail criteria, failure modes, legible-error requirements).

## The one distinction everything hangs on

- **Capability** — the *contracted tool surface*. Core-owned, **additive** enum (tts, textToImage,
  imageEdit, textToVideo, llm, imageAnalysis, videoAnalysis, audioSeparation, … — 18 cases at
  contract 1.2.0). One canonical input schema + one canonical output artifact per capability.
- **Mode** — a *per-request tag within a capability* (rides the request envelope). Modes are
  **never** minted as separate surfaces — that's the combinatorial-explosion failure.
- **Specialty** — *model-level metadata* the Model Manager uses to rank/select. Governed
  vocabulary, multi-valued with strength. Never a surface.

> **Capability is the contract. Mode is the request. Specialty is the advertisement.**

One model, N surfaces = **one** `ModelPackage` (license/requirements/specialty declared once on
the manifest; `run(_:)` dispatches on `request.capability`). A T2I→T2V pipeline is two discrete
tool calls that happen to resolve to the same loaded model.

## Canonical output is fixed; `metaData` is the flex

| Capability | Canonical output |
|---|---|
| tts / audioSeparation / audioPolish | Audio (.wav) |
| textToImage / imageEdit / imageRestore / imageUpscale | Image (.png/.jpeg) |
| textToVideo / videoUpscale / frameInterpolate | Video |
| llm | text |
| imageAnalysis / videoAnalysis / speechEmotion / imageQualityScore / contentClassify | structured text |
| audioCodec | codes |
| opticalFlow | flow |

**The governance line conformance polices:** `metaData` is for *genuinely package-specific
extras*, never for anything that *should be canonical*. If two TTS ports both smuggle `voice`
into `metaData` under different keys, the TTS schema needed a `voice` field — reject the smuggle,
propose the schema revision (C5).

## Two parameter planes (never conflate)

- **Init-time** → `PackageConfiguration` (Codable, defaultable; C9). Stable for the session:
  weights path/id, quant, defaults. The engine stamps `modelsRootDirectory` onto `ModelStorable`
  configs.
- **Per-request** → the canonical request envelope: mode tag, sampler knobs, input artifacts,
  `metaData`. If a "config" value changes per call, it belongs in the envelope. You never re-init
  a package to switch mode.

## V1 artifact rule

Canonical artifacts (`Image`, `Audio`, `Video`) cross every boundary **serialized** (C3). The
in-process zero-copy path (tensor / IOSurface) is a V2 additive optimization — don't let a
contributor fork the artifact type with a live-reference fast path.

## Eligibility & selection

`RequirementsManifest` (footprints per quant, required backends, min OS, chip floor) is what a
model *costs to run*; the `ToolDescriptor` is what it *can do* — keep them separate (C10 vs C2).
The Model Manager filters the registry by capability ∩ device eligibility, ranks by specialty +
footprint, and exposes both the ranked eligible set (apps that pin) and a "best eligible X"
convenience. **Multi-package per capability is live** (engine-side `PackageID` registry,
2026-06-12): several packages can back one capability concurrently; routing uses the capability's
default or an explicit per-request `package:` — the seam the lower-tier/full-tier pairs (ERNIE-
Turbo vs Lens on `textToImage`) plug into.

## The C0–C13 gate (summary — `conformance.md` is ground truth)

- **C0** Contract version declared
- **C1** Capability registration (≥1 canonical enum case; multi-capability registers each surface)
- **C2** Canonical schema conformance (correct canonical output artifact)
- **C3** Canonical artifact I/O (serialized round-trip form)
- **C4** Mode-as-parameter (no mode masquerading as a surface)
- **C5** `metaData` hygiene (no should-be-canonical smuggles)
- **C6** Specialty declaration (governed vocabulary; never a surface)
- **C7** Weight license gate (passes `.permissiveOnly`)
- **C8** Port-code license gate (distinct layer; failure names which layer)
- **C9** `PackageConfiguration` conformance (init-time, Codable, defaultable)
- **C10** Requirements manifest (footprint per quant, backends, min OS, chip floor)
- **C11** Introspection (each surface exposes a valid, well-described schema; MCPBridge consumer)
- **C12** Forward-compat discipline (`@unknown default` on capability/quant switches)
- **C13** Runtime governance cooperation (engine constructs/owns; `@InferenceActor` execution;
  cooperatively evictable; no private queues or pinned compute; no self-caching)

Review stance: each item is a reviewable pass/fail — point at the C-level, not an opinion.

Since engine 0.19.0 the gate has an **executable adjunct — the MAT gate** (MAT-1..5): the
package's own conformance suite runs
`MLXServeConformance.MaterializationConformance.check(freshConfiguration:satisfiedConfiguration:)`
to prove the auto-materialization declarations offline — `WeightSourcing` declared, role/repo
hygiene, honest fresh-machine missing set, explicit-paths-satisfy. It reviews like a C item:
pass/fail straight from the report. Requirements + the MLXLTX2 reference implementation:
`references/porting-conformance.md` §4.

## Versioning the contract

The conformance spec is public API. `Capability`/`Quant` and C-levels are **additive-only at
minor versions** (1.1.0: `referenceTranscript`, int5/int6; 1.2.0: `imageEdit` + IEdit types);
breaking revisions need a major bump + deprecation window. Every package declares its target
version (C0). The additive guarantee is paid for by C12 discipline — when you bump the contract
with a new `Quant`, grep every package's `switch` over `Quant` before publishing.

## When to stop and ask the user

- A model exposes a capability **not in the enum** — the enum is core-owned; an addition is a
  versioned core change (the user opted into 1.2.0's `imageEdit` explicitly), not a package call.
- Two conformant packages need **incompatible I/O for one capability** — fix the canonical
  schema, don't waive it.
- A should-be-canonical parameter with no schema field yet — propose the schema revision; don't
  bless a `metaData` smuggle.
- Weight license and port-code license disagree — hard stop at C7/C8.

## Feed process gaps back

When a real port surfaces a gap, judge it on one axis: package-specific (fix in the package) vs
generally useful (fix the process **in the same pass**): schema gaps → `capability-contract.md`
(+ `MLXToolKit` types, versioned); ambiguous checks → `conformance.md`; workflow gaps → the
SKILL.md; integration traps → `integration-lessons.md`. The bar: the next contributor hitting the
same wall finds the answer already in the gate.
