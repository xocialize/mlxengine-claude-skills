---
name: mlx-swift-integration
description: Port a Python-MLX model into Swift and integrate it into MLXEngine (MLXServeCore/MLXServeEngine) — the Xocialize coordinator that drives MLX-Swift packages but does no inference itself. Use whenever taking an MLX-available model onto Apple Silicon — porting Python-MLX to a Swift-MLX `Core`, wrapping it as a `ModelPackage` (PackageManifest, RequirementsManifest, two-layer license gate, measured split QuantFootprint — resident weights + activation — adopting the 1.14 efficiency contract AND the 0.19.0 WeightSourcing auto-materialization/MAT gate AND the 0.27.0 cancellation/CAN gate born-clean), deciding capability-vs-mode-vs-specialty, consuming `mlx-swift-lm`, scaffolding a `-swift` SPM, wiring it into the test app, driving register/prepare/run via `MLXServeEngine` (multi-package per capability via PackageID), or reviewing against the C0–C13 conformance gate. Trigger phrasings — "port to Swift-MLX", "integrate into MLXEngine", "is this package engine-pluggable", "review against C0–C13", "capability vs specialty", "PackageConfiguration", "WeightSourcing", "MAT gate", "CAN gate", "auto-materialize weights", "honor cancellation". Runs AFTER `mlx-porting` (PyTorch→Python-MLX parity); do NOT use it for that layer-translation step. Supersedes the `mlx-engine` skill.
---

# Porting a model to Swift-MLX and integrating it into MLXEngine

## Where this sits in the pipeline

```
PyTorch / Python ──[mlx-porting skill]──► Python-MLX ──┐
                                                       │  THIS SKILL
                          ┌────────────────────────────┴───────────────────────────┐
                          │  Stage 1: port Python-MLX → Swift-MLX (a conformant       │
                          │           ModelPackage in a -swift SPM)                   │
                          │  Stage 2: integrate that package into MLXEngine and prove │
                          │           it in a consuming app                           │
                          └──────────────────────────────────────────────────────────┘
```

`mlx-porting` **creates** the MLX port (PyTorch→MLX layer translation, weight conversion,
Python parity). This skill **takes that result to Swift and into the engine.** If the model
isn't MLX-available yet, do that first with `mlx-porting`. (A Python-MLX reference that already
exists upstream — e.g. mflux — counts as "MLX-available": gate the Swift port against IT plus
PT goldens; no separate Python port needed.)

> **The former `mlx-engine` skill is folded into this one** as
> `references/engine-contract.md` (capability/mode/specialty, canonical outputs + metaData
> governance, the C0–C13 summary, contract versioning, reviewer stop-and-ask). The repo-owned
> spec docs under `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/` (architecture.md,
> capability-contract.md, conformance.md) remain ground truth — when this skill and those
> disagree, the repo docs + live `MLXToolKit` source win; update this skill's references.

This is a **living lessons file**. Append new gotchas to `references/integration-lessons.md` as
each port teaches them. Keep heavy, project-specific detail (entitlements, exact versions, the
full play-by-play) in `~/Development/MLXEngine/EngineeringDocs/CLAUDE.md` and
`~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/{first-integration-notes,conformance}.md`. Keep this file the
fast, reusable workflow + the router to the references.

## When to use / skip

- **Use** when: porting a Python-MLX model to a Swift `Core`, wrapping a model as a `ModelPackage`,
  writing a `PackageManifest` / `RequirementsManifest`, consuming `mlx-swift-lm`, scaffolding a
  `-swift` package into the MLXEngine workspace, declaring/measuring footprints, routing
  downloads into the models folder, or driving `register()` / `prepare()` / `run()` from the
  app/engine.
- **Skip** when: doing the PyTorch→MLX port itself (→ `mlx-porting`), or pure contract design of
  `MLXToolKit` with no model attached.

## Core mental model

1. **MLXEngine is a coordinator, not an engine.** A contribution is ONE `ModelPackage` declaring a
   `PackageManifest`; the engine constructs / loads / drives / evicts it. The package never
   constructs or caches itself — inversion of control (C13).
2. **Two layers, kept apart.** The **contract** (`MLXToolKit` — pure protocols + value types, no
   heavy deps) vs the **runtime** (`mlx-swift-lm`, Metal, HF). Author against the contract first;
   add the runtime second. Proving the contract compiles **offline** before pulling the multi-GB
   MLX/Metal graph keeps every later failure localized to the runtime, not the contract.
3. **Reuse, don't port.** If the architecture is already in `mlx-swift-lm` (standard LLM/VLM) or you
   already ported a base, **reuse it** — don't re-translate layers. Only write a Swift `Core` for a
   novel architecture the runtime can't already load.
4. **One model, N surfaces = one package.** Declare license / requirements / specialty once on the
   manifest, dispatch the surfaces inside `run(_:)` on `request.capability`.

## Decide the path first

Before any scaffolding, answer one question: **does `mlx-swift-lm` already load this architecture?**

- **Path A — already loadable (standard LLM / VLM, or a base you already ported).** Skip Stage 1
  entirely. There is **no Swift `Core` to write** — you reuse the `MLXLLM` / `MLXVLM` loader. Go
  straight to Stage 2 and wrap the loader as a `ModelPackage`. (The "Reuse, don't port" lesson.)
- **Path B — novel architecture (audio, diffusion, custom layers `mlx-swift-lm` can't load).** Do
  Stage 1: port the Python-MLX implementation into a Swift `Core`, then Stage 2 wraps it.

Getting this wrong is the most expensive mistake in the workflow — writing a `Core` for a model the
runtime already loads is wasted weeks. When unsure, try loading via `mlx-swift-lm` first.

## Stage 1 — Port Python-MLX → Swift `Core` (Path B only)

Produce a **harness-ready, conformant** package: a `-swift` SPM whose `Core` is the ported model and
whose `ModelPackage` conformer declares the manifest. The deliverable is defined by the conformance
gate, not by "it runs once."

Read **`references/swift-port-parity.md`** FIRST for the port itself — the phase-gated workflow
(key contract → component gates → e2e golden → GPU smoke → deltas → quant), the
Python-MLX↔Swift-MLX numerics doctrine (bit-identical RNG seed streams; bit-exact scalar code;
the donor lift-vs-translate decision via flattened key paths), the **Metal-watchdog family**
(CPU-stream weight loads, GPU-stream quantized forwards, never eval giant constant fills,
ARC-scope big models), and where gates can actually run (CLI gate modes — the SPM test product's
metallib is unreliable; plain `swift run` does GPU inference fine).

> **⚠ Highest-cost gotcha — a quantized FORWARD must run on the GPU stream, never CPU.** When you
> write a P7/quant parity gate, do NOT pin the whole test to `Device.setDefault(.cpu)` the way the
> fp32 gates do. Quantized matmul is Metal-only; under a CPU pin it has no efficient path and
> **silently grinds for HOURS** (state `R`, ~100% CPU, zero output — a Z-Image quant gate ran 10 h
> before it was killed; it does NOT error or trip the watchdog, it just looks hung). Load + quantize
> on CPU is fine; run the forward on GPU (the cosine gate absorbs the ~1e-3 GPU-vs-CPU-golden fp32
> gap). Because the SPM test target's metallib is unreliable for GPU, put quant gates in the **CLI
> lane** (`swift run … --quant-gate`), not an XCTest. Detail: `swift-port-parity.md` Metal-watchdog
> family item 2.

> **⚠ Gate metric for DIFFUSION quant variants — PSNR/LPIPS against the fp reference at a fixed
> seed, never FID** (NEUROSTREAM-ACTIONS QW4, 2026-08-01). FID is blind to quantization damage:
> FLUX FID stays flat (20.3→19.9) across BF16/W8A8/NF4 while PSNR collapses 27.0→19.5 (SVDQuant,
> arXiv 2411.05007 Table 1). W8 is effectively free (PSNR 27.0 / LPIPS 0.089 vs BF16); **weight-only
> int4 on a DiT/UNet is real, measurable quality damage** — and buys no speed either, since
> diffusion is compute-bound (`quantized_matmul` saves memory, not time). int4/int8 on
> memory-bound autoregressive components (LLM backbones, token decoders) is unaffected — that
> remains the standard lever. Fleet state 2026-08-01: every image-diffusion package defaults
> `.bf16`, with int4 only as declared opt-in variants. Keep diffusion defaults ≥ int8-activation-free
> (bf16/fp16/W8), and never let a FID number admit a quant variant.
Read **`references/porting-conformance.md`** for the full topology, the `MLXToolKit` surface you
implement against, the per-port conformance checklist (C0–C13), and a worked `ModelPackage` example.
Read **`references/memory-harness.md`** for how to produce the empirical `minUnifiedMemory` every
variant must declare — peak active unified memory at the input envelope, not weight size.

Stage 1 is done when the package satisfies the conformance gate **and** every variant carries a
committed memory report. Critically: **offline conformance never runs a kernel.** A green Stage 1
does not mean the model is correct — the first real forward pass happens in Stage 2's app harness,
which is where the silent-failure class surfaces (see `references/integration-lessons.md`).

## Stage 2 — Integrate into MLXEngine

```
1. Confirm availability + license   → mlx-community repo exists (or FoundationModels); BOTH license layers permissive
2. Scaffold the -swift SPM          → package name carries -swift; module stays clean PascalCase; depends on MLXToolKit
3. Author the contract side         → Configuration + Manifest + ModelPackage conformer; declare the SPLIT
                                      footprint + QuantConfigured (+ BudgetAware if there's a dtype lever) NOW,
                                      born sweep-clean, + WeightSourcing sources (quant-aware globs), born
                                      materialization-clean (see below); build OFFLINE vs MLXToolKit + tiny tests
4. Add the runtime + load/run       → mlx-swift-lm (+ HF stack); implement load()/run() — the ENGINE materializes
                                      dir-less configs from the declared sources pre-load() (contract 1.24;
                                      SelfMaterializing = opt-out), load() just loads; resolve packages; build
5. Link into the app + smoke test   → manual Xcode "+"; registration → license gate → load → run (console first)
6. Storage integration              → download into the chosen models folder + write mlx-package.json marker + refresh panel
7. Promote to MLXServeCore          → register/admission via MLXServeEngine (multi-package per capability:
                                      PackageID selection, setDefault, per-request package), then a
                                      chat/validation UI in the consuming app
8. Register in the model registry   → add/update the package row in mlx-engine-swift/docs/model-registry.md
                                      (capability · model · role · home · Avail/Val/Eff/Eng) IN THE SAME
                                      CHANGE — the registry is maintained by integration, never regenerated
```

**Never skip step 3 (offline contract build).** Proving the contract compiles before pulling the
graph is the whole point of the two-layer split.

**Integrate born sweep-clean — adopt the 1.14 efficiency contract AT integration, not as a later
retrofit.** A freshly-wrapped package should declare the **split footprint** (`residentBytes` weights floor
+ a *measured* `peakActivationBytes`), `QuantConfigured`, `BudgetAware` wherever a quality/dtype lever exists
(e.g. a near-lossless int8 the engine can drop to under pressure), `unload()` → `MLX.Memory.clearCache()`,
and mmap/lazy weight load — all in step 3's manifest + load/unload. Doing it now means the package never
enters the efficiency-sweep backlog (the alternative is a whole separate retrofit pass later). A flat
`residentBytes` that bakes the activation into residency is the anti-pattern this avoids. The four levers,
the per-stage-evict pattern for multi-component pipelines, and the measurement traps (in-app `phys_footprint`
vs smoke MLX-peak ~2.7×; measure the floor **post-load** not post-run; flat-vs-climbing retention) are all in
**`references/package-efficiency.md`** — read it during step 3, not after.

**Integrate born materialization-clean too — the v0.19.0 auto-materialization contract adopts the
same way.** In step 3 the `Configuration` conforms to `MLXToolKit.WeightSourcing` — declare every
fresh-machine weight source (`WeightSource{role, repo, revision, matching globs}`; quant-tiered
configs EXCLUDE files their quant doesn't need) and implement `missingWeightSources(storeRoot:)`
(explicit local paths first, then the ModelStore `<root>/<org>/<name>` layout). In step 4 `load()`
auto-materializes missing sources into the store layout when explicit dirs are nil (native
downloader, per-file progress via `WeightDownloadProgress.report` so the engine surfaces
`.downloading` — a silent download is a conformance smell), and `prewarmPaths` resolves against
the store so later cold launches prewarm downloaded weights. The package's own suite runs the
offline **MAT-1..5** gate next to its C0–C13 tests. Full requirements + the MLXLTX2 reference
implementation: **`references/porting-conformance.md` §4**; the consumer/app side (folder pick,
`needsDownload` routing, progress UI, live `MaterializationBench`) is the `mlxengine-implementation`
skill's `references/materialization.md`.

**And integrate born cancel-clean — the CAN gate (engine ≥ 0.27.0) adopts the same way.** In
step 4, `run()` honors cooperative cancellation: `try Task.checkCancellation()` is the **first
act** of `run()` (before `notLoaded` validation), then at every natural yield point (per denoise
step / decode chunk / generated token / frame — the LTX-proven placements), rethrowing the
`CancellationError` **unchanged** (never wrapped in a package error — the engine disambiguates
user-cancel from governor-preempt by the type; user cancel surfaces `.cancelled` to the caller,
governor preempt requeues). Report `RunProgress` at the same seams (contract 1.18) — per-step
progress doubles as observable evidence of the checkpoint cadence. The package's own suite runs
the offline **CAN-1..3** gate (`MLXServeConformance.CancellationConformance`: pre-cancelled-run
propagation + classification, plus the checkpoint-cadence declaration for long-run manifests;
sub-second packages declare `.subSecondRuns(reason:)`). The live timed cancel probe is
`MLXEngineTestKit.CancellationBench` (`[CAN]`, Xcode-app harness only — replaces bespoke
LTX_CANCEL_TEST-style levers). Full requirements: **`references/porting-conformance.md` §5**.

Step 5 onward is where real bugs live. Drive every package through **one reusable validation
harness** in the app (model picker → `evict` → `register` → `prepare` (timed) → `run` (timed) →
decode), capturing load/run seconds, peak process memory, and the engine-charged footprint. **Quantify
the result — don't trust eyes or ears**; a silent stem reads −∞ dBFS, which is the tell that catches
a whole class of "looks fine, is wrong" ports.

The full, hard-won detail for every step — `mlx-swift-lm` 3.x pins and load/generate calls, the
`@InferenceActor` + Sendable rules, Metal API Validation, the silent-failure class, sandbox/storage,
`MLXServeEngine` admission, version drift — lives in **`references/integration-lessons.md`**. Read it
before and during integration; append to it after.

## The C0–C13 conformance gate

A port merges only when it passes C0–C13 (each item a reviewable pass/fail — point at the
C-level, not an opinion). The full summary, the capability/mode/specialty distinction, metaData
governance, and the reviewer stop-and-ask cases live in **`references/engine-contract.md`**;
the authoritative enumeration is `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/conformance.md`
+ the `MLXServeConformance` target. Highlights: the two-layer license gate (**C7** weight +
**C8** port-code), **C10** device eligibility, **C12** `@unknown default` on the additive enums,
**C13** inversion of control (the engine constructs the package, never the reverse). Since engine
0.19.0 the gate has an executable adjunct — the **MAT gate** (MAT-1..5, offline): the package's
own suite runs `MLXServeConformance.MaterializationConformance.check(…)` to prove its
`WeightSourcing` auto-materialization declarations (`references/porting-conformance.md` §4).

## Reference router

| Read this | When |
|---|---|
| `references/swift-port-parity.md` | Stage 1 — the Python-MLX→Swift-MLX port itself: phase-gated workflow, key contracts, donor lift-vs-translate, cross-binding RNG/bit-exactness, the Metal-watchdog family, CLI gate modes, oracle-gate cross-validation, gate-matrix input-envelope coverage (largest production grid + decoded output per tier). |
| `references/porting-conformance.md` | Stage 1 — package topology, the `MLXToolKit` contract surface, the C0–C13 per-port checklist, the v0.19.0 `WeightSourcing` auto-materialization requirements + MAT gate (§4), the worked `ModelPackage` example, discovery/loading expectations. |
| `references/memory-harness.md` | Producing the empirical `minUnifiedMemory` (C-memory item) for each variant via `MemoryProbe`; the persistent/transient footprint split; how `MemoryGovernor` admits/evicts against it. |
| `references/package-efficiency.md` | The library-revisit efficiency sweep: declare the split footprint, mmap/lazy weight load, per-stage load→use→evict, `BudgetAware` adaptive dtype. What we ask of every port (paired with the app-side seams in `mlxengine-implementation`). |
| `references/integration-lessons.md` | Stage 2 — the living gotchas checklist: `mlx-swift-lm` runtime, Metal, silent-failure class, audio/visual wrappers, sandbox/storage, `MLXServeEngine` coordinator, retrieval, version drift, build environment, wrapper-level live gates (`--e2e-<surface>-pkg`) + test-input hygiene. |
| `references/engine-contract.md` | Contract design + conformance REVIEW: capability/mode/specialty, canonical outputs, `metaData` governance, parameter planes, the C0–C13 summary, versioning, stop-and-ask. |
| `mlxengine-implementation` skill → `references/materialization.md` | The CONSUMER/app side of v0.19.0 auto-materialization — folder pick, `needsDownload` routing, progress UI, the live `MaterializationBench` `[MAT]` measurement. Package-author requirements live here in `porting-conformance.md` §4. |
| `~/Development/MLXEngine/mlx-engine-swift/docs/model-registry.md` | The living provider registry — every package's capability/home/availability/validation/efficiency state. Update the package's row as Stage 2 step 8 (it's maintained by integration, not regenerated). |
| `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/conformance.md` | The authoritative C0–C13 enumeration (ground truth for the gate). |
| `~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/first-integration-notes.md` | The full first-package play-by-play. |
| `mlx-porting` skill | The PyTorch/Python → Python-MLX port that this skill consumes. |

---

## Runtime-agnostic packages + the external-registration seam (2026-07-31)

Two patterns proven by the first non-MLX engine package (`CoreAIRealESRGAN`, GAP-PROGRAM V13-E):

1. **An engine package does not need MLX.** `MLXToolKit` (the contract layer: `ModelPackage`,
   `PackageManifest`, requests/responses) is dependency-free — a package over a CoreAI/ANE core
   depends on MLXToolKit alone and registers like any other. Conformance notes: pay expensive
   preparation (E5RT specialization) in `load()` per MAT semantics; CAN cadence = entry checkpoint
   + per-tile `Task.checkCancellation()`; `unload()` has no MLX pool to flush — dropping refs is
   enough. `RequirementsManifest.Backend.coreMLANE` is the ANE placement value (named pre-CoreAI;
   rename queued engine-side).

2. **Higher-OS-floor backends enter via injection, not dependency.** SPM refuses a lower-floored
   package depending on a higher-floored one (macOS-26 ForgeCore ↔ macOS-27 CoreAI package). The
   seam: the host service accepts `ExternalRegistration` closures (name + capability +
   `(MLXServeEngine) async throws -> PackageID`), runs them in `registerAll()` beside built-ins,
   and surfaces outcomes under the same honesty rules. The APP's deployment target decides what to
   inject; below the floor, nothing is injected and the incumbent backend serves the capability.
