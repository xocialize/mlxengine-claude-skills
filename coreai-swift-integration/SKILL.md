---
name: coreai-swift-integration
description: Take a finished Core AI `.aimodel` asset onto Apple silicon in Swift and integrate it into MLXEngine (MLXServeCore/MLXServeEngine) alongside the MLX siblings. Covers the Swift CoreAI runtime API (AIModel, InferenceFunction, NDArray, SpecializationOptions), SPM setup for a `coreai-*-swift` package, vendoring an `.aimodel` DIRECTORY as a resource, the prepare-seam placement of first-load E5RT cost, wrapping a CoreAI core as an MLXEngine ModelPackage with MLXToolKit only (zero MLX linked), the macOS-26-host vs 27-floored-package SPM conflict and the ExternalRegistration injection seam that resolves it, and publishing to the `coreai-community` HF org. Trigger phrasings — "package the aimodel", "CoreAI Swift package", "InferenceFunction", "SpecializationOptions", "register CoreAI backend", "ExternalRegistration", "coreai-community", "publish the aimodel", "CoreAI ModelPackage", "CoreAI conformance". Runs AFTER `coreai-porting` (which produces and validates the asset). Do NOT use for the export/parity layer.
---

# Core AI → Swift → MLXEngine

The packaging and integration layer. `Skill("coreai-porting")` produces and validates the
`.aimodel`; this skill ships it.

**Related:** `Skill("coreai-porting")` (export, parity, placement) ·
`Skill("mlx-swift-integration")` (the MLX sibling — its C0–C13 gate is the model we are trying
to reach) · `Skill("mlxengine-implementation")` (the app-consumer side) ·
`Skill("working-with-coreai")` (Apple's vendored reference).

> **Status: thin, and honestly so.** This skill rests on **one** shipped package
> (`coreai-realesrgan-swift`). The MLX sibling has a C0–C13 conformance gate derived from many
> packages; **CoreAI has no equivalent yet** and one cannot be invented from a single port. See
> `references/conformance.md` for the derivation plan.

Evidence tags — **MEASURED** / **INHERITED** / **ASSUMED** / **OPEN** — as in `coreai-porting`.

---

## Swift runtime API — the parts the docs don't state

**MEASURED** on macOS 27.0 / Xcode 27 beta.

```swift
import CoreAI

let model = try await AIModel(contentsOf: url,
    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
guard let fn = try model.loadFunction(named: "main") else { return }

var input = NDArray(shape: [1, 3, 128, 128], scalarType: .float16)
input.mutableView(as: Float16.self).withUnsafeMutablePointer { /* fill */ }

let outputs = try await fn.run(inputs: ["x": input])
```

- The function type is **`InferenceFunction`** (from `CoreAIRuntime`, re-exported by `CoreAI`).
  It is **NOT** `AIModel.Function` — that name does not exist and the compiler error is
  unhelpful.
- Compute preference: `SpecializationOptions(preferredComputeUnitKind: .neuralEngine / .gpu /
  .cpu)`. **Passing it is not the same as getting it** → `coreai-porting`'s
  `placement-and-residency.md`. The Swift side has the same silent-fallback exposure as Python.
- Outputs come back as a **dictionary**. Key order is not guessable — probe `"output"` then
  `"out"`, or better, read the names recorded in the asset metadata at export time.

---

## SPM setup

**MEASURED:**

- `platforms: [.macOS("27.0")]`
- `linkerSettings: [.linkedFramework("CoreAI")]`
- swift-tools **6.2**
- An `.aimodel` is a **DIRECTORY** — vendor it with `.copy(...)`, not `.process(...)`.

Asset size is checkpoint-scale (MEASURED: 431 MB fp16 for a 226M model). For anything past
toy size, download at runtime into the shared model store rather than vendoring —
→ `Skill("mlxengine-implementation")` for the store and downloader seam.

---

## First load belongs at the prepare seam

**MEASURED:** first load per (model × machine) pays E5RT specialization — **~8 s** at 1.4M
params, **254 s** at 226M — then OS-caches (~0.2–1.1 s after).

Pay it in your package's `load()` / prepare seam, **never** inside the first user-visible
inference, and surface progress for the large case. This is a product-visible constraint, not
an implementation detail.

---

## Engine integration without linking MLX

**MEASURED.** An MLXEngine package wrapping a CoreAI core needs **MLXToolKit only** — the
engine's dependency-free contract layer. **Zero MLX linked.** The engine coordinates; it does
not care which framework does the arithmetic.

### The deployment-target conflict, and the seam that resolves it

**MEASURED.** A macOS-26 host package (e.g. `ForgeCore`) **cannot** depend on a 27-floored
CoreAI package — SPM refuses outright. CoreAI requires macOS 27; the host may not be there yet.

The working pattern is an **injection seam** (`ForgeCore.ExternalRegistration`):

1. The **app's own deployment target** decides whether CoreAI is available.
2. The app injects a registration closure into the host package.
3. The backend registers **beside** its MLX sibling, under its own `PackageID`.

This keeps the 27 floor at the app boundary instead of propagating it down the dependency graph,
and it is what makes "ship both runtimes and route" practical.

→ `references/engine-integration.md`

---

## Publishing

**MEASURED.** HF org `coreai-community`, naming `<Model>-CoreAI`.

**The 403 trap:** org membership approval **and** adding the org to your fine-grained token are
**two separate steps**, and both 403 *identically* when missing. Check both.

Ship the model card with parity numbers and the export script. **Fresh-download-verify before
announcing.**

→ `references/publishing.md`

---

## Reference map

| File | Read it when |
|---|---|
| `references/engine-integration.md` | Wiring a CoreAI core into MLXEngine beside MLX packages |
| `references/publishing.md` | Publishing to `coreai-community`; model-card contents |
| `references/conformance.md` | Asking "is this package engine-pluggable?" — and the plan to derive a real gate |
