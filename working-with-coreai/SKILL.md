---
name: working-with-coreai
description: Use this skill whenever the user mentions coreai-torch, TorchConverter, coreai-build, AIModel, AIProgram, .aimodel, or wants to export/compile/run a PyTorch model on Apple silicon (iPhone, iPad, Mac). Also triggers for "deploy on device", "optimize for on-device performance", onboarding new models to Core AI, or choosing between iOS and macOS deployment paths.
---

> **Provenance — vendored from Apple, not written here.** Copied 2026-07-31 from
> `apple/coreai-models` (`skills/skills/working-with-coreai`), **BSD-3-Clause, © 2026 Apple Inc.** Installed as a plain
> skill rather than via the repo's `coreai-skills` plugin, so intra-repo `Skill("coreai-skills:x")`
> references have been rewritten to `Skill("x")`. **Do not edit in place** — refresh from upstream and
> re-apply that one rewrite, or our local edits will be silently lost on the next refresh. Local
> findings belong in `mlx-porting` or `mlxengine-todo/BOUNDARIES.md`, not here.

# Working with Core AI

Deploy PyTorch models on Apple silicon: export with coreai-torch, compile with coreai-build, run with the Core AI runtime (Swift or Python).

**Related skills**: `Skill("model-authoring")` (Neural Engine and GPU authoring patterns, use when re-structuring model architecture) | `Skill("model-compression-exploration")` (quantization/palettization sweeps — use when exploring compression tradeoffs)

______________________________________________________________________

## Documentation and reference material

The Core AI toolchain has extensive documentation. Use these as reference — **do not read all pages upfront**. Instead, consult the relevant docs when you need specifics about a particular step.

| Resource | What it covers | When to consult |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [coreai-torch](https://apple.github.io/coreai-torch/index.html) | TorchConverter API, externalization, composite ops, custom lowerings, Metal kernels, debugging | Export questions, API details, custom op registration |
| [CoreAI framework](https://developer.apple.com/documentation/coreai) | AIModel, InferenceFunction, NDArray, specialization, caching | Swift runtime API, on-device integration |
| [coreai-build (AOT compilation)](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time) | Ahead-of-time compilation flags and options | Compilation questions |
| [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core) | Python runtime: AIModel, InferenceFunction, NDArray, state management | Python runtime questions |
| [coreai-models repo](https://github.com/apple/coreai-models) | Export recipes, Swift runtime utilities, reusable primitives | Export patterns, running models, reference implementations |
| [`guidance.md`](references/guidance.md) | Platform and general guidance: use cases, model sizing, compression strategy | Resolving decisions around platform targeting, model sizing, and compression strategy |

### coreai-models: the reference implementation

The [coreai-models](https://github.com/apple/coreai-models) repo is the canonical source for how to export and run models with Core AI. **Before writing export code from scratch, always explore this repo** — it has working export recipes for many model families, Swift and Python runtime utilities, and reusable primitives. If the user has a local clone, explore it. If not, suggest cloning it.

Explore these directories to find relevant patterns:

- **`models/`** — Per-model export recipes with READMEs and CLI commands for many popular model families (LLMs, vision, audio, diffusion).
- **`python/src/coreai_models/export/`** — Export pipeline code covering macOS and iOS export paths, compression presets, and custom MLIR lowerings.
- **`swift/Sources/`** — Runtime utilities for LLMs (engines, text generation, KV cache, sampling, decode loops), diffusion pipelines, object detection, image segmentation, and constrained decoding.

______________________________________________________________________

## Pipeline overview

The Core AI pipeline transforms a PyTorch model into an optimized on-device asset:

```text
1. AUTHOR        Re-structure model for target platform
                  → Skill("model-authoring")

2. COMPRESS      Explore quantization/palettization tradeoffs
                  → Skill("model-compression-exploration")

3. EXPORT        Convert PyTorch → AIProgram via TorchConverter
                  → coreai-torch docs

4. COMPILE       Ahead-of-time compilation for target platform
                  → coreai-build CLI

5. RUN           Load and run on device (Swift or Python)
                  → CoreAI framework / coreai Python API
```

Steps 1 and 2 are optional — many models export directly without re-authoring or compression. Start with export, then add authoring or compression if needed (poor accuracy, poor performance, too large).

For models already in [coreai-models](https://github.com/apple/coreai-models), the export recipes handle all steps. Check the `models/` directory first — if the user's model family is there, point them to the recipe.

______________________________________________________________________

## Export (Python — coreai-torch)

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(torch.randn(1, 3, 224, 224),))
ep = ep.run_decompositions(get_decomp_table())

program = (
    TorchConverter()
    .add_exported_program(ep, input_names=["image"], output_names=["logits"])
    .to_coreai()
)
program.optimize()
program.save_asset("model.aimodel")
```

This is the simplest export pattern. Real models often need more — consult the [coreai-torch docs](https://apple.github.io/coreai-torch/index.html) and explore the export code in the coreai-models repo for patterns around:

- **Externalization** of composite ops via `add_pytorch_module()` with `externalize_modules`
- **Mutable state** (e.g. KV cache) via `state_names`
- **Custom Metal kernels** via `TorchMetalKernel` and `register_torch_lowering()`
- **iOS static shape specialization** via `set_static_shape_config()`
- **Compression presets** for macOS vs iOS (different default strategies per platform)

______________________________________________________________________

## Compile (coreai-build CLI)

Ahead of time (AOT) compilation of models can optionally be performed with:
```bash
xcrun coreai-build compile model.aimodel --platform iOS
```

**Docs**: [Ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)

______________________________________________________________________

## Run (Swift)

```swift
import CoreAI

let model = try await AIModel(contentsOf: modelURL)
guard let fn = try model.loadFunction(named: "main") else { return }

var input = NDArray(shape: [1, 3, 224, 224], scalarType: .float32)
var view = input.mutableView(as: Float32.self)
// fill view with data...

var outputs = try await fn.run(inputs: ["image": input])
let result = outputs.remove("logits")?.ndArray
```

For LLMs, diffusion, and other complex models, explore the Swift runtime utilities in the coreai-models repo — they provide complete inference engines, decode loops, sampling, and KV cache management that handle the complexity beyond basic `AIModel` usage.

**Docs**: [CoreAI framework](https://developer.apple.com/documentation/coreai)

## Run (Python)

```python
from coreai.runtime import AIModel, NDArray
import numpy as np

model = await AIModel.load("model.aimodel")
fn = model.load_function("main")
outputs = await fn(
    {"image": NDArray(np.random.randn(1, 3, 224, 224).astype(np.float32))}
)
logits = outputs["logits"].numpy()
```

**Docs**: [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core)

______________________________________________________________________

## Verifying outputs (PSNR)

Compare on-device outputs against PyTorch reference:

| Scenario | Expected PSNR | Investigate if below |
| ------------------ | ------------- | -------------------- |
| float32 end-to-end | > 70 dB | 60 dB |
| fp16 on-device | > 50 dB | 40 dB |
| 4-bit palettized | ~40 dB | 30 dB |

______________________________________________________________________

## Onboarding new models

When helping a user deploy a model they haven't deployed before, follow this protocol. If the user already knows what they want (specific export question, compilation flag, etc.), skip directly to the relevant section above.

### First response: conversation, not code

Your first response is always a conversation. Even if the deployment path seems obvious, ask first.

**Rules:**

- Summarize what you understood (2-3 sentences)
- Ask 1-2 clarifying questions about what's unclear
- Do not recommend or dive into compute unit specific optimizations unless you have already reviewed the general and platform level guidance in [`guidance.md`](references/guidance.md)
- Do not produce compilation commands, code, or technical recommendations yet
- Only discuss what's on the user's path — don't mention irrelevant options
- Always check coreai-models for existing recipes before suggesting a from-scratch approach

**What to learn:**

- What are they building? (real-time camera, batch processing, background service, LLM chat)
- What platform? (iOS, macOS, multi-platform)
- What does the model look like? (architecture, size, input shapes, dynamic vs static)
- What matters most? (speed, energy efficiency, quality, broad device reach)

**Example interaction:**

> **User**: I want to deploy my image classifier on iPhone.
>
> **Assistant**: Got it — you're looking to run an image classification model on iPhone. A couple of questions to help me give you the best guidance:
>
> 1. Roughly how large is the model (parameter count or file size)?
> 2. Does it need to run in real-time (e.g., processing camera frames), or is batch/offline processing fine?
>
> **User**: It's a ResNet-50, about 25M params. Needs to run at 30fps on camera frames.
>
> **Assistant**: Perfect — a 25M param ResNet-50 for real-time camera on iPhone is a great fit for efficient on-device inference. Here's the plan...

### After the conversation

1. Check if the model family exists in [coreai-models](https://github.com/apple/coreai-models) — explore the `models/` directory for a matching recipe
2. If the user's needs involve platform targeting, model sizing, or compression strategy, read [`guidance.md`](references/guidance.md) to resolve the deployment path — present the outcome, not the reasoning
3. Walk through the pipeline steps relevant to their situation, consulting the docs above as needed
4. If the model needs architectural changes, invoke `Skill("model-authoring")`
5. If compression tradeoffs need exploration, invoke `Skill("model-compression-exploration")`

---

# Fleet learnings (xocialize, 2026-07-31 — NOT part of Apple's vendored skill)

Hard-won facts from shipping the first production `coreai-*` package
(`coreai-realesrgan-swift`, GAP-PROGRAM V13/V13-P/V13-E). Verified on macOS 27.0 / Xcode 27 beta.

## Swift runtime API (the parts the docs don't state)

- `AIModel(contentsOf:options:)` + `model.loadFunction(named: "main")` → the function type is
  **`InferenceFunction`** (from `CoreAIRuntime`, re-exported by `CoreAI`). It is NOT
  `AIModel.Function` — that name does not exist and the error message is unhelpful.
- Compute preference: `SpecializationOptions(preferredComputeUnitKind: .neuralEngine/.gpu/.cpu)`.
- I/O: `NDArray(shape:scalarType:)`, `mutableView(as: Float16.self).withUnsafeMutablePointer`,
  outputs come back as a dictionary — probe `"output"` then `"out"`.
- SPM: `platforms: [.macOS("27.0")]`, `linkerSettings: [.linkedFramework("CoreAI")]`,
  swift-tools **6.2**. An `.aimodel` is a DIRECTORY — vendor it with `.copy(...)` resources.
- First load per (model × machine) pays **~8 s E5RT specialization, OS-cached after** (then ~0.2 s).
  Pay it at your package's `load()`/prepare seam, never inside the first user-visible inference.

## Export (coreai-torch) recipe that works

`uv run` script with `coreai-core==1.0.0b2`, `coreai-torch==0.4.1`:
`torch.export.export` → `run_decompositions(get_decomp_table())` → `TorchConverter()
.add_exported_program(ep, input_names, output_names).to_coreai()` → `program.optimize()` →
`save_asset(path)`. Static shapes only for ANE residency.

## Measured design facts (Real-ESRGAN SRVGG, M5 Max)

- fp16-on-ANE parity vs fp32 torch: 58–69 dB min across three variants — and variants MUST each be
  measured (sibling ports have had dtype verdicts invert between variants).
- ANE wall-clock ties well-tuned MLX-GPU at t128 while drawing **≈4.5–4.9× less energy** and not
  throttling. Static CoreAI executables also beat MLX *dynamic* GPU 2.2–2.4× at parity.
- Memory is a different SHAPE, not just size: activations stay on-die, so process footprint is the
  host-side accumulation buffers only (19 MB resident / 0.86 GB peak for 1080p→×4 vs the MLX
  sibling's 21.24 GB whole-frame). Tile geometry is a build-time property of the asset (one
  executable per static shape) — the inversion of an MLX port's runtime-injectable geometry;
  document the inversion in both packages so nobody "fixes" either toward the other.

## Engine integration without MLX

An MLXEngine package over a CoreAI core needs **MLXToolKit only** (the engine's dependency-free
contract layer) — zero MLX linked. A macOS-26 host package (e.g. ForgeCore) CANNOT depend on a
27-floored CoreAI package (SPM refuses); the working pattern is an injection seam
(`ForgeCore.ExternalRegistration`): the app's own deployment target decides, injects a registration
closure, and the backend registers beside the MLX sibling under its own PackageID.

## Publishing

HF org `coreai-community`: naming `<Model>-CoreAI`; membership approval AND adding the org to your
fine-grained token are two separate steps that both 403 identically when missing. Ship the model
card with parity numbers + the export script; fresh-download-verify before announcing.

## Second port: Moebius UNet (2026-08-01, moebius-m0/coreai/) — the diffusion-scale lessons

The first *complex* export (226M latent-diffusion UNet: depthwise convs + BatchNorm + linear
λ-attention + einsums, vs SRVGG's plain dense convs). Everything below was paid for, not read.

### Placement is a claim you must prove, not a flag you pass

- **A preference is not a placement, and the runtime never volunteers what it did.** In Python,
  `AIModel.load(path)` with no `specialization_options` silently uses default delegate placement.
  A benchmark script that parses `--compute` but forgets to pass the options object produces
  identical-looking output with a false label — this happened, and a published "ANE" table was
  actually the GPU. Build the options, pass them, and echo `options.preferred_compute_unit_kind`
  back off the runtime object, never the argparse string.
- **Always run the CPU lane as a control.** If CPU/GPU/ANE all return the same latency, the flag is
  inert (here: CPU 601 ms vs GPU 205 ms once fixed; all ~205 ms before). Three agreeing lanes is
  the tell.
- Explicitly requesting `neuralEngine` on an ANE-ineligible graph has TWO failure modes, and only
  one is loud: (a) hard raise (`SystemError` out of `load_function`) — seen when validation rejects
  rank-6 reshapes; (b) `_ANECompiler: ANECCompile() FAILED` printed to stderr followed by a SILENT
  fallback that returns GPU numbers with your "ANE" label on them — seen when validation passes but
  codegen fails (`CompilationFailure`). Mode (b) means a completed run is NOT evidence of ANE
  execution. The reliable detector is stderr `ANECCompile` + the CPU/GPU-lane latency controls:
  if your "ANE" latency equals your GPU latency, it IS the GPU. Never benchmark on default
  placement, and never trust a compute-unit request that you haven't cross-checked against a
  control lane.
- The stderr `ane_validation_message` warnings are the best residency evidence available — they
  name the failing op, rank, and ORIGINAL Python source line (via debug locations). But they're
  embedded in enormous MLIR `warning: loc(...)` blobs; filter with `grep -v "^warning: loc"` and
  grep for `ane_validation_message` separately.
- `SpecializationOptions.is_supported()` gates on `USE_OS_COREAI` semantics (wheel installs on
  macOS 27+ default to the OS framework). Exit hard when False; silent fallback = mislabeled data.

### ANE eligibility: max tensor rank is 5

- torch.export decomposes an einsum with six distinct indices (e.g.
  `'n m k u, b u v m -> b n k v'`) through **rank-6 reshapes**, and MPS-ANEC hard-rejects rank > 5.
  The GPU delegate doesn't care, so this only bites when you explicitly request ANE.
- Fix: fold multi-index einsums to batched matmul (contract axes merged, batch axes merged) —
  algebraically exact, verify `max abs diff == 0.0` in fp64 before trusting it. Patch at the
  MODULE-GLOBAL binding of the einsum helper (a `from x import _einsum` means patching the source
  module after import is a no-op — rebind the consumer module's global).
- Gate the rewrite numerically inside the export script itself: fp32 eager forward pre- vs
  post-patch, hard-exit on divergence. Cheap, and catches transcription slips at the only moment
  they're catchable.

### Precision at export

- **Mixed precision does not lower.** Pinning BatchNorm running stats to fp32 inside an fp16 graph
  (standard MLX-converter hygiene) fails legalization outright: "unresolved materialization from
  tensor<*xf32> to tensor<...xf16>". Uniform dtype per export; measure whether fp16 stats are safe
  (count running_var values that round to zero) instead of assuming the fp32 pin transfers.
- fp32 export is cheap insurance: it isolates "graph wrong" from "quantization loss" in one number
  (here fp32 = 138 dB proved the graph; fp16 = 41 dB localized the problem to precision).

### Runtime/API sharp edges (Python, coreai-core 1.0.0b2)

- `Profiler(...)` requires ALL THREE callbacks; passing only `on_log_event` leaves the interval
  hooks None, the native side calls them anyway, and it surfaces as an opaque
  `SystemError: ... returned a result with an exception set` from `load_function`.
- First-load E5RT specialization scales with graph size: ~8 s for SRVGG, **254 s** for this 226M
  UNet (fp16). OS-cached after (1.1 s load + 0.3 s first call). Budget it at the prepare seam and
  warn the user the first time.
- **The cache also caches FAILURE-then-fallback.** A first ANE attempt that prints
  `ANECCompile() FAILED` and falls back to GPU gets its (GPU) specialization cached under that
  asset — every subsequent load is fast, error-free, and still on the GPU. The diagnostic evidence
  (validation warnings, compiler errors) exists ONLY on the first cold-cache attempt; capture raw
  stderr on that run or lose it. To retry diagnosis, delete the asset's entries under
  `~/Library/Caches/coreai-cache/<os-build>/<bundle-id>/<hash>` (identify by mtime) — a re-run with
  a warm cache tells you nothing.
- An `.aimodel` is a directory and `main.mlirb` is roughly checkpoint-sized (431 MB fp16 / 862 MB
  fp32) — gitignore the exports dir BEFORE the first `git add -A`, not after the push is rejected.
- A SIGABRT during specialization lands in MPSGraph/Metal inside YOUR process (`__assert_rtn` →
  `MPSGraphExecutable runMLIRModulePassesAndCommonInit`), with the assertion text only on stderr —
  capture stderr or lose it; the .ips has no `asi` field. A crash in a backend you don't believe
  you're using means your placement instrumentation is lying.

### Carried over from SRVGG and reconfirmed

- ⚠️ **`patch_nearest_upsample` (repeat_interleave for nearest-×2) is NOT universal hygiene —
  A/B it per port.** The upstream coreai-models comment says MPSGraph rejects nearest-mode
  `coreai.interpolate` and routes it to BNNS/CPU, and that reads like a blanket rule. MEASURED on
  Real-ESRGAN SRVGG (2026-08-01, same checkpoint, 128² tile, M5 Max, median of 20 post-warmup):
  patching it is **neutral on the GPU lane (2.17 vs 2.19 ms) and 16% SLOWER on the ANE lane
  (6.74 vs 5.79 ms)**. The patched export genuinely drops the op (`strings main.mlirb` finds no
  interpolate), so the rewrite works — it just doesn't pay for that architecture. Plausibly
  `repeat_interleave` materialises the scaled tensor explicitly where the native op is handled
  better. **Honesty note: the patch's value for the Moebius UNet is UNMEASURED** — it was applied
  there from the upstream comment and never compared against an unpatched export, so "load-bearing"
  was an inherited claim, not a finding. Export both ways and probe each lane before believing it.
- `save_asset` wants a `Path`; an fp32 asset rejects fp16 inputs (good: no silent cast) — derive
  input dtype from the asset name/metadata, not from habit.

### The full-day arc, distilled (Moebius, continued)

- **Bisection method that worked**: per-block export + subprocess ANE load with stderr-parsed
  verdicts → stage-split inside the failing module → seam-split between passing modules → minimal
  repro. Each level took ~10 min; guessing from error text alone would have found none of it (the
  errors were `<private>`-redacted in the unified log and empty in the exception).
- **Non-monotonic stage verdicts mean your scaffolding is in the graph.** A cumulative stage that
  "fails" while its superset passes is impossible — look for harness artifacts. Keep-alive hacks
  (`+ x.sum()*0`) fail the ANE compile on their own.
- **Compositional compiler bugs exist**: two 64²-level Transformer2DModels in one graph break
  ANECCompiler's input-channel-split pass ("failed create split by input channel, graph is
  changed"; macOS 27.0, CoreAICompiler 3600.79.1) even though every component and every
  single-transformer composition compiles. The split pass only engages at large spatial extents —
  the same pair at 16²/32² is fine. No graph-level dodge worked: linearized projections, resnet
  separation, and forced region-breakers (rank-6 reshape roundtrip, fp32 bounce ×1.0000001) were
  all absorbed by the optimizer/segmenter.
- **ANE rewrites can be GPU wins**: the rank-5-free λ formulation (fold Conv3d depth into conv
  batch; einsums → broadcast/batched matmul) made the MPSGraph GPU delegate 4.1× faster
  (205 → 50 ms on a 226M UNet forward). Placement-motivated fixes are worth keeping even when the
  placement itself stays blocked.
- **After ANY graph rewrite, re-measure EVERY lane before attributing speedups to placement.**
  A 4× "ANE" speedup was really the new graph's GPU lane; the elimination argument compared
  against the OLD graph's GPU number. The macmon GPU-idle signature (gpu_freq pinned at the 338 MHz
  idle clock during sustained inference) is the residency oracle; latency deltas are not.

### Precision + upstream (Moebius, closing entries)

- **fp16 BN fix that closed a 27 dB gap at zero cost**: don't evaluate `(x−mean)·rsqrt(var+ε)`
  in fp16 when running_var has subnormal channels — replace each BN with per-channel
  scale/shift computed at fp64 THEN cast (`scale = γ/√(var+ε)`, `shift = β − mean·scale`; the
  composites are fp16-representable even where var is not). Moebius: 41.4 → 68.3 dB, landing at
  the same fp16 floor as the MLX port (rel 9.3e-04).
- **isinstance(nn.BatchNorm2d) also matches timm `BatchNormAct2d`**, whose forward appends
  drop + activation — a swap that keeps only the affine silently deletes the ReLU (rel ~1.0).
  Always run a per-module differential harness over every instance you replace; the export's
  fp32 pre/post gate is what catches it at the only moment it's catchable.
- **Value-dependent compiler bugs exist.** The two-transformer ANECCompile failure reproduces
  with trained weights and vanishes with same-architecture random weights; four [320] LayerNorm
  affine vectors alone flip it, and same-range synthetic values do NOT. Consequences:
  (a) random-weight smoke tests can NOT clear a graph for ANE — test with real weights;
  (b) when building a "random control", construct it by deepcopy-then-randomize — building from
  config can silently produce a different architecture (diffusers attention-class routing did
  exactly that and voided a whole verdict table).
- Upstream: filed as apple/coreai-models#138 with a public-weights validated repro
  (`moebius-m0/coreai/repro_upstream.py`). Re-test per macOS update before assuming the ANE
  door is still closed.
