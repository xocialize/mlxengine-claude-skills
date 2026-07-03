# Stage 2 — Integration Lessons (living gotchas checklist)

Append a new bullet every time an integration teaches one. This is the fast, reusable trap list;
project-specific play-by-play lives in `~/Development/MLXEngine/EngineeringDocs/CLAUDE.md` and
`~/Development/MLXEngine/EngineeringDocs/MLXEngineDocs/first-integration-notes.md`.

## Packaging

- [ ] **`-swift` suffix on the package/repo name ONLY** (mirrors `mlx-engine-swift`); the
  module/product stays clean PascalCase (`MLXQwenLLM`, not `mlx-qwen-llm-swift`). The Python port
  keeps the bare name.
- [ ] **Target names must be GLOBALLY UNIQUE across the package graph — prefix every smoke/CLI/util
  target with the model name.** SwiftPM requires unique *target* names across the whole dependency
  graph, so the moment an app links 2+ of our wrappers a generic target name collides:
  `multiple targets named 'Smoke'` (real: `mlx-birefnet-swift` + `mlx-ddcolor-swift` + `mlx-lama-swift`
  each declared an executable **target** named `Smoke` — products `birefnet-smoke`/`ddcolor-smoke`/
  `lama-smoke` were unique, but **unique products are NOT enough; the underlying target name is what
  collides**). Name targets `BiRefNetSmoke`/`DDColorSmoke`/`LaMaSmoke`, `…Convert`, `…PackageSmoke`,
  etc. — never bare `Smoke`/`Convert`/`Tools`. The `…-smoke` *product* name can stay short. Audit:
  `grep -hE 'name: "(Smoke|Convert|Tools|PackageSmoke)"' */Package.swift` should be empty. (Very likely
  to recur — "Smoke" is the default name we reach for every port.)
- [ ] Package depends on **`MLXToolKit`**; never fold a package into `mlx-engine-swift`.
- [ ] **UI boundary (engine vs app):** `MLXEngineUI` ships only *reusable, engine-management* UI +
  the shared design tokens (the model-storage settings panel, `MarqueeColor`/`MarqueeFont`).
  *App-specific product UI* (e.g. a chat surface) lives in the **consuming app**, importing only the
  tokens for visual consistency and talking to `MLXServeEngine`. Never put product UI in
  `MLXEngineUI`, and never put UI in `MLXToolKit` (contracts) or `MLXServeCore` (coordinator).
- [ ] Conformer is `@InferenceActor final class … : ModelPackage` (class-level isolation makes it
  `Sendable` + satisfies the actor-isolated `load`/`unload`/`run`); `nonisolated` `init`, `manifest`,
  and `registration` (`PackageRegistration.of(Self.self)`).
- [ ] Model checkpoint as **size × quant** → compute the repo id; don't hardcode one string.

## mlx-swift-lm (3.x runtime)

- [ ] Products: `MLXLLM`, `MLXVLM`, `MLXLMCommon`, `MLXHuggingFace`. **Link both MLXLLM AND MLXVLM** —
  a model may live in either; the loader's `ModelFactoryRegistry` finds factories via
  `NSClassFromString` trampolines, so LINKING the product is enough (no manual registration), and it
  auto-dispatches by `config.json`. **Caveat:** that link-time auto-registration is exactly what makes
  the MLXVLM-shadowing pitfall (next bullet) fire process-wide.
- [ ] **PITFALL — linking MLXVLM anywhere in an app shadows text architectures for ALL packages in the
  process.** `ModelFactoryRegistry` is process-global and probes `MLXVLM.TrampolineModelFactory`
  BEFORE `MLXLLM` (`Libraries/MLXLMCommon/ModelFactory.swift:480` via `NSClassFromString`).
  Architectures registered in BOTH factories — e.g. `"gemma3"`: `VLMModelFactory.swift:91` (multimodal
  Gemma3) vs `LLMModelFactory.swift:31` (`Gemma3TextModel`) — resolve to the VLM variant once MLXVLM
  is linked. This happens at LINK time, even when the MLXVLM-linking package is never used. Real
  incident (2026-07-01, BRIDGE-LTX-003 in `~/Development/mlxengine-video-ltx/AGENT_BRIDGE.md`):
  registering `mlx-qwen-llm-swift` (which links MLXVLM to serve VL checkpoints) into the LTX app made
  LTX's own `GemmaEncoder` auto-dispatch load multimodal Gemma3 and fatal-error on its
  `Gemma3TextModel` cast. Mitigations: (a) packages that auto-dispatch via `#huggingFaceLoadModel…`
  should downcast defensively (throw a typed error), NOT `fatalError`; (b) app integrators must audit
  whether any linked package pulls MLXVLM before combining it with packages that load text models by
  architecture name; (c) utility kits (like prompt-enhance-kit) should offer host-model closures so
  they never oblige linking a model package — see `prompt-enhance-kit-swift` v0.3.0's
  `generate(system:user:)` overload and its README warning. Consumer-side counterpart lives in the
  `mlxengine-implementation` skill (`references/routing.md`).
- [ ] **3.x decoupled the HF stack** — mlx-swift-lm depends only on mlx-swift + swift-syntax. The
  download macro/tokenizer need the CONSUMER to add `swift-huggingface` + `swift-transformers`
  (products `HuggingFace`, `Tokenizers`). Known-good pins: mlx-swift-lm 3.31.x, swift-huggingface
  0.9.0, swift-transformers 1.3.x.
- [ ] **Load (default cache):**
  `#huggingFaceLoadModelContainer(configuration: ModelConfiguration(id:revision:))` (needs
  `import MLXHuggingFace, HuggingFace, Tokenizers` where it expands).
- [ ] **Load (into a chosen folder):**
  `loadModelContainer(from: #hubDownloader(HubClient(cache: HubCache(cacheDirectory: root))), using: #huggingFaceTokenizerLoader(), id:, revision:)`.
- [ ] **Generate:** `ChatSession(container, instructions:, generateParameters:).respond(to:)`.
  `GenerateParameters` uses `temperature: Float`, `topP: Float`, `maxTokens: Int?`.
- [ ] **Multi-turn:** the engine is stateless per `run`, so thread the whole transcript yourself —
  seed prior turns via the history initializer
  `ChatSession(container, instructions:, history: [Chat.Message], generateParameters:)`, then
  `respond(to:)` the latest user turn. Build `Chat.Message` with `.user(_:)` / `.assistant(_:)` /
  `.system(_:)`. System turns → `instructions`.
- [ ] **`ChatSession` is NOT `Sendable`** (not thread-safe). Create + consume it inside a
  `nonisolated` helper that takes only `Sendable` inputs (your own `ChatMessage`, the
  `ModelContainer`, `GenerateParameters`) and maps to `Chat.Message` internally — never let the
  session cross the `@InferenceActor` boundary, or Swift 6 errors *"sending 'session' risks causing
  data races"*. (Adding the `history:` arg is what tends to trip the region-isolation check.)
- [ ] **TTS / audio: Kokoro-82M via `Blaizzy/mlx-audio-swift`** (MIT; products `MLXAudioTTS` +
  `MLXAudioCore`). `TTS.loadModel(modelRepo:)` → `model.generate(text:voice:…)` → MLXArray →
  `.asArray(Float.self)`; encode a 16-bit PCM WAV **in memory** and return canonical `Audio(.wav)`
  (don't return a temp file). Kokoro uses **named voices** (`af_heart`, …);
  `VoiceSelector.auto`/`.referenceAudio` → fall back to a default.
- [ ] **Un-Sendable-audited MLX libs with `@concurrent` methods** (mlx-audio-swift's `generate`) hit a
  hard Swift-6 "sending" error that `@preconcurrency` / `nonisolated(unsafe)` do NOT clear when the
  resource is long-lived actor state. Escape hatch: build **that interop target** in Swift language
  mode v5 — `swiftSettings: [.swiftLanguageMode(.v5)]`. `@InferenceActor` still holds; the engine
  serializes lifecycle so there's no real race. (If the resource is Sendable — like `ModelContainer`
  — prefer the nonisolated-helper pattern instead and stay in v6.)

## Running MLX in the app (Metal)

- [ ] MLX GPU kernels trip **Metal API Validation** in Debug (e.g. `MTLDebugComputeCommandEncoder
  setBytes…` → `__assert_rtn` inside `mlx::core::Gather::eval_gpu`) — a SIGABRT on otherwise-fine
  code (zero-length/strided `setBytes`). Recurring — surfaced on **Kokoro TTS** and again on **BiRefNet
  matting** (`Gather::eval_gpu` at indexing.cpp:200, the Swin/DCNv2 indexing); whichever package first
  exercises that `Gather` shape trips it (NAFNet/Real-ESRGAN never did). Fix: **disable Metal API
  Validation** for the Run scheme (Edit Scheme → Run → Diagnostics → uncheck *Metal API Validation*).
  Release builds and the companion app run with it off. Not a code bug.
- [ ] **The Metal-watchdog family (`kIOGPUCommandBufferCallbackErrorTimeout`, ≈10 s ceiling): a
  Metal command buffer must never wait on slow non-GPU work.** Four disguises, one root cause
  (Bernini 2026-06-12; full doctrine in `swift-port-parity.md`): (1) **weight loads ride the CPU
  stream** — lazy `Load` ops bind to the stream current at array creation; GPU-stream loads over
  multi-GB disk reads hold one buffer past the watchdog (chunked GPU-side eval is NOT enough on a
  slow disk) — wrap `loadArrays`+eval in `Device.withDefaultDevice(.cpu)`; (2) **quantized
  forwards run wholly on the GPU stream** — quantized matmuls route to Metal even under a CPU
  pin, so a CPU-pinned quantized graph is one Metal buffer fenced on CPU ops at every block;
  (3) **never eagerly eval giant constant fills** (zero-filled params pre-quantize: correct to
  create, fatal to eval — 57 GB/expert fp32, observed as a 161 GB swap-storm whose paging also
  reads as the watchdog); (4) **ARC-scope big models** so a phase's 28 GB frees before the next
  load. Symptom triage: it can fire during *loading* (1), *quantized inference* (2), or under
  *memory pressure* (3/4) — bisect with staged stderr markers (stdout is block-buffered when
  redirected) before theorizing.
- [ ] **Footprint realism for big pipelines:** the resident set is more than the DiT — e.g. an
  fp32-upcast umT5-XXL is ~22 GB that stays resident for the whole generation in a naive
  pipeline. Measure the manifest's `residentBytes` from the real peak (Bernini: bf16 ≈ 91 GB,
  int4 ≈ 52 GB peak at 832×480), and treat encoder residency as a wrap-level eviction
  opportunity (encode → release → govern), not a fixed cost.

## Live validation in the app (the harness — where real bugs surface)

- [ ] **Offline conformance never runs a kernel; live is the FIRST real forward pass.** So live
  validation is where silent-failure-class bugs appear (see below). Each offline-green package gets an
  entry in `APP-VALIDATION.md`; tick it after an in-app run.
- [ ] **Build ONE reusable validation harness**, not per-model glue. The app's `ValidationView` drives
  any package generically: model picker → input picker → `useModelStore` → `evict` → `register` →
  `prepare` (time it) → `run` (time it) → decode, capturing **load/run seconds**, **peak process
  memory** (`task_vm_info.phys_footprint`, sampled across the run), and the **engine-charged
  footprint** (`engine.memory.residents[capability]`). The app uses a file-system-synchronized group,
  so a new `.swift` file is picked up with no pbxproj edit.
- [ ] **Quantify the result — don't trust eyes/ears.** Add per-capability metering so "did it work" is
  a number: separation/polish → **peak & RMS dBFS per stem** (a silent stem reads −∞/−90, the tell
  that caught Mel-RoFormer); codec → `numCodebooks`, `T ≈ ⌈seconds·frameRate⌉`, index range; IQA →
  score discrimination (clean-high vs degraded-low); restore/upscale → output dims vs input (×scale +
  `appliedScale`). This is what turns "looks empty" into a localizable defect.
- [ ] **Same-capability swap / selection:** with the multi-package registry, re-registering still
  works as the swap (the new registration becomes the capability's default), and the harness's
  `evict` → `register` every run keeps variant/config changes honest. For co-resident backers,
  validate the SELECTION path too: `prepare(capability, package:)` both ids, confirm
  `residentPackages` charges both, and `run(request, package:)` routes to the named one.

## Silent-failure class (what live validation catches that offline + norms do NOT)

- [ ] **CGBitmapContext + `CGContext.draw(CGImage:)` is ALREADY top-row-first — do NOT add the
  AppKit-style vertical flip when building an image-input tensor.** The `translateBy(0,H)`/`scaleBy(1,-1)`
  flip (needed for NSView/UIKit drawing conventions) INVERTS a CGImage drawn into a bitmap context. LTX
  i2v shipped with this exact bug: every generation opened with an upside-down first frame that "fell"
  upright as the frame-0 conditioning released — invisible to parity gates (they feed tensors, not
  images), caught only by watching output video. Verify orientation in 10 s with a probe: draw a
  red-top/blue-bottom CGImage into the context and assert buffer row0 reads RED (no flip) — row0=BLUE
  means you flipped it. Same class as the Wan temporal-reversal (W5): orientation bugs live at the
  app/codec/image-input boundary, never in the model.
- [ ] **Validate the PUBLISHED artifact end-to-end (download → run), not just the local build.** "Build
  green + local-weights parity passes" is NOT the finish line — the *published* dtype-rounded weights are
  a separate artifact and can be silently broken. LaMa shipped as `-fp16` built fine and the local fp32
  parity was 3e-5, but the published fp16 weights produced **garbage** (its FFC bottleneck activations
  ~1e3 collapse under fp16 rounding; bf16 fixed it — see `mlx-porting` parity-testing "Choose the publish
  dtype"). It was caught ONLY by downloading `mlx-community/<repo>` and running an erase. **Make the last
  step of any publish a fresh `hf_hub_download` → run on a real input → eyeball the output.** Re-run the
  parity gate at the *published* dtype, not just fp32.
- [ ] **Compositing hides a broken model in the untouched region.** Inpaint/matte/colorize outputs paste
  the model result back under a mask (`out·mask + src·(1-mask)`); if the model emits garbage, only the
  *masked* pixels show it — the rest is the original image, so a thumbnail looks "basically fine." Judge
  the model on the region it actually generated (the hole / the matte / the recolored pixels), not the
  composited whole. (LaMa fp16: garbage was visible only inside the erase box.)
- [ ] **Norm/energy probes can look perfectly healthy while the model is functionally wrong.** A RoPE
  **convention mismatch** — "halved" pairing `(x[:half], x[half:])` vs `rotary_embedding_torch`'s
  **interleaved adjacent pairs** `(x[2i], x[2i+1])` — is a *valid* rotation that doesn't match
  training, so attention decorrelates and the output collapses to ~0, yet every per-stage energy probe
  stays healthy (rotations preserve norms). Lesson: confirm a port with **elementwise parity vs the
  Python reference** on a fixture (cosine ≈ 0.999), not norm/energy sanity. (Mel-RoFormer silent-vocals
  bug; fixed in core v0.1.1.)
- [ ] **`module.update(parameters:, verify: .noUnusedKeys)` tolerates UNSET module params.** It only
  rejects *extra* file keys — a renamed/missing key silently leaves that param at its init value
  (zeros/random) and the build runs clean. The shipped pattern (every image-era port) is a
  **two-way strict `verifyAndLoad`**: `moduleKeys − fileKeys` (missing) AND `fileKeys − moduleKeys`
  (unconsumed) must BOTH be empty before `update` — a partial load emits garbage with no other
  symptom. Copy it from `lens-mlx-swift` / `qwen-image-edit-swift` `Weights.swift`.
- [ ] **Evolution of the two-way strict load: a pinned, GENERATED key contract** (Bernini
  `BerniniWeightKeys` + `loadVerifiedSafetensors`). Generate the expected key set from the
  architecture (blocks × per-block paths + globals), verify it against the actual safetensors
  *headers* of every published variant in an offline test (pure-Foundation header read — no MLX),
  then have the loader enforce set-equality after dropping declared `toleratedExtras` (e.g. a
  stray serialized rope `freqs` in a quantized variant — and SUBTRACT the tolerated key from the
  expected set, or the check contradicts itself). Three structural gates fall out free, all
  weight-free and Metal-free: header↔contract, module-paths↔contract
  (`parameters().flattened()` on a lazy init), and loader enforcement at runtime. Keep
  non-parameter buffers in a plain holder class so Module reflection can't see them.

## Audio wrappers (separation / speechEmotion / audioCodec / audioPolish)

- [ ] **Reuse the core's `AudioIO`**: write the contract `Audio` bytes to a temp file and let the core
  load+resample via AVFoundation (it expects a URL); don't re-implement resampling. Cores target their
  own rate (RoFormer/Demucs 44.1 kHz stereo; emotion2vec 16 kHz mono; Mimi 24 kHz mono) — the contract
  `Audio` need not match.
- [ ] **AVFoundation WAV finalize / `avfaudio error -50`:** `AVAudioFile(forWriting:)` only writes the
  RIFF/`data` chunk sizes when the writer is **deallocated**. Reading the bytes while the writer is
  still alive yields a truncated file that `AVAudioFile(forReading:)` rejects with error −50. Scope the
  writer (`do { let out = …; try out.write(from:) }`) so it deallocs **before** you read the file.
- [ ] Canonical outputs by capability: separation → `[Stem: Audio]` (.wav); speechEmotion →
  label+confidence+**softmax scores Σ≈1.0**; audioCodec → `[numCodebooks][T]` int grid
  (T ≈ ⌈seconds·frameRate⌉, indices in range). **audioPolish is classical (no MLX/weights)** → empty
  footprints + empty backends; takes a per-request `mode` (`.broadcast` −23 LUFS) and reports
  input/output LUFS — the non-MLX capability seam.

## Visual wrappers (imageQualityScore / imageRestore / imageUpscale)

- [ ] Canonical `Image` (.png/.jpeg). Decode via `CGImageSource` → BGRA `CVPixelBuffer` → NHWC, run,
  re-encode PNG; carry true pixel `width`/`height`. Validate the geometry contract: restore preserves
  dims; upscale output == `appliedScale × input` (assert both, not just the flag); IQA score must
  **discriminate** (clean ≫ degraded), not merely return a float.
- [ ] **`Image` name clash:** `MLXToolKit.Image` (artifact) vs `SwiftUI.Image` (view). In app/UI code
  `typealias ImageArtifact = MLXToolKit.Image` (internal, not private, if an internal enum references
  it) and use `SwiftUI.Image(nsImage:)` explicitly for previews.

## Build environment (this machine)

- [ ] CommandLineTools `swift` is broken (missing `BuildServerProtocol.framework`). Use the Xcode
  toolchain:
  `env DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift|xcodebuild …`.
- [ ] **After changing package deps, modules won't resolve until you re-resolve:**
  `xcrun xcodebuild -resolvePackageDependencies -workspace "MLXEngine.xcworkspace" -scheme <scheme>`.
- [ ] **Adding a package product to an app target has no tool and pbxproj edits are blocked** — the
  user links it by hand in Xcode (target → General → Frameworks, Libraries → +). Ask, then continue.
- [ ] To screenshot a macOS app window, print `window.windowNumber` then `screencapture -l<n>` —
  full-screen capture is occluded by the IDE.
- [ ] **Programmatic AppKit app (no storyboard `MainMenu`) → paste/copy/cut/select-all don't work in
  text fields.** macOS routes ⌘X/⌘C/⌘V/⌘A through the **Edit menu's** key equivalents; with no main
  menu they never reach the focused field. Install `NSApp.mainMenu` with an App + Edit menu (items
  `cut:`/`copy:`/`paste:`/`selectAll:`, target nil → responder chain) at launch.

## Sandbox / storage

- [ ] App needs: App Sandbox + **outgoing network** (download), **user-selected read-write**, and an
  **app-scope bookmark** to persist the chosen folder; disabling library validation helps MLX dylibs.
- [ ] The chosen folder's **security scope must be active** during download + marker write. Share ONE
  `ModelStorageModel` (owns the bookmark; exposes `resolvedModelsDirectory` + `refresh()`).
- [ ] HF cache layout = `<root>/models--<ns>--<repo>/…`; the engine writes `mlx-package.json` under
  `<root>/<ns>/<name>/`. `ModelStorageModel` counts markers (Models Installed) and sums files (Disk
  Used).
- [ ] **Symlinked weight components need the symlink TARGET's parent granted, not the link's.** A
  checkpoint dir that symlinks shared components (e.g. VACE reusing Bernini's `vae.safetensors` /
  `t5_encoder.safetensors`) resolves through the link at read time → the security scope must cover the
  *target* path. Simplest fix: grant a common ancestor of both (e.g. the whole `/Volumes/DEV_ARCHIVE`)
  rather than the per-model dir, so links into sibling model dirs stay in-scope.
- [ ] **An absolute out-of-store `snapshotPath` must win over the engine-stamped `modelsRootDirectory`.**
  The engine stamps `modelsRootDirectory` (the store root) onto **every** `ModelStorable` config at register.
  If your config also exposes an explicit **absolute** local snapshot path (staged-on-disk weights — NC/
  gated weights, a hand-placed `dist/`), `load()`'s path resolution must honor the absolute path over the
  stamp: a naive `modelsRootDirectory.appending(snapshotPath)` prepends the store root to an already-absolute
  path → broken resolution. Safe shapes: resolve `snapshotPath` directly with `URL(fileURLWithPath:)`
  (ignoring the stamp when it's absolute — Qwen-Image-Edit, ERNIE), or branch on `snapshotPath.hasPrefix("/")`
  and prefer it (Anima's v0.1.1 fix). The bug only surfaces once an app sets a store — verify it when wiring
  any absolute-snapshot package into a store-backed app.

## Contract

- [ ] License gate is two-layer (weight C7 + port-code C8):
  `LicensePolicy.permissiveOnly.evaluate(manifest.license)`; `.isAdmitted` before constructing.
  Qwen/Apache-2.0 passes. Allowlist has grown: `LicenseRef-FunASR-Model` (emotion2vec), `CC-BY-4.0`
  (Mimi).
- [ ] `run(_:)` dispatches on `request.capability`, downcasts to the canonical request, honors
  `Task.checkCancellation()`. Throw `PackageError.unsupportedCapability` / `.notLoaded` appropriately.
- [ ] **`Capability` AND `Quant` are additive enums** — the engine grows cases (recent:
  `imageQualityScore`, `imageRestore`, `imageUpscale`, `videoUpscale`, `frameInterpolate`; `Quant.fp32`
  for Mimi). A new `Quant` case makes an **older package's exhaustive `switch` non-exhaustive** → build
  break (bit Qwen's `mlxCommunitySuffix`/`bytesPerWeight` **twice** — once for `.fp32`, again for
  `.int5`/`.int6` at contract 1.1.0; contract 1.2.0 added `Capability.imageEdit` + the IEdit types).
  Handle the new case or `@unknown default`. Same discipline consumers owe `Capability` (C12). When
  you bump the contract with a new `Quant`, grep every package's `switch self` over `Quant` before
  publishing.
- [ ] **Engine contract needs macOS 26 → the port's manifest needs `swift-tools-version: 6.2`.**
  `.macOS(.v26)` is unavailable in 6.0/6.1 manifests; the wrapper target that imports `MLXToolKit`
  forces the whole `-swift` package to tools 6.2 + platform `.v26`. (Hit on both Lens and
  Qwen-Image-Edit wrappers.)

## Video I/O & sequence outputs (EdgeTAM `trackObject`, contract 1.11.0)

- [ ] **`MLXToolKit` has ZERO AVFoundation — the `Video` artifact is just serialized mp4/mov bytes.** The
  established video convention (`videoUpscale`/`frameInterpolate`/`opticalFlow`) is `Video → Video`; the
  **runtime package** owns decode/encode, not the contract and not the app shell. A video ModelPackage
  depends on a decode lib (`xocialize/frame-stream-native`, the FFmpeg-free AVAssetReader path RIFE/SeedVR2
  use), writes `Video.data` to a temp file, decodes, runs, re-encodes. **C13 (inversion of control) is about
  who CONSTRUCTS the package, NOT "no AVFoundation in the package"** — don't mistake the pure-value-types
  *contract* layer for a no-AVFoundation rule on the *runtime* layer. (Got this boundary wrong first — assumed
  decode had to live in the Forge shell; corrected by reading `videoUpscale`'s actual `Video↔Video` shape.)
- [ ] **Streaming frame-transform API ≠ stateful-tracker API.** `FrameStreamNative.run(input:output:transform:)`
  is an N:M frame-to-frame pipeline (fits upscale/interpolate). A STATEFUL sequence model — object tracker /
  masklet propagation that prompts on a chosen frame and carries cross-frame memory — doesn't fit it; it needs
  a **decode-to-frames seam** (`decode(input:) -> [CGImage]`). Check which shape the model is before assuming
  the transform API fits; adding the seam to the shared decode lib is the reusable move (every future tracker
  needs it), not rolling AVAssetReader inside the package.
- [ ] **A sequence-of-masks output is NOT a `Video`.** `trackObject` returns `[Matte]` (one lossless PNG per
  frame) + a new `CanonicalOutput.matteSequence`, deliberately NOT the `Video→Video` convention: **re-encoding
  hard-edged binary masks through H.264/HEVC rings the edges.** Lossy video codecs are for *pixel content*
  (upscaled frames), never for masks/alpha. Add a lossless per-frame sequence artifact kind rather than forcing
  a masklet into a mask-video — and surface the per-frame confidence as a parallel `scores: [Float]`, not baked
  into the artifact.
- [ ] **Measure a per-frame-stateful footprint across clip lengths, not once.** Peak scales with frames if the
  package pre-stacks input + retains all outputs (EdgeTAM measured 1.07 GB @ 5f → 1.79 GB @ 30f ≈ 0.9 GB fixed
  + 30 MB/frame). Declare the envelope from the measured *slope*, document the clip ceiling it implies, and note
  streaming (don't pre-stack, encode each frame's output as it lands, `mx.clear_cache()` per step) as the fix.
  `MLX.GPU.peakMemory` in a CLI surface-smoke (encode N real frames → run the package surface → IoU vs goldens)
  gives the number — and exercises the whole `Video bytes → decode → run → artifact` path the harness will hit.

## Coordinator (MLXServeCore)

- [ ] Route the consuming app through **`MLXServeEngine`** (actor), not the package directly:
  `register(registration, configuration)` runs the license gate at registration; `prepare(capability)`
  / `run(request)` lazily **construct + load + route** by capability (engine constructs, never the
  package — C13); `evict(capability)` unloads. That's the inversion of control the architecture is for.
- [ ] The engine + registry is unit-testable **offline** with a tiny mock `ModelPackage` (no MLX) —
  cover register→run routing, no-package error, license rejection, evict→reload.
- [ ] Admission gates on **C10 device eligibility** (`DeviceProfile.eligibility(for:)` — required
  backends present, chip ≥ floor, OS ≥ min) at `register`, and on **memory headroom**
  (`MemoryGovernor`, budget ≈ 0.7× unified memory) at load — it **evicts idle residents LRU** to fit a
  new working set, and rejects a footprint larger than the whole budget. `engine.memory` →
  `MemorySnapshot`. So a manifest's `RequirementsManifest` (footprint per quant, backends, chip, OS) is
  load-bearing — set it accurately or the package may be rejected or mis-budgeted.
- [ ] Expose a model's **variant catalog** (e.g. `QwenModel.allPublished`) with per-variant
  `requirements`, and use `engine.admissibility(for:)` (→ `Admissibility`) to report/filter what the
  current machine can load *without loading* — the Model-Manager seam + a handy startup sanity check.
  Caveat: a package's **static `manifest`** can't vary by configured variant, so engine admission gates
  on the default variant's requirements; per-variant gating needs per-variant requirements.
- [ ] **Footprint is conservative, not variant-aware.** `MemoryGovernor.footprint(for:)` charges the
  *largest declared footprint that fits the budget*. A multi-footprint manifest (variant package, e.g.
  NAFNet signage 0.6 GB / width64 2.0 GB) therefore **over-reserves to the max variant** regardless of
  which variant the config selected — safe (never under-reserves), but over-reserves on constrained
  devices. Not a wrapper bug; config-aware footprint is an open engine enhancement. (Declaring a single
  footprint under-reserves the big variant — don't.)
- [ ] **Multi-package per capability (shipped 2026-06-12, engine-side, no contract bump).** The
  registry is keyed by **`PackageID`** (defaults to the manifest's first surface name — "lens-t2i",
  "qwen-image-edit" — falling back to `provenance.sourceRepo`; pass an explicit `id:` to register
  the same package twice, e.g. bf16 vs 4-bit). A capability can be backed by N packages; routing
  goes to its **default** (last registration wins — preserves the historical swap flow), re-pointed
  via `setDefault(_:for:)` without re-registering. Per-request selection:
  `run(request, package:)` / `prepare(capability, package:)` / `evict(capability, package:)` +
  `evict(package:)`. `register` is `async`, returns the `PackageID`, and re-registering an existing
  id replaces it (evicting any stale resident). Residents + LRU are package-keyed, so one
  registration serving N capabilities is constructed ONCE and shared (previously one instance per
  capability). `packages(for:)` / `defaultPackage(for:)` / `manifest(for:)` / `residentPackages`
  are the Model-Manager seams. Two coexisting same-capability backers (Lens + ERNIE-Turbo on
  `textToImage`) is the motivating case — the app picks per device tier.
- [ ] **Debugging a core pinned by git tag:** the wrapper pins its core (`from: "x.y.z"`), so app builds
  use the resolved checkout, not your local clone. To make local core edits live (probes, a fix), **add
  the local core folder to the workspace** — a same-identity path member overrides the git dep. Drop the
  override (and re-tag) when done so the wrapper resolves the tag again.
- [ ] `any ModelPackage` / `any CapabilityRequest` / `any CapabilityResponse` are `Sendable`, so they
  cross the engine actor + store in actor state cleanly. Construct via
  `PackageRegistration.of(Type.self)` (a `.registration` convenience property is per-package, not on
  every `ModelPackage`).

## Web retrieval / grounding (`MLXRetrievalKit`)

- [ ] Current-knowledge access is a **reusable engine capability**, not app code: `MLXRetrievalKit`
  (MLX-free, network-only) + `MLXRetrievalKitContracts` (Foundation-only). `WebSearchProvider` seam,
  `BraveSearchProvider` (Brave API), `RetrievalService.retrieve(query)` → budgeted `RetrievalResult`.
- [ ] **Consumption is RAG**, not tool-use (v1): `RetrievalService.brave()` → retrieve → prepend
  `RetrievalResult.groundingText()` as a system/context turn before generation. Returns *sources not
  prose*; `retrieve` never throws (failure → empty `degraded` result, so the model just answers from
  its own knowledge). Key via `BraveKeyStore` (`BRAVE_API_KEY` env or UserDefaults).
- [ ] Keep retrieval **MLX-free** — it depends only on the Foundation contracts; full-page extraction +
  summarization are injected seams (Phase 2), so grounding never links Metal.
- [ ] Engine-management settings (the Brave key, enabled flag, depth) belong in the shared `MLXEngineUI`
  sidebar (`EngineSettingsView`). Put the key store / prefs (`BraveKeyStore`, `WebSearchPreferences`) in
  the **Foundation-only contracts** so the UI and the retrieval consumer share the same UserDefaults
  keys (UI depends on Contracts, not the impl).
- [ ] **Live/Metal MLX gates: prefer a CLI mode of an executable target over ANY test framework.**
  History: the SPM metallib workaround (colocate the `Cmlx` bundle in `.build/debug/`) only worked
  for XCTest, not swift-testing (separate helper process — VoxCPM2 2026-06-11). Bernini 2026-06-12
  went further: the test product's nested `mlx-swift_Cmlx.bundle` assembly can corrupt persistently
  (`missing creator for mutated node` build warning; survives product wipes AND clean rebuilds),
  after which **every MLX-touching test dies on "Failed to load the default metallib" — including
  CPU-only suites**, because mlx initializes Metal on first op regardless of the default device
  (a global `Device.setDefault(.cpu)` pin does not prevent the init). Meanwhile **plain `swift run`
  executables load the metallib reliably** (products-dir mainBundle lookup) and do full GPU
  inference — the old workspace-wide "live inference under Xcode only" rule is stale for current
  mlx-swift. Pattern: ship heavy/Metal gates as CLI modes (`swift run RunModel --s4-gate`), keep
  test-framework variants env-gated as repros, and leave only never-evals (key-path/structural/
  scalar) in `swift test`.
- [ ] **App-side version drift — verify what actually resolved, two layers.** A wrapper pinned by
  **local path** always uses its current local-checkout HEAD (no pin) — good, the app gets the latest
  commit for free. But a wrapper/core pinned by **tagged URL** (`from: "x"`) does NOT auto-bump: the
  app's `MLXEngine.xcworkspace/.../Package.resolved` stays on whatever was first resolved (Qwen3-TTS
  silently ran v0.1.0 for weeks). To bump it: (1) `git fetch --tags --force origin` in Xcode's package
  **mirror** (`DerivedData/.../SourcePackages/repositories/<pkg>`) — a freshly-pushed tag isn't there
  yet, so a naive Package.resolved edit silently re-reverts on build; (2) then edit the pin + resolve;
  (3) confirm via the DerivedData `checkouts/<pkg>` `git describe`.
- [ ] **Publish the Python-MLX weights in CANONICAL module-key layout → Swift load is remap-free**
  (anima-swift, 2026-06-26). If the Python rung exports each component via
  `mx.save_safetensors(dict(tree_flatten(model.parameters())))` (after `mx.eval`), the on-disk keys ARE
  the Swift module's flattened keys (gamma 1-D, Conv `(O,kt,kh,kw,I)`, `.conv`/`.to_out.0` wrapping all
  baked in) — so the Swift loader is `loadArrays → ModuleParameters.unflattened → update`, ZERO sanitize,
  vs the donor's diffusers-checkpoint loader that had to insert `.conv`/squeeze gamma/transpose convs.
  Enforce the contract with a strict `loadStrict` (module flattened keys == file keys, 0 missing/0 unused).
  Anima's VAE (verbatim decoder-only lift of qwen-image-edit-swift/QwenVAE.swift) loaded the real published
  `vae-bf16` this way and decoded **cos 1.000000 / maxabs 6.7e-6** vs the Python golden on the first gate.
  Lesson: decide the published layout on the PYTHON side to make the Swift side trivial — and gate the FIRST
  component through the actual published artifact early; it derisks deps-resolve + key-contract + donor-lift
  + `swift run` GPU all at once.

## Multi-component assembly + contract growth (TRELLIS.2 image→3D, 2026-06-26)

- [ ] **Reduced-precision (`manual_cast` bf16) ports: norms must CAST BACK to the input dtype.**
  Oracle norm modules (`LayerNorm32`, per-head RMSNorm) and the modulation add compute in fp32 but
  `return x.astype(x_dtype)` — i.e. they return the *compute* dtype, not fp32. A naive Swift port that
  returns fp32 is **invisibly correct in an all-fp32 path** (Cores validated fp32 → still pass) but
  **silently diverges the moment you add the bf16 compute lever** (the block runs bf16 between ops in
  the oracle, fp32 in your port). Fix: cast every norm/modulation result back to the input dtype
  (`out.asType(x.dtype)`) — a no-op for fp32, required for bf16. Catch it by gating the *assembled*
  model (which applies the cast), not just the fp32 block. Corollary: even on the "bf16 path" the
  rope/SDPA-softmax/affine upcast to fp32 internally, so after the first residual the block is mostly
  fp32 anyway → assembled parity stays ~1e-7, NOT bf16-grade. If you see bf16-grade (~1e-2) drift,
  you have a real cast mismatch, not "expected bf16 noise."
- [ ] **Assemble validated blocks on SMALL SYNTHETIC configs first; key weights by the REAL module
  param paths.** The "full-model assembly" (stage loops, in/out projections, final norm, dtype casts,
  coord/subdiv threading) is its own parity surface even when every block is already validated — wire
  bugs (loop order, an eps that differs from the block default, a missing cast, a dim÷8 constraint)
  live here. Dump the real module at a tiny config (random weights via `tree_flatten`/`unflatten`),
  port the assembly, gate on the CPU stream. Key the Swift weight dict by the **real `blocks.i.j.…`
  param paths** (project to your helpers' short keys inside) so the later real-safetensors load is a
  pure key-match, not a second mapping. Watch model-specific constraints (TRELLIS C2S up-block needs
  input channels ÷ 8 for the skip-repeat) — pick the synthetic config to satisfy them.
- [ ] **A "dense (B,N,C)" block is usually bit-identical to its "sparse (N,C)" twin at B=1.** Don't
  re-port it — reuse the sparse block per batch item (squeeze B), generate the grid coords the dense
  path implies (meshgrid 'ij' row-major to match a `reshape(B,C,-1).transpose` flatten), and validate
  the equivalence with one parity gate. (TRELLIS structure-flow reused the SLat-flow stack this way.)
- [ ] **Adding a `Capability`/`CanonicalOutput` case is safely additive — the only exhaustive switch
  to update is the derived `canonicalOutput` mapping.** `Capability` is `CaseIterable`, so `.allCases`
  consumers absorb a new case for free; a net-new artifact kind needs a `CanonicalOutput` case + the
  one `canonicalOutput` arm. Grep the engine for other exhaustive `switch` over those enums before
  assuming a sweep is needed (there were none beyond the property). Mirror an existing capability file
  (e.g. `Matting.swift`) for the new `*Request/*Response/*Contract`, add the artifact to
  `Artifacts.swift`, bump `ContractVersion` with a changelog line. This is a STOP-AND-ASK core change.
- [ ] **A custom non-SPDX weight license can be *functionally permissive* — allowlist it deliberately,
  with the obligations recorded.** Precedent already in `permissiveAllowlist`: `ltx2Community`
  (revenue gate + non-compete) and `funasrModel`. The DINOv3 License (commercial OK, redistribution OK,
  **no revenue gate, no non-compete**, only "Built with DINOv3" attribution + standard AUP) is *less*
  restrictive than `ltx2Community` → added as `SPDXLicense.dinov3`. Read the actual license text, compare
  to the existing allowlist precedents, and **carry the attribution/AUP obligation forward** (note it on
  the manifest + product). For a multi-checkpoint pipeline, the manifest's single `weightLicense` is the
  **most-restrictive** component.
- [ ] **Host-side output tooling (mesh→GLB, etc.) belongs in the conformer/runtime layer, NOT the
  pure-MLX `Core`** — and validate it with an external loader. The GLB encoder lives in `Trellis2Kit`
  (imports MLXToolKit, emits the `Mesh` artifact), keeping `Trellis2Core` engine-free. Validate the
  bytes by loading them back (`trimesh` in the oracle venv: vert/face counts, bbox, finite, in-range
  indices) — a self-written binary format passes a shape check but can still be malformed.
- [ ] **A `-swift` package that depends on `MLXToolKit` inherits the engine's macOS floor.** MLXToolKit
  is `swift-tools 6.2` / `.macOS(.v26)`; a consumer at `.macOS(.v14)` fails resolution
  ("requires minimum platform 26.0"). Use the **string form `.macOS("26.0")`** to raise the floor while
  staying on `swift-tools-version:5.9` — avoids opting into Swift 6 strict-concurrency language mode
  (which would cascade concurrency errors through MLX GPU-state types). Bump any sibling probe/test
  package to match.
- [ ] **Offline contract build is real leverage even when the model can't run yet.** A `ModelPackage`
  conformer whose `load()`/`run()` throw `…assemblyPending` still compiles against `MLXToolKit` and
  proves the manifest/dispatch/license-gate (a runtime `runManifestCheck` confirming C0 version + C7/C8
  `admitted` + C1→artifact). Ship that skeleton first; it derisks the contract surface before the
  multi-GB graph exists. Declare placeholder footprints from the Python-oracle e2e measurement, clearly
  flagged for re-measure.
- [ ] **Profile peak memory PER STAGE before you declare a footprint or "optimize" precision.**
  `Memory.peakMemory` (`MLX.GPU.peakMemory`) is a never-auto-reset global high-water — one end-of-run
  read is useless for locating a spike. Wrap each stage with `MLX.GPU.resetPeakMemory()` + an
  `eval(...)` + a read; it becomes a profiler that names the culprit op in one run. TRELLIS.2 read 44 GB
  end-to-end but per-stage no stage exceeded 12 GB once block boundaries eval'd — the "44 GB" was MLX
  evaluating the whole 512³ decoder as one deferred graph. The fix (eval at block boundaries; stream
  K-reductions; cast *activations* not weights) lives in `mlx-porting/references/streaming-decode.md`
  ("deep sparse/conv decoders"). Net: **44.24 GB → 11.93 GB, and faster** (deferred graphs stall on
  memory pressure). Gate the lever with an injected-input parity (oracle coords+latent → Swift tail →
  vert/face counts): TRELLIS f16 decoder kept **vert ratio 1.000**.
- [ ] **Declare the MEASURED Swift peak as `residentBytes`, not the Python-oracle placeholder.** After
  the memory lever, set the footprint to the per-stage max you measured (TRELLIS: 11.93 GB → declared
  13 GB for headroom, since peak scales with output size — voxel/token count). The placeholder from the
  Python e2e is only valid until the Swift package runs; re-measure and replace.
- [ ] **Image input → MLXArray is host-side in the conformer (CoreGraphics), not the `Core`.** Decode
  the canonical `Image` artifact (png/jpeg via `CGImageSourceCreateWithData`; `rawBGRA8` via a direct
  `CGImage`), draw into an N×N RGBA8 `CGContext` (`.high` interpolation), then normalize to the model's
  convention (ImageNet mean/std for DINOv3) → `(1,N,N,3)`. Keep it in the `*Kit` layer (it imports
  Foundation/CoreGraphics); the pure-MLX `Core` takes the normalized array. Background removal is
  composable preprocessing (shipped BiRefNet matting), not a `Core` concern — V1 may skip it (note that
  the input background then leaks into the result).
- [ ] **Prove the engine integration with an engine-driven CLI, not just the Xcode app.** Stage-2 step 7
  ("promote to MLXServeCore") is fully exercisable by a `swift run` executable target in the `-swift`
  package that depends on `<Pkg>Kit` + **`MLXServeCore`** + `MLXToolKit` and drives the real coordinator:
  `MLXServeEngine()` → `register(Pkg.registration, configuration:)` (runs the C7/C8 license + C10 device
  gates) → `prepare(capability)` (memory admission) → `run(request)` → decode the canonical artifact →
  `evict`. It loads the metallib reliably (plain executable, unlike `swift test`), runs real GPU
  inference, and prints the engine-charged footprint (`engine.memory.residents[capability]`) — so the
  engine path is validated WITHOUT the Xcode link blocker. The Xcode `ValidationView` then adds only the
  interactive picker + visual viewer. (TRELLIS.2 `RunTrellis2Engine`: register→prepare 13 GB→run
  39s/10.7 GB→decode Mesh→GLB→evict, all before the app was even linked.)
- [ ] **rembg/background-removal as composable preprocessing: do the alpha-USING step in the `*Kit`
  conformer, keep the alpha-PRODUCING model a separate capability.** Faithfully reproduce the reference
  preprocess (TRELLIS.2 = honor input alpha → **bbox-crop to a square around the foreground → composite
  `RGB×alpha` on black** → resize+normalize; the crop/recenter is the real quality lever, not just the
  mask) inside `ImagePreprocess` — premultipliedLast `CGContext` gives you `RGB×alpha`-on-black for free.
  Do NOT hard-depend on the shipped matting package to PRODUCE the alpha: many `-swift` wrappers pin
  `mlx-engine-swift` by **remote URL**, which collides with your local `.package(path:)` on the same
  identity and jams resolution (the wrapper's own Package.swift warns about this). Instead honor an alpha
  the input already carries (the reference's `has_alpha` branch — pre-masked RGBA), and leave producing
  alpha for a plain RGB input to the consumer's `.matting` step. Validate via the engine CLI: the
  structure/voxel count should drop to the rembg regime (TRELLIS: 5837 no-rembg → 2439 ≈ oracle 2981).
- [ ] **Sandboxed app + HF-cache weights: `homeDirectoryForCurrentUser` is the CONTAINER, so the
  package's `~/.cache/huggingface` fallback is invisible.** Give the configuration an explicit
  `weightsRootOverride: URL?` and have the app pass a **security-scoped grant** to the real cache dir
  (same pattern as out-of-sandbox snapshot models). Make `resolve` understand BOTH a flat
  `<root>/<repo>/<rel>` layout AND the HF-cache `<root>/models--<ns>--<repo>/snapshots/<hash>/<rel>`
  layout under that root. Gate `canRun` on the grant for the input kind (the `.image` arm too, not just
  the instruction kinds).
- [ ] **GLB / 3D mesh artifact: there's no native SceneKit/ModelIO GLB loader on macOS — use
  `QLPreviewView`** (QuickLookUI / `import Quartz`) pointed at the temp `.glb` for an inline,
  rotate/zoom viewer in the harness; `previewItem = url as NSURL`. The mesh result panel reports
  vertex/face counts + GLB byte size (the "did it work" numbers) alongside the render, and carries the
  weight-license attribution label ("Built with DINOv3") per the C7 obligation.

## Performance & video-output gotchas (LTX-2.3 profiling deep-dive, 2026-07-01)

Found by profiling why LTX 48-frame generation ran >1000s at <10% GPU. All four generalize to any
video/DiT port — the first is a **cross-package latent bug** in the shared encoder.

- [ ] **A two-track `AVAssetWriter` (video + audio) DEADLOCKS if you append all video before any
  audio — append/finish the audio track FIRST.** (CORRECTED diagnosis 2026-07-01 — this was first
  misread as hardware-VideoToolbox/GPU contention.) With two inputs and `expectsMediaDataInRealTime =
  false`, the writer INTERLEAVES tracks: once the appended video runs **~1.8 s ahead** of a
  still-empty audio track (~43 frames @24 fps software; ~32 hardware), it parks the video input's
  `isReadyForMoreMediaData` *waiting for audio* — an audio-appended-last loop then spin-waits forever
  at ~0 % CPU ("looks like a loop", GPU <10 %). Fix: build the audio `CMSampleBuffer`, `append` it and
  `markAsFinished()` the audio input **before** the video frame loop — the writer then never blocks
  video on interleave. Validated 113 frames + audio on BOTH hardware and software encoders (~3 s).
  Corollaries: (a) **single-track/video-only writers (frame-stream-native → RIFE/SeedVR2) cannot hit
  this** — don't cargo-cult a software-encoder default there; (b) the software-forcing spec key
  `RequireSoftwareOnlyVideoEncoder` **does not exist in the SDK** — only
  `EnableHardwareAcceleratedVideoEncoder: false` is real; (c) still NEVER leave an unbounded
  `while !isReadyForMoreMediaData` loop — bound it with a timeout AND check `writer.status == .failed`
  inside it (a failed writer reads as "not ready" forever and masks the real error). **Meta-lesson for
  stress gates: reproduce ALL tracks the real path writes** — our first gate passed `audio: nil`,
  which silently removed the failure mode and produced a coherent-but-wrong contention theory (the
  "fix" then validated on a 41-frame clip that simply sat under the 43-frame threshold). A
  hardware-speed refinement remains driving `VTCompressionSession` directly + zero-copy IOSurface
  handoff (the `h00mankind/MetalVideoEngine` pattern) instead of per-frame GPU→CPU `asArray` + Swift
  pixel copy.
- [ ] **The first DiT forward pays a huge one-time Metal kernel-compile cost — warm it up during
  `load()`.** A multi-block DiT JIT-compiles its whole kernel set on the first forward: measured ~162s
  cold (24f) / ~49s with a warm OS shader cache. It's **single-threaded CPU work** → the GPU sits <10%
  and only one core is busy (invisible on a multi-core monitor), so mid-generation it reads as a hang.
  MLX specializes kernels per shape, so **every new frame count / resolution recompiles**. Fix: a
  `warmup()` — a tiny `nv=1` forward (dummy zero latents + dummy text embeds, output discarded) in
  `load()` — moves the cost into the expected "Loading" phase; measured s1-step0 162s → 4.1s. Per-8-layer
  `eval` keeps command buffers small so warmup doesn't trip the watchdog (same reason the real cold
  first-forward completes). Gate it behind an env escape hatch.
- [ ] **Prefer fused `MLXFast.rmsNorm`/`layerNorm` over manual `mean/rsqrt`(/mean-subtract) chains —
  but ONLY where you hand-rolled them.** The manual affine-free norm is ~6 ops → 6 kernels; the fused
  kernel is one (fp32-internal reduction, so parity-equivalent — gated at dit-full cosine 0.999914 / vae
  1.000000 / e2e 0.999971). Hundreds of tiny norm kernels across the blocks collapse to a handful →
  **DiT warmup compile 38.8s → 7.7s** AND faster steps. For affine-free RMS pass `weight =
  ones([x.dim(-1)]).asType(x.dtype)` (the block applies its own scale/shift). **Nuance (checked across
  the image family):** `MLXNN.RMSNorm` and `MLXNN.LayerNorm` ALREADY call `MLXFast.rmsNorm`/`layerNorm`
  internally — so any port using the stock modules is already fused; there is no gain and nothing to
  change (Ernie/Lens/Qwen-Image-Edit/Anima's Qwen3 TE). This lesson only applies to **functional ports
  that hand-roll norms on a flat weight dict** (LTX-2.3's `rms0`/`layerNormAffineFree`) or a **custom
  norm `Module` with manual math** (Boogu's `Transformer.swift` mean/rsqrt; Anima's `WanRMSNorm`). Grep
  a port for `MLX.mean(… ) … rsqrt` before assuming there's a win — if it's all `MLXNN.RMSNorm`, skip.
  GroupNorm/InstanceNorm/pixel-space norms have no fused kernel, so they stay manual regardless.
- [ ] **Profile every heavy pipeline under the shared instrument — it's what makes all of the above
  findable.** This is now a package, not a per-port reinvention: **`MLXProfiling`**
  (`MetalToolBox/PROD/mlx-profiling`, `mlx-swift`-only, MIT, zero overhead when unset) — the unification of
  the old `LTX2Profiler`/`WanProfiler` under one comparable schema. Enable with `MLX_PROFILE=1` (add
  `=csv` for a CSV, `MLX_PROFILE_DEEP=<subsystem>` for fine per-block barriers). It emits live `[MLXPROF]`
  rows (per-region wall-ms + MLX active/cache + OS `phys_footprint`/workingSet + a `⚠PAGING` flag when
  `phys > GPU.maxRecommendedWorkingSetSize`) so a stall is visible AS it happens, plus a grouped end-of-run
  summary. It's what separates the three look-alike failure modes — all "<10% GPU, long wall-clock":
  **compile** (one slow first step, flat memory), **paging** (phys crosses the working-set ceiling, cache
  balloons), **encoder stall** (happens AFTER the pipeline returns, in the uninstrumented encode — so wrap
  the encode too). Declare from `phys_footprint`, NOT `Memory.peakMemory` (cumulative allocations, misleads
  under the cache cap). **Adopt it instead of writing a new profiler** (precedent: seedvr2 v0.5.0, rife
  v0.6.0; LTX/Wan migrating). Same instrument feeds the manifest footprint (`memory-harness.md`) and the
  efficiency sweep (`package-efficiency.md`).
- [ ] **A streaming `FileHandle.read(upToCount:)` loop needs a per-iteration `autoreleasepool` — or
  every "discarded" chunk stays alive and the residency later kills a GPU eval.** (LTX `--i2v-spot`,
  2026-07-01.) A CLI prewarm that streamed ~50 GB of safetensors left **~60 GB of dead `Data` chunks
  resident** (autoreleased, never drained in the tight loop): `phys_footprint` read a deterministic
  100.57 GB after a "lazy" load, `Memory.clearCache()` freed ~nothing (it's not MLX memory), and the
  next real GPU eval (a 4.9 GB LoRA apply) faulted pages under that pressure inside a live command
  buffer → `kIOGPUCommandBufferCallbackErrorTimeout`, **deterministically** — a watchdog crash whose
  cause was plain ObjC bridging, not Metal. Wrapping the read in `autoreleasepool { }` collapsed the
  load state 100.57 → 40.58 GB and the identical run passed clean. Diagnosis pattern that found it:
  bisect with the model's own escape hatches (`LTX_NO_WARMUP=1` proved 60 GB resident with ZERO GPU
  evals → the residue predated MLX). Generalizes to any Foundation streaming loop (prewarm, hashing,
  upload chunking) in long-lived engine/CLI processes; in-app code paths often dodge it only because
  the engine's own `WeightPrewarmer` does the paging. Corollary: a "mystery +N GB that clearCache
  can't free" is usually NOT MLX — check the ObjC autorelease pool before blaming the buffer cache.
