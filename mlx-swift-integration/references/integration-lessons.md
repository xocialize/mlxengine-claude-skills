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
- [ ] **KV-cache reuse across `run()` calls (mlx-qwen-llm-swift v0.3.0, the reference):** a fresh
  `ChatSession` per run re-prefills the WHOLE history every turn (latency + prefill scratch grow
  linearly with conversation length). `ChatSession` already retains `[KVCache]` across `respond()`
  calls — hold ONE session on the package in an `@unchecked Sendable` box (serialized by the
  `@InferenceActor`, C13) with an **exact-transcript fingerprint**: hit ⇔ incoming messages ==
  held transcript + exactly one new user turn AND same system prompt; on hit send ONLY the new
  turn (`TokenIterator` does NO prefix dedupe — resending history double-encodes it); on ANY
  mismatch rebuild fresh. Traps learned live: (1) the system prompt must seed `history[0]`, NOT
  `instructions:` — instructions re-template into the KV stream on EVERY respond(); (2) the
  responded turn is usually `dropLast`'d out of the history you seeded — remember to record it
  (AND the reply) in the fingerprint or you never hit; (3) drop the held session on any
  mid-generation throw (partially appended cache must never be reused) and in `unload()`;
  (4) `generateParameters`/`additionalContext` are mutable session vars captured per call —
  refresh them instead of rebuilding on sampling/mode changes; (5) held-vs-fresh temp-0 outputs
  are NOT byte-equal by upstream design (the cached generation prompt keeps the empty `<think>`
  block a re-template omits) — gate on counters/recall/latency + exact-match only on
  both-rebuild legs; (6) document the retained KV as intentional active-memory retention so
  pool-telemetry triage doesn't read it as a leak, and expose hit/miss counters + a Logger line
  so consumers can verify hit rate in situ (a consumer that rewrites replies before echoing them
  back silently degrades every turn to a rebuild).
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
  **THIRD cache layer — `~/Library/Caches/org.swift.swiftpm/manifests` (measured 2026-08-01).** A
  freshly-tagged fix kept resolving to the OLD version through: deleting the project
  `Package.resolved`, deleting `DerivedData/.../SourcePackages/Package.resolved`, purging
  `SourcePackages/checkouts/<pkg>`, AND purging `SourcePackages/repositories/<pkg>` (verified
  refetched — `git tag` in the mirror listed the new tags). The stale view lived in the
  **user-level manifest cache**, which no DerivedData purge touches:
  `rm -rf ~/Library/Caches/org.swift.swiftpm/manifests` → resolve → correct version, immediately.
  ⚠️ Raising the consumer's `minimumVersion` to the new tag ALSO appears to fix it — but that only
  MASKS the stale cache (it forces the floor above the cached version), which is how this gets
  misdiagnosed as "the resolver picks the lowest version" or "an Xcode bug". It is neither: SPM
  picks the highest in range correctly once the manifest cache is clear (proven by lowering the
  floor back to the old value after the purge — still resolved to the new tag).
  Order to try: (1) purge the manifest cache, (2) fetch tags in the mirror, (3) delete both
  `Package.resolved`s, (4) resolve, (5) confirm with `git -C .../checkouts/<pkg> describe --tags`.
  Independently, bumping `minimumVersion` to the fix tag is still good hygiene — a floor left at an
  old version lets any consumer legitimately keep serving it.
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

## Wrapper-level live gates + test-input hygiene (Boogu imageEdit RCA, 2026-07-03)

Found root-causing "imageEdit incoherent in-app but CLI gates pass": the CLI gates drove the CORE
directly, so the `ModelPackage` wrapper's run path had **zero live executions** — a wrapper refactor
(v0.1.2 per-request encoder-evict) shipped "gate-green" without its edit branch ever running live.
The actual bug turned out to be elsewhere (the qwen3vl 1024-token-grid divergence,
`swift-port-parity.md` "gate matrix must span the input envelope"), but the coverage hole was real
and cost the first day of diagnosis.

- [ ] **Every ModelPackage surface gets a wrapper-level live gate** — a `--e2e-<surface>-pkg` mode
  in the port's gate CLI that constructs the real `CapabilityRequest` and calls `Package.load()` +
  `run(request)`, exactly as the app does (precedent: `BooguGate --e2e-edit-pkg`). Core-level e2e
  gates do NOT cover the wrapper: per-stage eviction, request decode, size defaults, and response
  encode all live only in the wrapper. Rule of thumb: if a commit touches `run(_:)`, a wrapper-level
  gate must run live before it ships.
- [ ] **Wrapper-vs-core A/B is the cheapest first bisect** for "in-app broken, CLI fine": run both
  gates on identical inputs. Identical output → wrapper exonerated, suspect inputs/envelope;
  different → the wrapper delta is the bug.
- [ ] **Validate your test INPUTS before trusting any repro.** A `samples/` file that was itself a
  NaN artifact (all-black PNG saved by a broken earlier run) masqueraded as an edit input and
  produced instruction-only outputs ("model ignores the ref image") — hours chased before anyone
  looked at the input. When a conditioned model seems to ignore its conditioning, EYEBALL THE INPUT
  first; quarantine known-bad artifacts with a `KNOWN_BAD_` prefix instead of leaving them
  plausible-named in samples/.

## Embedding heads on stock MLXLLM models (mlx-qwen-embedding-swift, 2026-07-04)

- **Check the runtime's inner-model access level BEFORE planning a fork or vendored decoder.**
  MLXLLM's `Qwen3Model` exposes `public let model: Qwen3ModelInner`, and the inner
  `callAsFunction` returns `norm(h)` — the `last_hidden_state` — directly. An embedding variant
  of an already-loadable architecture (Qwen3-Embedding = plain `model_type: qwen3`) therefore
  needs ZERO layer work: load via `#huggingFaceLoadModelContainer`, downcast `context.model`,
  call the inner model, pool. The pre-session plan assumed the qwen3vl lastHiddenState pattern
  or a fork would be needed; a 2-minute access-level grep collapsed that. (Corollary: guard the
  downcast at `load()` with a diagnosable WrongModelTypeError — the MLXVLM-shadowing hazard
  applies to any factory-resolved key.)
- **`import MLXLMCommon` + `import Tokenizers` makes `Tokenizer` ambiguous** — MLXLMCommon
  declares its own `Tokenizer` protocol (`encode(text:addSpecialTokens:)`). `ModelContext
  .tokenizer` is the MLXLMCommon one; qualify `any MLXLMCommon.Tokenizer`. And you still need
  `import Tokenizers` when using the `#huggingFace…` macros (their expansions reference it) plus
  `import MLXHuggingFace` for the macros themselves — "no macro named …" almost always means the
  missing import, not a version mismatch.
- **Last-token pooling's silent-failure mode is the EOS token.** Qwen3-Embedding's tokenizer
  post-processor appends `<|endoftext|>` (151643 — note: NOT `tokenizer.eos_token`, which prints
  `<|im_end|>`) and the reference pools THAT position. swift-transformers honors the
  post-processor (token ids came out exactly equal to the Python fast tokenizer), but append
  defensively when missing — a pooled non-EOS position produces plausible-looking, quietly
  degraded vectors. Gate tokenizer ids EXACTLY (dump ids beside vectors in the parity fixture);
  cosine alone won't localize a tokenizer break.
- **Sequential single-sequence forwards beat left-padded batching for embed surfaces at consumer
  scale.** `cache: nil`, one text per forward: pooling is exact by construction, no mask/padding
  bookkeeping to get wrong, and the batched-GEMM win is noise for a 0.6B model embedding a
  handful of texts. Declare it in the package doc so a future "optimization" doesn't reintroduce
  the padding bug class. Parity int8-vs-fp32 landed at cosine ≈0.9996 with this shape.

## Engine ≥ 0.21.0 owns the GPU buffer-pool policy (N5, 2026-07-05) — package-author implications

- **MLXServeCore now links mlx-swift** (the engine repo's first runtime dependency — allocator
  API only, for the process-global `Memory.cacheLimit` policy + pool telemetry; R-MEM-2 in the
  engine's architecture.md). **MLXToolKit stays dependency-free**, so the step-3 "build the
  contract offline before pulling the MLX graph" workflow is unchanged. Don't "fix" a package's
  Package.swift to drop its own mlx-swift dep on the theory the engine now provides it — packages
  still declare their own.
- **Your `unload()` contract is UNCHANGED.** Packages still call `MLX.Memory.clearCache()` in
  `unload()` (the C-level expectation). The engine's `trimAfterEvict` knob is opt-in,
  default-OFF belt-and-braces on top — never a substitute.
- **Refinement of the metallib lesson above: the ALLOCATOR API also initializes Metal.** Not just
  kernels/evals — a bare `Memory.cacheLimit` get/set or `Memory.activeMemory` read initializes the
  Metal device and dies on the metallib in swift-testing helper processes. The engine's own N5
  bring-up hit exactly this: engine init's cacheLimit write aborted every downstream package's
  offline admissibility suite (they construct `MLXServeEngine` under swift-testing). The fix
  pattern is reusable: **scope any process-global MLX touch that can run under a test framework
  through `try? MLX.withError { … }`** — it converts the default aborting handler into a caught
  throw, and the environments where it fails are exactly those that can't run GPU work anyway,
  so degrading to a no-op is exact. Engines ≥ 0.21.0 do this internally — constructing an engine
  in offline tests is safe again; `engine.appliedGPUCacheLimitBytes == nil` is the tell that the
  write didn't take (unmanaged policy OR metallib-less process).
- **Mem-bench / engine-e2e measurement interplay:** an engine-driven live gate (the
  `--engine-e2e` pattern) now runs under a bounded pool by default (min(2 GB, 5% of budget)),
  which can shave phys peaks vs historical unbounded runs — pass
  `gpuCache: .unmanaged` at engine construction when a measurement must match pre-0.21.0
  numbers. Package-direct benches (`pkg.load()`/`run()` with no engine, e.g. `--mem-bench`)
  never construct an engine and are unaffected.

## Weight/adapter downloads: the async `download(for:delegate:)` progress trap + honest-phase doctrine (P2, 2026-07-05)

- **`URLSession.shared.download(for: request, delegate:)` (the async convenience) never delivers
  `didWriteData` progress to the task-level delegate** — verified empirically (LTX `--lora-fetch-gate`:
  0 progress callbacks on a 336 MB fetch). If a download must report progress, use the CLASSIC
  pattern: a session-level delegate + explicit `downloadTask(with:)` bridged to async via a
  one-shot continuation (+ `withTaskCancellationHandler` cancelling the URLSession task, so Swift
  Task cancellation → `URLError.cancelled` → rethrow as typed `CancellationError`, keeping hosts'
  "cancelled ≠ failed" reporting intact). Relocate the temp file synchronously inside
  `didFinishDownloadingTo` (it dies on return) — into the destination directory, so the final
  atomic move never crosses volumes.
- **The `WeightDownloadProgress` TaskLocal sink does NOT flow onto URLSession's delegate queue.**
  Capture `WeightDownloadProgress.sink` in the calling task and hand it to the delegate; a bare
  `WeightDownloadProgress.report(...)` inside a delegate callback is a silent no-op.
- **Honest-phase doctrine for hosts (the LTX R5 lesson):** any multi-GB first-use fetch that runs
  inside a package's `run()` (adapters, per-request weights) reads as "Generating…" + a GPU-idle
  hang to the operator, AND contaminates the run timer (R5: runSeconds 3628.7 s of which ~52 min
  was a silent 4.9 GB HF download at anonymous ~1.4 MB/s). Hosts should pre-materialize via the
  package's cache API (`isCached` + `ensure` with a bound sink) in an explicit
  "Downloading…" phase BEFORE starting run timing; the wrapper's own in-run `ensure()` then finds
  the file warm. Mirror the wrapper's cache-root derivation EXACTLY (engine-stamped store root
  from `useModelStore` → user-caches fallback) or the pre-materialization warms the wrong path.
- ~~**Watchdog-family datum (one observation, desktop M5 Max, 2026-07-05)**~~ ✅ **DIAGNOSED
  2026-08-03 — this was weight page-in from a SLOW WEIGHT VOLUME, and `prewarmPaths` has a
  ceiling.** The original sighting: a two-stage q4 run at a tiny shape (384×224×9f) died with
  `kIOGPUCommandBufferCallbackErrorTimeout` during the Gemma lazy cold load
  (`GemmaEncoder.load` → `loadWeights` → eval, pread threads active) DESPITE file prewarm having
  paged all 28.5 GB. Root-caused in LTX ISSUES.md **I9** after it recurred and blocked every CLI
  e2e bench:
  - **Mechanism.** MLX safetensors arrays are **lazy** — `load` only maps them (a 35.4 GB
    checkpoint "loads" in 1.4 s), so the bytes are actually pulled **inside the first
    generation's Metal command buffers**. If the weight tree sits on slow storage (here a **USB**
    volume, ~250–475 MB/s) and the process working set grows past what still lets the OS serve
    those reads from page cache, the reads fall back to the device *while a command buffer is
    live* and the fault stall trips the watchdog.
  - 🔑 **Prewarm cannot fix this, and that is the generalizable correction.** Prewarm only warms
    the **OS page cache**; it cannot pin those pages against eviction by the process's own later
    allocations. A run with the cache **hot** (prewarm 5.9 s) still died. **On any box where
    `weight tree + working set > RAM`, `prewarmPaths` buys the first few GB and nothing more.**
  - **Only the largest-working-set arm fails**, which is why it masquerades as a quant/component
    bug: at 704×512×121f bf16 (66 GB working set) crashed 0/7 across three sessions while int8
    (35 GB) and int4 (26 GB) passed on the same volume, same binary, same session.
  - **Discriminator, in order:** (1) `diskutil info $(df --output=source <weights> | tail -1)` —
    check `Protocol:`; (2) restage the tree on internal/PCI-E storage and re-run the identical
    arm (this alone flipped 0/7 → 3/3); (3) only then suspect the component the profiler blamed.
  - ⚠️ **Localization ≠ cause.** A profiler pointed at `encode/connector`, and an int8-connector
    default correlated 4/4 — both were amplifiers, not causes: that init materializes every packed
    weight in ONE `eval`, i.e. the largest fault-exposed command buffer in the run. The exonerating
    A/B is the storage swap with everything else held fixed.
  - **App paths often dodge it** because the engine governor keeps the working set lower and
    orders allocation differently — so "works in the app, dies in the bare CLI gate" is a
    *symptom of this*, not evidence against it.
  - **Durable fix beyond fast storage** (open, engine-side): materialize weights OUTSIDE live
    command buffers, so slow storage degrades to *slow* rather than *fatal*.

## Sandboxed area-app harness: headless validation runners (IndexTTS2 Stage 2, 2026-07-09)

- **Redirected stdout is BLOCK-buffered — a headless in-app validation runner that `print()`s
  looks hung when launched with `> log 2>&1`.** Flush per line (`fflush(stdout)` inside the log
  helper) and `exit(0)` when the run completes (headless-autorun semantics) so the harness can
  tail the log live and detect completion. Symptom otherwise: app runs for minutes, log file
  stays 0 bytes.
- **You cannot stage weights/fixtures INTO a sandboxed app's container from the agent shell** —
  `~/Library/Containers/<bundle-id>/Data/...` writes fail with `Operation not permitted` (TCC),
  and the sandboxed app can't read `~/Development/...` paths either (user-selected-files only).
  Two clean lanes: (a) small fixtures → bundle them as app resources (a
  `PBXFileSystemSynchronizedRootGroup` auto-bundles non-source files; read via `Bundle.main`);
  (b) weights → let the package's auto-materialization DOWNLOAD into the container store —
  which doubles as the live first-run validation of the WeightSourcing path. Don't fight the
  sandbox with env-var paths; they're unreadable anyway.
- **Consuming an UNRELEASED engine contract from an area app: convert the app's
  `XCRemoteSwiftPackageReference "mlx-engine-swift"` to an `XCLocalSwiftPackageReference`
  in-place (same object ID → existing `XCSwiftPackageProductDependency` pointers stay valid),**
  and point WIP packages' `Package.swift` at the same local path. Mixing a project-level remote
  ref with a package-level local path dep for the same identity fails resolution. Restore the
  remote pin when the engine change ships in a tag (note it as publish-time debt).
- **A WIP package whose runtime needs checkpoint data shipped only as torch pickles
  (feat matrices, mean/var stats, CampPlus .bin)**: dump to npy/safetensors with the oracle venv
  and BAKE into package resources, with a provenance comment marking them as weight-license
  data (not port-code license) + a debt note to move them into the weight repo at the
  own-conversion re-publish. Swift never parses pickles.

## WeightSourcing/MAT retrofit sweep across an accumulated fleet (audio, 8 packages, 2026-07-09)

- **The retrofit is mechanical once the first package locks the pattern — budget ½ day was
  generous; simple single-source wrappers take ~20 min each.** Per package: (1) add a
  `modelDirectory: URL?` explicit-dir escape hatch (non-Codable) — REQUIRED even where none
  existed, because MAT-5's satisfied-config check calls `missingWeightSources(storeRoot: nil)`
  and can only be satisfied via explicit paths; (2) `WeightSourcing` extension with a
  `requiredFiles` presence probe derived from what `load()` actually reads; (3) copy
  IndexTTS2's `WeightMaterializer.swift` verbatim (+ swift-huggingface dep) and reroute
  `load()`; (4) `resolved(storeRoot:)` + `WeightPrewarming`; (5) `MaterializationTests`
  (satisfied temp-dir probe files can be 1-byte — the gate only checks existence).
- **Retrofits REPLACE legacy download paths, and layouts differ — audit where old snapshots
  landed before assuming reuse.** Three legacy layouts found in one fleet: swift-transformers
  `HubApi(downloadBase:)` → `<root>/models/<org>/<name>` (MOSS — re-downloads once), the
  kokoro core's `HubCache` → `<root>/mlx-audio/<org>_<name>` (re-downloads once), and
  qwen3-tts's own `materialize()` → `<root>/<org>/<name>` (already ModelStore layout — zero
  migration). Note the one-time re-download in the commit message; don't build fs-dependent
  legacy probes into `resolved()` — keep it pure (store layout or explicit dir).
- **When the CORE already owns a store-layout downloader with a progress hook (qwen3-tts's
  `HuggingFaceDownloader`), keep it — the WeightSourcing delta is declaration + missing-set +
  `WeightDownloadProgress.report` forwarding, not new download code.** Map source *i* of *n*
  onto fraction `[i/n,(i+1)/n)` exactly like the reference materializer.
- **Variant-keyed declarations: probe generically where checkpoint file names vary.** Across
  qwen3-tts's 25-checkpoint catalog the weights are `model.safetensors` OR shards, so the
  presence probe is "any `.safetensors` in dir" (the core's own `weightsExist` semantics) +
  config/sidecar files — a hardcoded filename probe would false-negative half the catalog.
  Run the MAT gate over `allPublished`, not just the default (it's offline — 25 configs cost
  milliseconds). Conversely a per-VARIANT repo (mel-roformer) just computes `weightSources`
  from the variant and the tests assert per-variant honesty (populate one repo in a temp
  store → the other variant still reports missing).
- **Fleet-wide behavior change to call out: rootless (no `modelsRootDirectory`) configs with
  missing weights now THROW instead of silently downloading into a default cache.** That
  silent path is exactly the conformance smell the contract kills; real apps always stamp a
  store root. Dev/test flows use explicit dirs.
- **`load()`-from-directory beats `fromPretrained(repoID)` for wrappers**: every core in the
  fleet already had a public from-directory lane (`fromModelDirectory`, `WeightLoader.
  loadWeights(into:from:)`, `Pipeline.load(from:)`) — loading from the resolved directory
  makes the no-download path genuinely network-free (HubApi `snapshot()` re-checks remote
  metadata even on cache hits).
- **Before retrofitting, verify the package actually DOWNLOADS — bundled weights are a distinct
  posture (mlx-realesrgan-swift, 2026-07-09).** Real-ESRGAN's "private-cache download" premise
  was false: all 3 checkpoints (~2 MB each) are vendored `Bundle.module` resources, present on
  every fresh install. Its fresh-install failure was `MLXServeEngine.needsDownload`'s heuristic
  (no prewarm paths + no store marker ⇒ "needs download" forever). The bundled posture:
  `ModelStorable` (MAT-1) + `WeightPrewarming` over the bundle URLs — always-present prewarm
  paths short-circuit `needsDownload` to false. Do NOT declare `WeightSourcing` for bundled
  weights: a network source either fails MAT-4 honestly or forces a download of bytes already in
  the binary. MAT-2..5 are N/A until the engine grows bundled-weights gate vocabulary (gap
  filed 2026-07-09). E2e-assert the fix through a real engine
  (`register → needsDownload == false`), not just path existence.
- **Materialization audits must cover EVERY network touch in `load()`, not just weights — the
  qwen25vl tokenizer side-fetch (2026-07-09).** `Qwen25VLPipeline.load` called
  `AutoTokenizer.from(pretrained: "Qwen/…")`, silently fetching the stock repo into HubApi's
  PRIVATE cache — so even a fully materialized snapshot needed the network on first load, and
  the bytes were invisible to the ModelStore. mlx-community snapshots are self-contained;
  switch to `AutoTokenizer.from(modelFolder: directory)` and pin the special-token ids in a
  cheap gated test (tokenizer sidecars are a few MB — no need for the multi-GB snapshot to
  validate the local-tokenizer lane).

## CAN-gate fleet retrofit lessons (2026-07-09 sweep, ~38 conformers / ~30 repos)

- **The validation-first `run()` shape was copy-pattern-propagated across the ENTIRE fleet.**
  All but one conformer (edgetam) had `guard … notLoaded` / capability guards before the
  cancellation check — the exact CAN-1 signature failure, inherited by scaffolding from older
  siblings. Even LTX, the package that PROVED the cancellation program, had the entry-ordering
  drift (a live bench structurally can't catch it: it always runs loaded). Scaffold new
  packages from a post-0.27.0 reference, never an older sibling.
- **Four classes of real cancellation bug the offline gate flushed out:**
  1. `try?` around a cancellation seam (Helios's `try? onChunk?()` silently discarded the
     documented per-chunk cancel). Grep closure-threading seams for `try?`.
  2. Silent partial-return: mlx-swift-lm's `ChatSession.respond` ends its stream on cancel and
     returns partial text without throwing — and on a KV-reuse package the partial turn got
     re-held into the session (stale KV). Post-call `try Task.checkCancellation()` after every
     ChatSession consumer.
  3. Laundering on SIDE paths, not the inference core: async system-framework calls
     (AVAssetImageGenerator) and network fetches (LoRA download, `URLError(.cancelled)`) inside
     diagnostic-wrapping catches. Add `catch is CancellationError { throw }` first; map
     `URLError(.cancelled)` back to `CancellationError`.
  4. Core "cancel support" that never reads the Task: both separation cores' per-chunk
     `checkCancelled()` only read their own `cancel()` flag. Grep the helper for
     `Task.checkCancellation`, not the word "cancel".
- **A mid-loop bail can CREATE a laundering path**: an empty-output guard downstream of a
  `break` (IndexTTS2's `emptyGeneration`) surfaces instead of the cancel — put a throwing
  checkpoint between the loop and any partial-result consumer. Bail AFTER the first append so
  a trailing `concatenated([])` can't crash; place stage-seam checks BEFORE shape-sensitive
  consumers (Trellis2: partial feats crash flexiPost).
- **CAN-3 honesty rules of thumb:** the manifest is ground truth for `longRunImplied` (stale
  briefs/docs mislead — ddcolor's re-baselined 4.54 GB peak flipped it); a flat pre-1.14
  footprint (no peakActivationBytes) defeats the long-run implication on genuinely heavy
  packages (anima) — flip the suite assert when the split lands; "iterative model" ≠ iterative
  execution (SEA-RAFT's loop is lazy graph construction — the cadence test is whether an
  eval/item/throwing-emit sits inside the loop); "streaming" describes causality, not
  execution (Mimi encodes one whole-clip graph). RunProgress-evidenced means per-UNIT
  reporting, not phase presence.
- **Mechanics:** the 0.9.x→0.27.0 engine pin jump is painless (pre-1.14 manifests compile
  unchanged); an `exact:` mlx-swift pin blocks it (engine floors 0.31.5 — relax to `from:`);
  ~a third of the fleet had no MLXServeConformance test dep and several repos had NO test
  target at all — the CAN suite is a fine first test target. Core loop fixes ride patch tags
  on the core repo + an explicit `from:` bump in the wrapper.

## First `stt` capability — Nemotron 3.5 ASR streaming (2026-07-10)

The first speech-to-text provider (contract 1.20.0). A cache-aware FastConformer-RNNT
language-port (Python-MLX `Blaizzy/mlx-audio` `nemotron_asr` → Swift-MLX). Lessons:

- **A streaming model does NOT need a streaming engine contract.** Nemotron's cache-aware
  chunked decode is package-INTERNAL memory discipline (bounded STFT + per-layer attention/conv
  ring-buffer caches), not a request surface. It maps cleanly onto the one-shot `run()`:
  capture a complete utterance → one `run(STTRequest)` → transcript. Live partial hypotheses are
  the deferred companion-N2 token-streaming contract; don't reach for it to ship voice input. At
  ~27× realtime a 12 s clip transcribes in ~0.5 s, so turn-based feels instant anyway.
- **A cache-aware STREAMING model still parity-tests bit-identically.** Because the streamed
  encoder is frame-identical to the offline chunked_limited encoder at the native chunk size, the
  per-chunk goldens (cold caches at chunk 0, WARM caches at chunk 2) matched the Swift port to
  max|Δ|=0.0 on the CPU stream. Capture goldens by MONKEYPATCHING the real Python module (class-
  level `__call__`, not instance — dunders resolve on the class; instance patching silently
  captures nothing), never by reimplementing — zero reimplementation drift.
- **`stt` is genuinely long-run — add it to `CancellationConformance.longRunCapabilities`** as
  part of the capability bump (arbitrarily long audio via the chunk loop). Cadence of record:
  `decode/chunk`, RunProgress-evidenced (the per-chunk `shouldContinue` poll is the same seam
  that reports `.decode`). The core exposes cancellation as a `shouldContinue: () -> Bool` closure
  threaded into the decode loop (not `Task.checkCancellation` inside the core — keeps the core
  engine-free); the wrapper polls `Task.isCancelled` in it and throws `CancellationError` after.
- **Reuse `mlx-audio-dsp` + bake the filterbank.** The slaney 128-mel filterbank is a baked
  safetensors resource generated by the Python oracle (the "bake fixed transforms" rule); the
  STFT/window/framing/power-spectrum/mel-apply primitives already live in `MLXAudioDSP`. Only the
  signal-level (not kaldi per-frame) pre-emphasis and the streaming frame-range mel needed hand-
  porting. Add `MLXFast` + `MLXRandom` as explicit mlx-swift products (not pulled transitively).
- **Illegal-key remap belongs in the loader's `sanitize`, not the module tree.** Reference lists
  with heterogeneous entries (`prompt_kernel.{0,2}` Linear-around-a-ReLU, `joint.joint_net.2`
  after activation/Identity, sparse `pre_encode.conv.{0,2,3,5,6}` among ReLUs) have dotted numeric
  indices that are illegal Swift `Module` keys — remap to `prompt_kernel_0` / `joint_net_2` /
  dense `convs.N` in `sanitize`, keep `@ModuleInfo(key:)` clean. `verify: [.all]` then passes with
  653/653 tensors; relax to `[]` only for the quantized checkpoint (quantize the tree first when
  `.scales` keys are present / config carries `quantization`).
- **STT input decode = AVFoundation.** `STTRequest.audio` is a canonical WAV `Audio` artifact of
  any rate/channels; `AVAudioConverter` → 16 kHz mono Float32 is the package's front door (write
  the bytes to a temp file so `AVAudioFile` can parse the container). The consumer app's mic
  capture produces the same 16 kHz mono WAV via `AVAudioEngine` with `setVoiceProcessingEnabled`
  (AEC keeps the app's own TTS out of the mic — required for barge-in in a talking companion).
- **Consumer-app SPM interlock (the sequencing trap):** a companion app that pins the engine by
  REMOTE version can't reference a new capability (`.stt`) until that engine is pushed + tagged —
  its other packages (LLM/TTS/embed) all transitively resolve the same engine, so the whole graph
  must move together. Develop the kit against a local `git worktree` of the engine capability
  branch; write the app code against the new symbols and verify the path via the kit's own live
  engine-gate CLI (register→prepare→run through `MLXServeEngine`); but the app's SPM-graph repoint
  + Xcode build is BLOCKED on publish — don't do throwaway `pbxproj` surgery against an untagged
  branch. Sequence the graph move with the tag.

## Pre-quantized artifacts as a WeightSourcing DOWNLOAD lever (MLXMageFlow, 2026-07-23)

Klein's configuration notes flagged "pre-quantized repos are a later download-size
optimization" — MLXMageFlow implemented the pattern, and it's worth copying for any
multi-GB DiT wrap:

- **Quantize ONCE offline** (a `--quant-export` CLI mode on the gate executable) into a
  single self-describing safetensors (metadata carries bits/group_size/kept-block
  recipe); publish it next to the port's other artifacts. Consumers
  `quantize(model:filter:)` with the SAME filter reconstructed from metadata, then load —
  **no bf16 peak on the loading machine** (vs Klein's quantize-at-load, which needs the
  full bf16 resident first).
- **Quant-tiered `weightSources`**: the int8/int4 configurations EXCLUDE the bf16
  `transformer/*` glob from the components source and add the quant file to the artifacts
  source — a fresh 16 GB machine downloads 4.3 GB instead of 7.7 GB and never
  materializes weights it can't hold. MAT tests should assert the tiering
  (`testWeightSourcesQuantTiering`).
- Two-source shape (upstream components repo + port artifacts repo) round-trips fine
  through `MaterializationConformance.check` — implement `missingWeightSources` to check
  each source's own satisfaction predicate and return only the missing ones.
- Quant recipe + gate thresholds belong in the mlx-porting skill (common-pitfalls #40-41:
  baseline-relative int8 gating on high-dynamic-range DiTs; trailing-block bf16
  protection; the MLXNN middle-block quantize() container fatal).

## Tool-calling LLM wraps: format-exact prompts + the two stream planes (LFM2.5, 2026-07-24)

The mlx-lfm-llm-swift port (LFM2.5-8B-A1B, `lfm2_moe`, Path-A over stock `LFM2MoEModel`)
surfaced two lessons that apply to ANY tool-calling LLM wrapped through mlx-swift-lm:

1. **swift-jinja's `tojson` breaks tool grounding — render the tools line yourself.**
   Passing tools through the chat template kwarg renders them with swift-jinja's `tojson`:
   COMPACT, ALPHABETICALLY-ordered JSON. Python (`json.dumps` inside HF's jinja) renders
   `", "`/`": "` separators with dict-insertion key order (`name → description →
   parameters`) — the format in the training data. On a 1.5B-active model that drift alone
   flips a temp-0 tool call into an "I don't have access" refusal. Fix: the package renders
   the `List of tools: […]` system line itself in exact `json.dumps` style and byte-pins it
   in a unit test against the Python oracle's rendering. Check this for every tool-calling
   port; small models are far more format-sensitive than benchmark numbers suggest.

2. **The prompt-format-vs-numerics oracle experiment (one run, both answers).** When Swift
   output diverges from the Python reference, feed the SWIFT-rendered prompt (from a
   `--dump-prompt` debug lane: `processor.prepare` → decode tokens) to Python mlx-lm at
   temp 0. Same output as Swift ⇒ numerics are fine and the PROMPT is the defect (this
   port: byte-identical refusal, so the LFM2MoE forward was proven correct in the same
   experiment that isolated the formatting bug). Different output ⇒ real numerics hunt.
   Add `--dump-prompt` to every wrap's gate CLI — it's ~20 lines and turns "model behaves
   weird" into a one-experiment bisect.

3. **`ChatSession.respond` silently DROPS tool calls — drive `streamDetails`.**
   mlx-swift-lm auto-detects `ToolCallFormat` from `config.json`'s `model_type` (any
   `lfm2*` prefix ⇒ `.lfm2`; others get `.json`) and `ToolCallProcessor` EXTRACTS the
   marker-delimited call text out of the token stream as structured `.toolCall` events.
   The string plane (`respond`/`streamResponse`) never sees them (upstream docs:
   `toolDispatch` "required for toolcalls if streaming strings"). A wrapper whose contract
   is canonical TEXT must collect `streamDetails` events and re-serialize `.toolCall`s
   back into the model's emission format (`<|tool_call_start|>[name(arg="v")]<|tool_call_end|>`)
   appended to the text. This silently bites ANY model whose model_type maps to a
   detected format — even when the package never passes template tools.

4. **MS-1/MS-2 store layout (engine ≥0.33): do NOT copy the gemma-era nested pattern.**
   `ModelStore.directory(for:)` is now `<root>/models--<org>--<name>/` (HF cache
   convention) with materialized files either flat under it (engine-executed, contract
   1.24) or in `snapshots/<commit>/` behind `refs/` (hub client). New configurations:
   `missingWeightSources` = explicit-dir probe, then `defaultMissingWeightSources`
   (MS-2 — accepts both layouts); resolution = explicit dir → `snapshotDirectory` →
   flat `directory(for:)`. Since 1.24 the ENGINE materializes pre-`load()`; the package's
   own materialize pass is a blessed defensive re-check, not the primary path. Tests that
   hand-build `<root>/<org>/<name>` are asserting the pre-MS-1 world and will fail against
   ≥0.33 engines (how this was caught).

5. **Always-reasoning models: mode-mapped think-stripping is package work.** LFM2.5 emits
   `<think>…</think>` unconditionally (no template disable; `preserve_thinking` only
   affects history rendering — the template strips prior turns' reasoning itself). The
   package strips the block for default/`.direct`/`.companion` (TTS-safe canonical text;
   split on the LAST `</think>`, mirroring the template's own
   `content.split("</think>")[-1]`), returns raw under `.thinking`, and returns EMPTY for
   a truncation mid-think (never leak reasoning into TTS-bound text). Budget maxTokens for
   reasoning + answer (~512+ for a companion turn).

6. **Measure the resident floor POST-LOAD, never post-warmup — and cross-check the CLI against
   the in-app harness.** The LFM port's first declaration (floor 6.5 GB / activation 0.5 GB) was
   wrong in BOTH directions because its `--mem-bench` ran a warmup generation and called the
   footprint after it "the floor." A post-run floor conflates resident weights with retained run
   intermediates: it over-reads the floor *and* subtracts that inflation back out of
   `peak − floor`, collapsing the activation toward zero. Corrected (floor read immediately
   post-load, `clearCache` first): floor **4.89 GB** — within 0.1 GB of the 4.83 GB on-disk
   weights, i.e. a near-perfect mmap load — and activation **1.31 GB**, ~2.6× what the bad
   measurement claimed. Same trap as the NAFNet/DDColor "floor-not-dropping" finding, and exactly
   what `MLXEngineTestKit.ValidationRun` documents; a hand-rolled CLI bench does not inherit that
   fix, so **make the CLI mirror `ValidationRun` semantics** (floor post-load; post-run retention
   reported as its own number) and validate a port through BOTH the CLI and the in-app
   `ValidationHarness`. They agreed on the floor to 0.13 GB here; the discrepancy is what exposed
   the bug. Corollary: pass a real `clearCache` — **`engine.trimCaches()`**, which the engine
   exposes (with `gpuPoolSnapshot()`) precisely so a consumer gets pool handling without importing
   MLX. `ValidationHarness(clearCache: nil)` makes the floor read ~0.1 GB HIGH and activation
   correspondingly LOW; linking MLX into an app target to avoid that is the WRONG fix.

7. **Report measured usage — `LLMResponse.usage` (contract 1.26.0, engine ≥ 0.35.0).** An `llm`
   package is expected to populate `promptTokens` / `generationTokens` / `promptSeconds` /
   `generateSeconds`. The rule is **measured, never estimated**: take them from what the runtime
   reported, never derive them from the text. `nil` means "this package doesn't report usage" and
   is *not* zero — a consumer must not backfill an estimate.
   - **Freeform path:** the counts already arrive. mlx-swift-lm emits one `GenerateCompletionInfo`
     at end-of-stream as `Generation.info`; a package on `streamDetails` just adds `case .info`.
     Packages still on the string plane (`ChatSession.respond`) never see it and must migrate —
     budget that honestly, especially where a **held** session backs KV-cache reuse.
   - **Structured / constrained-decoding path:** there is no `.info` — the package drives
     `TokenIterator` itself, so it counts and times its own loop. Start the prompt clock *after*
     `processor.prepare(…)`: upstream's `promptTime` likewise excludes templating/tokenization, and
     starting earlier silently charges tokenization to prefill and makes the two paths
     non-comparable.
   - **Free win in the same change:** `GenerateCompletionInfo.stopReason` maps 1:1 onto
     `FinishReason`, so capturing `.info` also retires a hardcoded `.stop` — a `maxTokens`
     truncation finally reports `.length`.
   - **For reasoning models, count the reasoning tokens.** LFM2.5 always emits `<think>…</think>`
     and non-thinking modes strip it from the canonical text. Usage counts what the model
     *generated*, not what was surfaced: decode throughput measures work done. Counting only
     visible text under-reports a reasoning model badly and makes it look slower than a
     non-reasoning model doing strictly less work.
   - **Comparability caveat to document on the field:** `promptTokens` is not comparable across
     packages that differ in KV-cache reuse (a held session prefills only the new suffix).
     `generationTokens` / `generateSeconds` is the axis to quote cross-model.

## Sibling checkpoints — adding a fine-tune as a family, not a mode (Lucida on BiRefNet, 2026-07-25)

Third instance of this shape (FireRed on QwenImageEdit, Klein base-vs-distilled, now Lucida on
BiRefNet), so it's a pattern worth following rather than re-deriving. A fine-tune of an
already-ported base is **a configuration + a manifest, never a port** — but *how* you expose it
matters more than the plumbing.

**Establish "zero port" before writing anything.** Diff the converted key set and the *resolved* arch
config against the base. Lucida: 754 tensors in → 687 out, zero keys or shapes differing vs BOTH
upstream bases, and the resolved config (from the upstream `Config` class, not `config.json` — which
carried no hyperparameters at all) matched the Swift default exactly. That diff is the whole
justification for reusing the core; do it first and it costs minutes.

**Prefer a distinct PackageID over a new `mode`.** `Mode` is an open string tag, so a third tier
*compiles* — but the **manifest is per-registration**, and that is what you actually need separated:

- **license posture.** A caveat on the new checkpoint (Lucida's training data mixes research-only
  datasets) would otherwise gate *every* consumer of the shared surface. Here that would have hit
  `EngineMatteProvider` and the TRELLIS.2 multi-view front door for a checkpoint they never ask for.
- **provenance.** `Provenance.sourceRepo` is the engine's download/marker key — a shared manifest
  credits the wrong repo in the storage panel.
- **footprint.** A single-tier fine-tune usually has a *smaller* resident floor than the multi-tier
  base package (one graph, not two).

Mechanics: `PackageRegistration(manifest:makePackage:)` takes an explicit manifest, so the sibling
reuses the existing package *class* under its own identity — no new conformer type.

**Then close the cross-family hole this opens.** Once one class serves several checkpoint families,
`mode` becomes able to select *another family's weights*. Fix it structurally, not by convention:
route every (variant, mode) → (repo, override, resolution) through **one** method on the
configuration, so repo and resolution are always decided together. A family with a single checkpoint
should **ignore** modes it doesn't advertise rather than falling through to a sibling's repo — and
have the sibling's `makePackage` **refuse a mismatched variant**, so the manifest's identity can
never be published over another family's weights. Both are one-line guards with a regression test
each; without them the failure is silent and looks like "the model is worse than the benchmark says."

**Adding a field to a shipped `PackageConfiguration` needs hand-written `Codable`.**
`PackageConfiguration: Codable`, so a synthesized decoder makes a new `variant` field **required**
and breaks every persisted registration. Write `init(from:)` with `decodeIfPresent` + defaults, and
test-decode a pre-field JSON payload.

**Footprint by transfer, but measured.** If the sibling runs the same graph at the same resolution as
an existing tier, that tier's in-app `phys_footprint` numbers apply — but *demonstrate* it rather than
asserting it: the CLI smoke reported byte-identical floor/peak for both families, and the peak matched
the exact MLX figure whose in-app equivalent was already on record. Then declare resident ABOVE the
measured floor (an initial draft under-declared it, which makes the governor under-reserve).

**Retrofits ride the sibling change well.** This package was the fleet's last non-MAT image package;
adding a checkpoint is exactly when you're touching the weight-resolution path anyway, so the
WeightSourcing/MAT retrofit cost almost nothing extra. One honest consequence to surface rather than
hide: declaring *all* servable tiers means a fresh install materializes tiers it may never use
(here +440 MB for the 2048 checkpoint, previously lazy). That is correct once the package no longer
downloads for itself — an undeclared tier just fails on a fresh store — and it is the concrete
argument for mode → PackageID (P1b), which would restore laziness by construction.

## Replacing an unshippable HOST dependency with Apple-native signals (Mage-VL codec path, 2026-07-27)

Some upstream pipelines depend on a host binary or native ext we won't ship — **ffmpeg** (license +
tens of MB), a CUDA extension, a bespoke C++ codec. On Apple Silicon a surprising amount of that is
already in the SDK, or already in our own fleet. Before writing off the capability — or vendoring
the binary — work this list.

**First, check the registry, not just `mlx-swift-lm`.** The Path A/B question ("does the runtime load
this architecture?") is about the *backbone*. It misses the other half: **does the fleet already
provide the auxiliary signal this pipeline needs?** Mage-VL's codec path needed dense motion; we had
already ported, published and validated **SEA-RAFT** under `opticalFlow`. Grep
`mlx-engine-swift/docs/model-registry.md` by *capability* before planning any host dependency —
the answer to "port something complementary" is sometimes "you already did."

**Then check what the SDK gives you free.** Verify against the actual headers
(`xcrun --sdk macosx --show-sdk-path`), not from memory — availability moves.

| Upstream needs | Apple-native source | Notes |
|---|---|---|
| Dense motion / optical flow | `VTOpticalFlowConfiguration` + `VTFrameProcessor` | `API_AVAILABLE(macos(15.4), ios(26.0))`; `qualityPrioritization` normal/quality; submission mode `.sequential` for an ordered walk. **Zero weights, hardware-accelerated, ships with the OS.** |
| I-frame vs P-frame | `kCMSampleAttachmentKey_NotSync` on the sample buffer | No decode required |
| Per-frame encoded byte cost | `CMSampleBufferGetTotalSampleSize`; `AVSampleCursor` walks samples **without decoding** | The true bitcost, exactly — just not per-block |
| Motion-compensated residual | Backward-warp *t−1* by the flow, subtract | The warp an `opticalFlow` package already exists to do |

**Consume a sibling package by vendoring its library product, not by engine composition.** One
package invoking another *capability* through the engine is an unproven pattern and buys nothing.
`mlx-sea-raft-swift` exposes `SEARAFTMLX` as a public `.library` with `SEARAFT(.s)` +
`loadWeights(...)` documented as supported — take that. (Check the donor's **C14** row first: a
public library path that bypasses the package's `load()` historically also bypassed
`train(false)`, which is exactly why C14 moved those calls to the loader choke point.)

**Bake-off before you commit.** When both a ported model and an OS API can supply the signal, measure
them against each other on the *consuming* metric, not on the signal's own accuracy. If the consumer
percentile-normalizes and sums the signal into coarse blocks, it is extremely forgiving — a
0.116 px EPE model may beat the free OS path by nothing measurable, in which case the zero-weight
path wins on residency, packaging and cold start simultaneously.

**Two doctrines carry over from `mlx-porting`** and belong in the plan, not discovered late:
substituting a signal the model was **trained** on is a distribution shift (gate on agreement with
the original, not on quality), and the unshippable dependency is still fine as a **dev-time oracle**
to bake those fixtures — an oracle is not a dependency. See `mlx-porting` pitfalls #43 / #43b.

## Product validation on public corpora — five things that changed a decision (2026-07-27)

Three licence-clean public corpora (NIND, RealBlur, DPDD) were run against shipped packages in one
day. Two shipped defaults changed as a result. The transferable parts:

**1. Judge a shipped default on the TAIL, not the mean.** Our shipped deblur backer scored **−0.02 dB**
against doing nothing, which reads as "harmless, just useless". Dumping per-image values told a
different story: *median* **+0.18** (it helps slightly on most images), std dev 1.96, **41% of images
hurt, 18% by more than 1 dB, worst case −18.70 dB** — gains and losses cancelling almost exactly
(+180 / −187 dB over 300 images). The competing model was +1.29 with a −0.69 dB worst case and 0%
hurt by >1 dB. **"Neutral on average" and "unpredictable" are opposite product verdicts and the mean
cannot distinguish them.** Always dump per-image results and report win-rate + worst case; a default
must be safe on the tail, not good on the median.

**2. A published cross-dataset number may be a differently-TRAINED checkpoint.** The port queue row
justified itself partly on *"RealBlur-J 32.62 — best cross-dataset of anything permissive"*. That
figure is the **RealBlur-trained** checkpoint; we hold the GoPro-trained one, which scores −0.02 on
the same test set. The row looked safe *because* a number from a checkpoint we do not ship was read
as evidence about the one we do. **Before treating a benchmark cell as evidence, confirm which
checkpoint produced it.**

**3. Benchmark rank can INVERT on real data — and the paper's stated mechanism may be wrong.** A
published study blamed *"GoPro training"* for real-blur failures. Our second arm was also
GoPro-trained and transferred fine (+1.29), so the mechanism is **model-specific**, not
dataset-specific. Meanwhile the models' GoPro ranking (34.21 > 32.92) reversed on real blur
(−0.02 < +1.29). Reproducing a paper's *conclusion* is not the same as reproducing its *explanation*,
and only the explanation tells you what to do next.

**4. Build in a control with a known correct answer.** On the full official DPDD split our harness
returned **25.98 dB — exactly the published figure**. That validated the *harness*, and retroactively
strengthened every number the same PSNR/pairing code produced on the other corpora. When a pipeline
agrees with a published value to two decimals, the pipeline stops being the thing to doubt. Pick at
least one arm where you know what the answer must be.

**5. Use the dataset's OWN evaluation protocol, and put the baseline through it too.** RealBlur pairs
come from a two-camera beam-splitter rig and are **not** pixel-aligned; the authors' script does
intensity matching + ECC homography alignment before PSNR. Skipping it understates every arm.
Critically, the do-nothing baseline must go through the *identical* transform — otherwise the
comparison is silently rigged toward the models. (Contrast NIND, where the premise was a locked-off
camera and the right move was to *verify* pixel alignment before measuring. Check which world you are
in; do not assume either.)

**Corollary on premises:** every one of these runs asserted its pairing before trusting a number —
NIND's alignment and brightness match, DPDD's positional pairing (source/target are *adjacent
captures with different filenames*; correct pairing scored 20.4 dB vs 10.1 dB shuffled), NAFNet's key
contract. A measurement harness should fail loudly on a bad premise, not quietly produce a plausible
table.

## A blocked comparison arm: state the REQUIREMENT, not the artefact (NIND/NAFNet, 2026-07-27)

The NIND real-noise ranking shipped with its most important arm missing, recorded as blocked on:

> *"NAFNet was not in the run — no PyTorch NAFNet in our oracle set."*

That sentence is a **category error**, and it survived review twice because it sounds like a fact.
The harness happened to be built from PyTorch references, so "no PyTorch checkpoint" got written down
as the blocker. But the actual requirement is **a verified implementation of the model** — and we held
two (an MLX-Python port and an MLX-Swift port, weights already on `mlx-community`). Adding the arm
took ~20 minutes and **inverted the headline**: from *"strongest of the three we hold"* to *"replaces
the incumbent"* (the incumbent scored **below its own input** at the lowest ISO).

**The rule: when you declare an arm blocked, write the requirement, then ask what satisfies it.**
"No PyTorch checkpoint" names an artefact you searched for; "no verified implementation" names the
requirement, and only the second is actually checkable. The first quietly scopes the search to
whatever the harness already used.

**Mixed-runtime arms are fine, and cheaper than they look.** Nothing required every arm to share a
framework — the comparison is between *models*. What it does require is that the odd arm out earns
the same trust as the rest: because that arm ran in MLX while the others ran in torch, its key
contract is **asserted in the harness** (`assert model_keys == ckpt_keys`), not assumed. A silently
partial load would have handicapped exactly the model the table was trying to unseat — the one way
the result could have been right for a bogus reason. **Assert the contract on the arm you most want
to lose.**

## Probe weight AVAILABILITY before planning a row (image-restoration batch, 2026-07-27)

**Two of fourteen queued ports turned out to have no obtainable weights**, and one of them cost a
wasted Stage 0 because a licence ✅ was read as an availability ✅. They are different claims.

- **P1 (Lai TransformNet)** — the only host 404s; Wayback never archived the binary, GitHub filename
  search returns `total_count: 0`, no HF mirror, upstream issues unanswered since 2024.
- **P6 (ESTRNN)** — 0 GitHub releases, 0 in-repo checkpoints, no HF mirror, no drive link in the README.

**Cheap sweep, run it first:**

```python
# GitHub: in-repo checkpoints + release assets
api(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")   # filter .pth/.ckpt/.safetensors
api(f"https://api.github.com/repos/{repo}/releases")                     # assets[].name/size/download_count
# HF: any mirror at all
api(f"https://huggingface.co/api/models?search={term}")
```

Rank what you find: **committed in-repo** > **first-party release** > **first-party HF org** >
third-party mirror > a drive link in a README.

### 🔴 A third-party mirror's licence tag is not evidence about the weights it mirrors

The queue recorded DRUNet and SCUNet as "third-party `deepinv` mirrors only — check the licence."
`deepinv/drunet`'s card says `bsd-3-clause`; that is the **DeepInverse library's** licence blanketed
across their HF org and says nothing about upstream, which is **MIT**. `deepinv/scunet` carries no
licence at all. In fact **the original author publishes every weight first-party** in the
[`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0) under MIT — both rows were
clean all along. **Look for the author's own distribution point before accepting a mirror**; authors
often park a whole model zoo in one sibling repo's releases.

## Footprints: the `--bench` MLX-peak number is not the admission basis

A gate-style `--bench` that calls the core model directly and reads `MLX.GPU.snapshot().peakMemory`
**bypasses register/prepare/the governor AND reads the wrong metric.** MLX-pool peak under-reads
process `phys_footprint` by ~2.7× (the BiRefNet re-baseline). Declaring from it under-declares, and
**under-declaring falsely admits on tight Macs** — the failure mode that matters.

**Measured proof from this batch:** CIDNet's activation, extrapolated from sibling ratios, was
declared **3.0 GB**; measured through the real engine it is **5.84 GB** — under by ~2×. Restormer's
`--bench`-derived 4.5 GB was also under the true 4.76 GB. Only FFTformer was over.
SCUNet is the widest miss yet: `--bench` read **1.33 GB**, the harness **4.38 GB** — **3.3× under**.
**Across five ports the sign never flipped in a way you could rely on**, so treat `--bench` as a tool for
*comparing tile sizes to each other* and never as a source for a declared number.

**The fix is a per-package `*-validate` executable**, not app wiring:

```swift
let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking) // match production
let result = try await ValidationHarness.run(
    engine: engine, registration: X.registration, configuration: cfg,
    capability: .imageRestore, request: req,
    isolate: true, clearCache: { MLX.GPU.clearCache() }, heartbeatLabel: "x")
print(result.run.splitLogLine("x"))   // [x] SPLIT floor= peak= act= retain= engine= reserve=
```

`MLXEngineTestKit.ValidationHarness` is the **same code path and same metric** the archived
MLXEngineImage app used — 150 ms `phys_footprint` sampling, floor read **post-load / pre-run**.
`package-efficiency.md` explicitly blesses an `xcodebuild`-or-`swift run` executable reading
`phys_footprint`. Making it a package target means the number is **reproducible** rather than a
one-off, and it needs no Xcode project.

Two caveats worth writing into the manifest comment: a CLI process carries no AppKit/Metal-view
overhead, so absolute floor/peak sit a few hundred MB below a GUI app's (conservative in the *wrong*
direction — declare with margin); and `retain=` above ~0.3 GB means the live model holds
intermediates, which belongs in the transient, not residency.

**Measure the model path, not the bypass.** CIDNet's validate feeds a deliberately *dark* image,
because a mid-grey one trips its luma gate and would report a footprint with no model in it.

## Tiling: when the model forces it, and the alignment nobody expects

Two of four ports in this batch **could not run a 1080p frame full-frame**: FFTformer peaked at
**39.55 GB MLX / 109 GB phys**, Restormer at **15.50 / 48.02 GB**. Both are small models (16.6 M and
26.1 M params) — the cost is that level 1 runs at full resolution with a wide channel expansion
(FFTformer 6×, plus a patch-decomposition copy). Predict it before measuring: `pixels × channels × 4
bytes × live tensors` got within ~10% both times.

**🔑 Tile geometry must align to the model's internal grid stride.** FFTformer's overlap sweep showed
no receptive-field trend at all — instead overlaps 16/48/112 scored ~20.6 dB against ~26 dB at
0/32/64/96. The error tracked `overlap % 32`, because FFT patches are measured from the **tile
origin** and a level-3 patch spans 32 full-res pixels. Restormer has the same class of bug at stride
**8** (three `pixelUnshuffle(2)` stages). Round tile *and* overlap down to the stride.

**Judge a tiler on SEAM VISIBILITY, not PSNR-vs-full-frame** — they disagree. Restormer's overlap 0
had the *best* PSNR (38.70 dB) while leaving a measurable seam (boundary gradient 1.31× interior);
every aligned overlap ≥ 8 measured clean (1.09–1.20×). Full-frame is unattainable at production sizes
anyway, so agreement with it is the wrong objective; boundary continuity is the right one. Metric:
mean |gradient| at columns on a tile seam vs everywhere else.

**Consequence for the manifest:** an internally-tiled package's peak is **one-tile-sized and flat in
resolution** (Restormer: 2.58 / 2.59 / 2.62 GB at 512² / 1024² / 1080p) — 4K runs *more* tiles, not
bigger ones. An untiled package's activation is **linear** (DRUNet 6.56 GB at 1080p ⇒ ~4× at 4K).
Say which one you are in the footprint comment; it is the difference between "4K is free" and "4K is
4× and untested."

### How badly a model tiles is predicted by its attention LOCALITY (SCUNet vs Restormer, 2026-07-27)

Same tiler, same seam metric, opposite verdicts — and the architecture says which you will get
before you measure:

| model | attention | overlap 0 | best overlap | seam ratio at best |
|---|---|---|---|---|
| Restormer | channel attention, reduces over the **whole feature map** | 1.31× (visible) | 32 | 1.09–1.20× |
| SCUNet | 8-px **windows**, shifted; strictly local | 1.08× | 64 | **1.00×** |

SCUNet's 1.00× means a tile boundary is *statistically indistinguishable from ordinary image
content* — window attention tiles almost for free, because no computation ever sees beyond a window
plus a few conv taps. Anything that reduces globally (channel attention, global pooling, an FFT over
the whole plane) changes its statistics when you crop, so its tiles are genuinely approximate.

**Practical read:** for a local-attention model, pick the overlap for memory and stop worrying; for
a global one, the overlap sweep is load-bearing and PSNR-vs-full-frame will lie to you.

**The alignment stride for a window-attention model is the WINDOW GRID, not the downsample factor.**
SCUNet has three stride-2 stages (⇒ 8) but pads to **64**, because the deepest blocks want a full
8-px window at 1/8 scale. The forward pass lays the window grid out from the **tile's own origin**,
so a tile origin off the 64 grid shifts the window phase relative to its neighbour — a seam
feathering cannot remove. Align tile *and* overlap; step is then aligned automatically.

### Tiling a GENERATIVE model: which per-tile globals matter, measured (SeedVR2, V10-fix 2026-07-30)

A one-step diffusion refiner tiled at 1:1 (SeedVR2-3B, 256/32) has two operations that are *global
by definition* being applied per tile — and a measured hierarchy of how much each costs. Receipts:
`mlxengine-todo/probes/v10fix_*.out`.

- **Global-statistics colour transfer per tile is the big, cheap win — and it is a TEMPORAL bug
  too, not just a spatial one.** `labTransfer` is histMatch (global stats); per tile, every tile
  lands in its own colour space → visible tile grid + chroma speckle in flats. On video the feared
  failure INVERTED under A/B: the per-tile match was the *unstable* arm (max frame-pair |Δa*| 1.62
  vs a Lanczos content floor of 0.269 — a tile's histogram churns as content crosses its border and
  the whole tile jumps), while a per-frame global match measured BELOW the content floor (0.235).
  Match once, against the whole pre-upscaled base, after assembly — stills and video both.
- **The noise field is a closed chain: three constructions, no measurable difference past the
  first.** Identical-per-tile noise (one seed, per-tile-sized draw) → periodic texture locked to
  the tile grid: fix that. But per-tile-seed decorrelation vs the *correct* construction (ONE field
  over the whole image latent, sliced per tile via a region-aware tiler closure) measured as pure
  noise-realization jitter (seam comb 1885→1875, flat-MAE 4.19→4.20). Ship the correct construction
  because it is correct and costs ~4 MB — but do not expect quality from it, and do not revisit.
- **The residual seam error in a generative tiler is per-call variance** — each tile reconstructs
  plausible-but-different content from different context, and feathering blends the disagreement
  into mottled overlap bands. No noise or colour construction fixes that; the levers left are
  extent (the model's resolution envelope) and per-call determinism-with-shared-context. Geometry
  sweeps trade it against MAE *backwards* (bigger tiles: seam energy ↓, MAE ↑, memory ↑).
- **Method note that made the above trustworthy:** every arm — including dumps from earlier
  sessions — was recomputed with ONE metrics script before comparing; and the tiled path's
  single-pass control rows had to reproduce byte-identically before the tiled row was read at all.
  A fidelity metric (SSIMULACRA2) rated an obvious smear→legible repair as 4.7 points on this path:
  dump the PNGs and look, every time.

## Growing a capability vs adding one (two worked decisions, 2026-07-27)

Both came up in one batch and resolved opposite ways. The discriminator is **request shape**, not
subject matter.

**New capability — `imageRelight` (HVI-CIDNet, contract 1.29.0).** Justified because the behaviour
differs, not just the model: every checkpoint drives output toward a target mean luma *regardless of
input*, so on an already-correct exposure it **degrades** the image (measured 23.37 / 20.99 / 16.10 dB
across three checkpoints). A planner told to "restore" must not silently re-expose. It also needed a
**bypass**, which restoration has no concept of.

**Grow the existing capability — `strength` on `imageRestore` (DRUNet, contract 1.30.0).** The
request/response shape was otherwise identical, so a separate capability would have fragmented a
surface three packages already shared. Added `ImageRestoreRequest.strength: Float?` and
`ImageRestoreResponse.appliedStrength: Float?`, both defaulted so existing backers are untouched.

Two design points worth reusing:

- **Report what you actually did, typed.** `appliedStrength == nil` distinguishes "this backer has no
  dial" from "the dial did nothing" — a UI greys the control, a planner routes elsewhere. Same
  reasoning gave `ImageRelightResponse.bypassed`. Do not bury this in `metaData`.
- **Gate the parameter off surfaces that ignore it**: `descriptor(supportsStrength:)`, so a planner is
  never offered a knob that does nothing.

## Publishing hygiene

- **`.gitignore` the HF download cache with a GLOB.** Round-trip verification (download the published
  artifact, re-run the gates) leaves a cache under `oracle/`. Four repos in this batch committed it —
  one with a 124 MB blob GitHub rejected outright, three that slipped through and bloated clones
  (restormer 203 MB → 15 MB after cleanup). Ignore `oracle/hf*/`, not the one directory name your
  script happened to use.
- **Verify by fresh clone, not by local state.** After a history rewrite, `refs/remotes/origin/*` and
  **pushed tags** keep old objects reachable. Deleting and re-creating the tag from the rewritten HEAD
  was what actually shrank the clones. Then `git clone` into a temp dir and check `du -sh` plus the
  largest blob — that is what a consumer gets.
- **Round-trip every publish.** Download the artifact fresh and re-run S0 + the e2e gate against it.
  Cheap, and it catches an upload that silently differs from what was gated.

## Platform-mismatched CPU code accumulates where you DIVERGE from the oracle without a parity reason (LTX FrameCodec, 2026-08-04)

The mirror-the-oracle discipline is a *correctness* rule, and it has a performance corollary nobody
had stated: **the places you write Swift with no oracle counterpart are exactly where CPU-shaped
code creeps in unexamined** — because no parity gate ever looks there, and C-family instinct
produces per-element loops that are natural on CPU and wrong on unified memory with an idle GPU.

- **The receipt:** LTX's `FrameCodec` repacked RGB→BGRA with a per-pixel scalar CPU loop — one
  pixel at a time, per frame, after a per-frame device→host copy: **~43.6 M scalar iterations at
  704×512×121f.** The Python oracle has no such loop (one bulk `memoryview` copy); this was our own
  Swift. Doing the channel reverse + alpha on-device (`MLX.take` + `concatenated`) and
  bulk-`memcpy`ing: **3.5 s → 0.2 s (~15×), byte-identical** (`--frame-codec-gate` recomputes the
  old path and asserts equality — the right gate for a pure refactor; an encode round-trip is lossy
  and proves less).
- **The audit heuristic that found it (and four siblings):** diff your port against the oracle
  specifically for *serial constructs with no oracle counterpart* — per-pixel image ingest loops,
  host-side index-table builders, scalar filterbank construction. Where you mirrored the oracle,
  serialization is usually deliberate (watchdog/eval discipline, bit-exactness); where you
  diverged, it is usually accident.
- **Second half of the lesson: measure before executing the PLANNED fix.** The backlog had this
  scoped as IOSurface-backed `VTCompressionSession` (days of work) under "zero-copy handoff".
  Measuring first showed the cost was the scalar loop, not the copy — the ~2 h on-device fix
  captured ~15× and the residual (121 device→host copies) is worth ≤1.7 ms/frame, correctly left
  for the streaming-decode work it belongs to. A stale backlog entry can encode the wrong
  diagnosis; the receipt, not the plan, decides what gets built.
- Cross-refs: the distilled-checkpoint/CFG premise trap and the flash-SDPA memory-lever premise
  trap live in `mlx-porting` `common-pitfalls.md` #55/#56 — both are reading-time checks that
  killed weeks of misdirected work on the same project.

## Two Swift-6 seam lessons from the streamed decode→mux port (LTX #8, 2026-08-05)

- **A class that synchronous sink closures must call cannot be actor-bound.** `MP4StreamWriter`
  started `@InferenceActor`; the pipeline drives it through plain `(MLXArray) throws -> Void`
  sink closures, and actor-isolated methods are uncallable from those even though they *execute*
  on the same actor — the closure's TYPE carries no isolation. Fleet idiom that resolves it:
  plain `final class` + `@unchecked Sendable`, documented single-context use.
- **Do not thread async through a synchronous GPU loop just because the consumer has awaits.**
  Making the VAE chunk sink `async` cascaded `sending`/region-isolation errors through every hop
  for zero benefit — the decode loop is seconds of blocking GPU work per chunk, and encoder
  back-pressure at chunk cadence is effectively never hit. A synchronous `appendSync` with a
  bounded usleep poll (same 90 s loud-error guard) matches the codebase's sync-heavy reality and
  dissolved the whole knot. Generalizes: match the async-ness of a seam to the LOOP that drives
  it, not to the callee's internals.

## AVAssetWriter/VideoToolbox H.264: deterministic per input+batching, but rate control responds to append BATCHING — and MP4 md5 never reproduces (LTX mux-bench, 2026-08-05)

Measured while gating a materialized-vs-streamed decode→mux seam (`ltx-2-mlx-swift`
`--vae-mux-bench`, 1024×576 noise content — codec worst case):

- **Whole-file MP4 md5 differs between IDENTICAL invocations of the same code** — the container
  embeds creation timestamps. Never use file md5 as an output fingerprint for AVAssetWriter
  products; hash the pixels you FEED the writer instead (per-frame, in append order).
- **The encoder is bit-deterministic for identical input + identical append pattern**: two runs
  of the same lane decode to max|Δ|=0.
- **But rate control responds to append batching**: one big `appendSync` of all frames vs
  per-chunk appends (with GPU decode gaps between) produced decoded outputs at SSIM 0.96 /
  33 dB on noise — with per-frame float digests of the encoder input PROVING the pixels were
  bit-identical. Same pixels in, different QP schedule out. Deterministic per pattern, divergent
  across patterns; software encoder behaves the same.
- Consequence: two mux lanes can only be equivalence-gated at the **encoder input** (pixel
  digest), never at the MP4 or its decoded frames. Quality deltas from batching are codec-normal
  variation, largest on noise-like content.

## A SYMLINKED repo dir in the model store reads as EMPTY — probe and loader both fail, differently (lfm-embedding, 2026-08-05)

Staging a dev store by symlinking an existing download into the MS-1 name
(`ln -s mlx-community/<name> models--mlx-community--<name>`) fails twice, with unrelated-looking
symptoms, because `FileManager.enumerator(at:)` does **not descend a symlinked root** (it yields
nothing; `Data(contentsOf:)`/`contentsOfDirectory` DO resolve symlinks, which is what makes the
failure partial and confusing):

- **MS-2 probe** (`WeightSourceProbe.relativeFiles`): sees zero files → source reads *missing* →
  a spurious full re-download (which lands *through* the symlink into the target — harmless but
  masks the cause: afterwards the store "has" the weights and the next symptom still fires).
- **`loadWeights`** (MLXEmbedders/MLXLMCommon): config.json loads fine (direct read resolves the
  link, so the model BUILDS), then the weights dict comes up empty and `update(parameters:)`
  throws `MLXNN.UpdateError.keyNotFound` on a **random first key** (dict order) — looks like a
  key-remap/sanitize bug, is actually "no tensors were enumerated".

The engine's materializer always creates real directories, so production never hits this — it is
a dev-workflow trap only. Stage a store layout from an existing download with an APFS clone
instead: `cp -Rc <nested-dir> models--<org>--<name>` (instant, CoW), and delete any
`.cache/huggingface` bookkeeping the hf CLI left inside. Diagnostic tell: keyNotFound whose
missing key EXISTS in the safetensors header + a store path involving a symlink ⇒ this, not a
sanitize bug.

## A silent backend fallback that changes RESOURCE CLASS is a leak/OOM factory — external case study (LTX-Desktop MPS, 2026-08-06)

Vendor case (their own internal doc, torch-MPS stack — not ours): the `mps-sdpa` attention shim's
zero-copy backend silently failed to build (ninja binary not on PATH — the Python *package* was
installed) and fell back to a pyobjc backend that allocates a fresh MTLBuffer per attention call
→ unbounded, sequence-length-dependent driver-memory growth → a 48 GB Mac couldn't finish 121
frames, and it presented as a backend crash, not as a fallback. Rules our stack already encodes,
now with external validation:

- **A fallback may be output-invisible, but it must never be resource-class-invisible.** If the
  fallback path has different memory/time behavior, it must announce itself (our streaming gate
  logs its resident fallback; the pruna decoder env fallback prints a note — "check the log
  before trusting an A/B").
- **Runtime-JIT native extensions are a deployment trap on macOS** (toolchain, PATH, codesign,
  hardened runtime). MLX-Swift sidesteps the entire class — but the same trap lives in any
  "compiles a Metal kernel on first use" dependency; know which of your deps JIT.
- Leak triage signal worth stealing: **allocator-level vs framework-level growth** (their
  `driver_allocated_memory` climbed while live-tensor memory stayed flat → below-Python
  retention). Our equivalent split is MLX active vs cache vs phys footprint — the bench already
  records all three per run, and the post-drop floor per block is the leak detector.
