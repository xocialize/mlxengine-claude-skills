# Topic 10 — Field issue log (consumer-side catches & resolutions)

Running log of issues real consuming apps hit against the engine/package fleet, with root cause
and resolution. **Append a row per catch, in the same change that fixes it** — an unlogged catch
gets re-hit by the next app. Status: RESOLVED (fix landed, pattern documented) · OPEN (tracked,
workaround noted) · ENGINE (filed as an engine/package gap, not an app bug).

Format per issue: ID `<APP>-<n>` · date · symptom → root cause → resolution/pattern.

---

## MSS — ModelSheet Studio (~/Development/ModelSheetStudio)

### MSS-1 · 2026-07-09 · RESOLVED — capability overlap rots default routing
**Symptom:** every `imageEdit` cell failed `unreadableSnapshot("mlx-community/Z-Image-Turbo-4bit")`
though klein was the intended editor and its weights were staged.
**Root cause:** packages GAIN surfaces across versions — z-image-swift v0.2.0 added an `imageEdit`
(img2img) surface, so "register Z-Image after klein" silently made Z-Image the default backer for
BOTH `textToImage` and `imageEdit` (last-registration-wins). Registration-order tricks are
load-bearing on the *current pin's* surface list and break when a pin moves.
**Resolution:** route by explicit `PackageID` everywhere in a multi-package app — capture the ID
from every `register(...)`; pass it to `run`/`prepare`/`needsDownload`; `ModelStateView` takes the
ID's `rawValue`. Reserve capability-default routing for single-backer capabilities. (Also appended
to [routing.md](routing.md).)

### MSS-2 · 2026-07-09 · RESOLVED — quant tier selects a quant-SUFFIXED repo; staged weights must match
**Symptom:** with `Z-Image-Turbo-bf16` cloned into the store, registering `.turbo(quant: .int4)`
still failed `unreadableSnapshot(...-4bit)`.
**Root cause:** some packages resolve `effectiveRepo` per quant (z-image: `-bf16`/`-8bit`/`-4bit`
suffix); others quantize AT LOAD from one bf16 snapshot (klein). The two schemes look identical
from the config call site.
**Resolution/pattern:** before staging local weights or picking a quant, read the package's
`WeightSourcing` extension (`effectiveRepo`/`weightSources`) to learn which scheme it uses; match
the registered quant to the snapshot actually in the store. When hand-staging, `cp -Rc` (APFS
clone) into `<store>/<org>/<repo-name>` — instant, no extra disk.

### MSS-3 · 2026-07-09 · RESOLVED (fleet) — pre-MAT packages don't auto-materialize
**Symptom:** fresh-install stages stuck "needs download" forever (realesrgan) or requiring a
hand-set `snapshotDirectory` (qwen25vl).
**Root cause:** packages predating the v0.19.0 materialization gate: qwen25vl had no
`WeightSourcing`; realesrgan wasn't even `ModelStorable` — AND its weights were **bundled in the
package all along** (~2 MB each), so its "needs download" was the store-marker heuristic
misreading bundled weights.
**Resolution:** fleet retrofit same day — qwen25vl v0.2.0 (full WeightSourcing + snapshot-local
tokenizer: it had been silently network-fetching the tokenizer outside the store at `load()`),
realesrgan v0.4.1 (`BundledWeightSourcing`, engine ≥0.24.0/contract 1.17). **App action when
hitting a pre-MAT package: file the retrofit (this playbook), repin — don't hand-stage around it.**
Check for vendored/bundled weights before assuming a download is needed.

### MSS-4 · 2026-07-09 · ENGINE (open) — tier-check API lives in TestKit
**Symptom:** the runtime FeatureAvailability map (simulated 16/32/64/128 GB grid) needs
`AdmissibilityTiers.check` + `AdmissibilityTierView`, which live in `MLXEngineTestKit` — a shipping
app now links a test kit.
**Status:** works fine; filed as promotion candidate (tier check → MLXServeCore proper) = ModelSheet
Studio PRD gap G3. Revisit when a second app needs it or at MSS ship time.

### MSS-5 · 2026-07-09 · RESOLVED — Settings folder Apply has no engine hook
**Symptom:** user picks a new models folder in `EngineSettingsView` → nothing re-binds; packages
keep the old (or default-cache) root.
**Root cause:** `useModelStore` only stamps packages registered AFTER it; the provided settings UI
mutates `ModelStorageModel` but can't know the app's registration list.
**Resolution/pattern:** observe the storage model and re-run the golden path:
`.onChange(of: storage.resolvedModelsDirectory) { Task { await rebind() } }` where rebind =
`useModelStore` → re-`register` everything → re-hand PackageIDs to consumers → re-survey. Guard it
on app-ready phase so it can't race bootstrap.

### MSS-6 · 2026-07-09 · RESOLVED — distilled klein under-follows framing prompts
**Symptom:** "head-and-shoulders portrait" expression cells came out ¾-body/chibi.
**Root cause:** guidance-distilled 4-step tier trades prompt/framing adherence for speed —
model-tier behavior, not an app bug.
**Resolution/pattern:** register BOTH klein tiers (distilled + `Klein4BBaseT2IPackage` base CFG,
distinct PackageIDs, co-resident under the governor) and route per cell type: fast distilled for
turnarounds, base for framing-sensitive cells. Plus belt-and-suspenders explicit crop prompts
("cropped at the shoulders, no body below the chest").
**Addendum (same day) — rotation collapse:** the distilled tier also under-rotates ("three-quarter
view" comes out ≈ front) when the front view rides as a reference — it anchors to the nearest ref.
Pattern: for interpolated angles, condition on BOTH endpoints (multi-ref: reference + front +
side, generated in that order) + explicit rotation language ("ROTATED 45 degrees, NOT facing the
viewer"). Generation order ≠ display order. Escalation if still collapsed: route that cell via the
base CFG tier.

### MSS-7 · 2026-07-09 · RESOLVED — Trellis2Kit is pre-MAT
**Symptom:** fresh install: 3D turnaround route silently degrades to diffusion forever — no
first-run download of the ~15 GB `xocialize/trellis2-mlx` snapshot.
**Root cause:** Trellis2Kit (mlx-trellis2-swift ≤0.4.0) has no `WeightSourcing` conformance.
**Resolution (2026-07-09, mlx-trellis2-swift 0.5.0):** full WeightSourcing retrofit — single
"snapshot" source, strict 10-file missing-probe (a lenient probe would silently drop textures /
cascade tiers on a half-materialized snapshot), `load()` auto-materializes via HubClient with
`.downloading` progress, MAT-1..5 green. Pin the app to ≥0.5.0. **HF_TOKEN caveat REMOVED (2026-07-09, same day):** the
repo was UNGATED after restructuring as a pipeline artifact (card frames the converted DINOv3
conditioner as one integral component, redirects DINOv3-seekers to Meta's official repo, and
moves terms-respect to license-travels-with-the-artifact + the engine's C7 gate + the
"Built with DINOv3" app caption — constraints per the owner's `trellis2-weights-distribution-plan`
memory). Token-less fresh-install auto-materialization verified (anonymous HTTP 200).

### MSS-8 · 2026-07-09 · RESOLVED (near-miss, package-side lesson) — `exact:` mlx-swift pins break workspace resolution
**Symptom (near-miss):** mlx-trellis2-swift 0.3.0 pinned mlx-swift `exact: 0.31.4` while the
workspace-local engine wants `from: 0.31.5` — SPM resolution would have failed for ANY app
combining them. 0.4.0 (relaxed pin) happened to exist and the upToNextMajor requirement resolved
to it.
**Pattern:** packages must never `exact:`-pin shared foundations (mlx-swift, mlx-swift-lm) —
one exact pin makes a package un-combinable with the rest of the fleet in a single app. App-side:
if resolution fails on an mlx-swift conflict, check for an `exact:` pin in the newest dependency
before touching your own pins. (Package-side rule belongs to `mlx-swift-integration`.)

### MSS-9 · 2026-07-10 · RESOLVED (app-side guard; model-domain limitation flagged to 3d bucket) — TRELLIS.2 card-collapses flat 2D illustration
**Symptom:** 3D turnaround route produced blank white cells at every yaw; no error, no fallback.
**Root cause (two layers):** (1) TRELLIS.2 emitted a **near-zero-thickness card** (extents
0.666×**0.0001**×1.0) for the app's input — the subject painted on a slab; (2) a card lying off the
renderer's yaw plane is edge-on ≈ invisible at every yaw, so renders come back "successfully"
white. Diagnosed by orbiting the dumped GLB freely in macOS Quick Look (the app's fixed-yaw
renders could never reveal it).
**Resolution/pattern:** validate mesh EXTENTS before rendering — non-finite vertices, near-zero
bounds, AND near-planarity (min extent < 8% of max) all throw → the pipeline's diffusion fallback
takes over with the reason surfaced. General lesson: an image→3D stage needs a geometry sanity
gate between generation and rendering; "the model returned a mesh" is not "the mesh is a subject."
Also: dump pipeline artifacts to a USER-GRANTED folder (`<store>/_debug`), not the sandbox
container — TCC blocks terminal reads of container files, which stalls exactly this kind of
diagnosis.
**UPDATE 2026-07-10 (3d-bucket eval, `mlxengine-3d/CARD-COLLAPSE-EVAL.md`):** the "flat 2D art →
card" framing is **too broad — NOT a model-domain limitation.** A faithful reconstruction of the
same input class reconstructs **fully volumetric** in BOTH the Python oracle (min/max 0.43) and the
Swift port (0.57), and stays volumetric under matting (0.49) and hard cel-flattening (0.52). So the
collapse is a **rare, input-specific** failure, not a property of flat/anime art, and not a port
bug. Couldn't reproduce from the described input (exact bytes behind TCC — the app dumped only the
mesh, not the input; **fix: dump the input image too**). The app-side extent guard is the correct,
sufficient mitigation — keep it; do NOT add a blanket "matte/shade first" preprocess expecting it
to prevent collapse (matting didn't change the outcome here).
**App follow-through (same day):** now dumps `last-3d-input.png` beside the mesh (replayable), and
**square-pads** the reference before the 3D stage.
**Input captured 2026-07-10 → squash hypothesis REFUTED:** the actual collapsing input is
**474×474, already square** (staged `mlxengine-3d/DEV/TrellisDev/card-collapse/app-repro/`), so no
aspect distortion. Live hypothesis is now **low solid-volume coverage** — a thin-featured,
high-whitespace full-body anime character (thin staff pole full-height, thin legs/braids, large
white bg) gives the sparse-structure sampler too weak a signal, so it planes out; a different input
class than the solid-bodied eval reconstructions that stayed volumetric. The extent guard +
diffusion fallback remains the correct, sufficient app-side mitigation regardless of root cause.

## MSS-10 — klein LoRA as a package config (specialty registration), MSS pose cells
**Symptom/need:** wire the self-trained RefControl **pose LoRA** into the app so a skeleton +
locked front view → the character re-posed, all-klein/Apache/16 GB.
**Engine capability added (reusable):** `KleinConfiguration.loraPath` + `loraStrength` (env-specific
⇒ excluded from Codable). `Klein4BT2IPackage.load()` applies `KleinLoRA.apply` **after** the quantize
switch (rides the QLoRALinear/int4-safe activation add — same path the CLI `--lora` exercises), before
the generator is built. Turns a stock klein registration into a **specialty** with no separate base.
**Two-registrations-one-package pattern:** the app registers `Klein4BT2IPackage.registration` a SECOND
time with `id: PackageID("klein-pose")` — the explicit `id:` is REQUIRED, else the default PackageID
(= `manifest.surfaces.first.name`) collides with the base klein and the second registration silently
REPLACES the first (evicting its resident). This is the clean way to co-resident a base + a
LoRA-specialty off one package. Gate the registration on the LoRA file existing in the store
(`loras/refcontrol-pose-4b/pose.safetensors`) → absent ⇒ specialty just doesn't register, the
Add-Pose affordance stays hidden (graceful degrade, MSS convention).
**App-side (no engine gap):** ported PoseKit's VisionKit extractor + OpenPose renderer VERBATIM into
one `PoseExtractor.swift` — the renderer IS the format, so inference skeletons must be byte-style
identical to training (do not restyle without retraining). Pose cells are dynamic (like detail cells)
⇒ needed explicit persistence incl. the skeleton PNG. RefControl ref ORDER is load-bearing: Picture 1
= skeleton, Picture 2 = the **clean anchor front view** (not the busy raw import) — the trained
convention "apply pose from image 1 with reference from image 2".
**Local-package dev loop:** the app pins klein from the REMOTE; to build against the local PROD klein
with the new config field, add a workspace `<FileRef>` for `flux2-klein-swift` (same override the
workspace already uses for `mlx-engine-swift`) — Xcode prefers the local package over the same-named
remote. Push `loraPath` to the klein repo + repin when publishing deliberately (not done unprompted).

---

*Add the next app's section above this line. Keep entries terse — symptom → cause → pattern.*

## MAT-green but load() never materializes — the un-wired executor (Mage Demo, 2026-07-23)

**Catch:** the Mage-Flow demo app registered both family packages (t2i + edit), staged only the
t2i weights, and the first edit run threw `unreadableSnapshot("microsoft/Mage-Flow-Edit-Turbo")`
instead of downloading. The package's offline MAT suite was fully green — because MAT-1..5 verify
the *declarations* (`WeightSourcing`, store resolution, honest missing-sets), not that `load()`
actually EXECUTES a download. A package can pass the whole offline gate with a
`// materialization would run here` comment where the downloader belongs (Klein and Z-Image
carried exactly that comment — both retrofitted 2026-07-23: flux2-klein-swift 7b826d1,
z-image-swift 4eaa910; the thin Base/Turbo tiers inherit via their inner packages).

**Triage tell:** `needsDownload` true + prepare() fails immediately with the package's
snapshot-not-found error (no `.downloading` phase ever shown) ⇒ the package declared sources but
never wired the executor. This surfaces the FIRST time an app switches to a capability whose
weights aren't staged — single-capability demos never hit it.

**Fix (package side, not app side):** lift the MLXLTX2 `WeightMaterializer` pattern —
`load()` checks `missingWeightSources(storeRoot:)`, downloads each missing source via
swift-huggingface `HubClient.downloadSnapshot` into the ModelStore layout, forwarding
`WeightDownloadProgress` (source i of n spans [i/n,(i+1)/n)). MLXMageFlow 483e23d is the
second conforming implementation. The app's only job stays consent UX: check
`engine.needsDownload(capability)` before running and tell the user a multi-GB pull is coming.

**Known wart — RESOLVED 2026-07-23 (mage-flow-swift 6faa4cb):** the 0%-stuck fraction had
THREE compounding causes, all now fixed in MLXMageFlow's WeightMaterializer (the new
reference implementation, superseding the HubClient-snapshot pattern):
1. swift-huggingface 0.9.0's snapshot Progress never ticks during a file transfer (and it
   double-stores via its own cache) → replaced with direct URLSession streaming to the store.
2. `WeightDownloadProgress.sink` is a **TaskLocal** — reports made from URLSession
   delegate-queue threads read an unbound sink and silently vanish. Bridge deltas through an
   AsyncStream consumed by a child task (inherits the caller's binding); never call
   `report` from a non-task thread.
3. HF's `resolve/` endpoint for **xet-backed** repos serves ~0.5 MB/s per cold connection
   (CAS-bridge reconstruction; classic LFS ~50 MB/s) → files ≥64 MB download as 8 parallel
   ranged chunks written at offsets (hf_transfer design): measured 60–65 MB/s sustained.
Also: never iterate `URLSession.AsyncBytes` per byte — it collapses to ~1 MB/s in -Onone
(Debug app) builds; stream chunk-wise via URLSessionDataDelegate.

## MFD — Mage Demo (mlxengine-image/PROD/Mage, github.com/xocialize/mage-flow-demo, private)

The Gradio-Space-parity macOS consumer over MLXMageFlow — a compact worked example of the
whole golden path (one engine, `ModelStorageModel` held for app lifetime, store-before-
register, `ModelStateView`, `needsDownload` consent line, cancellable run Task) rendered in
the MLXEngineUI Marquee tokens, hosted in an AppKit no-IB shell via `NSHostingView`. Entries
above ("MAT-green but load() never materializes", the resolved 0%-progress wart) came from
this app; additional catches:

### MFD-1 — two co-resident packages each hold their own conditioner: budget for the SUM

Mage's t2i and edit are separate packages, each loading its own Qwen3-VL (8.3 GB) + VAE.
With both resident (user ran t2i, then edit), in-app phys hit **30.87 GB** vs 16.4 GB
single-package — the engine happily co-resides them within budget, but a 32 GB machine is
suddenly tight. If an app flips between sibling capabilities rather than using both, evict
the other capability before prepare (or push the package author toward a shared-conditioner
design / conditioner-evict config). Watch for this whenever one model family ships as
multiple packages with a duplicated heavy component.

### MFD-2 — in-app metrics collection (the registry-evidence pattern)

The app doubles as the measurement harness (MLXEngineTestKit is linkable in a consumer):
- **[MAT] line**: a settings-panel action runs `MaterializationBench.run` against a FRESH
  `MLXServeEngine` + an EMPTY temp store under the app-container tmp (no folder grant
  needed; delete after). Don't reuse the app's main engine — the bench must not disturb the
  session's store or residents. Note the bench's `sourceRepo` marker check is per-repo: for
  variant-multiplexed configs pass a repo the config's `weightSources` actually declares,
  or the marker probe false-NOs (pre-MS-6 it false-NO'd regardless; markers are
  variant-aware since engine 021f012).
- **[RUN] lines**: after each `run()`, log `HostMemory.physFootprint()` + secs/steps/size/
  quant to a `*.log` in the models folder (inside the grant, so plain FileHandle appends
  work) and mirror phys into the status line. These lines are the registry's Val/Eff
  evidence — collect them as you go instead of scheduling a separate measurement pass.

### MFD-3 — per-request config overrides on a shared pipeline: snapshot + defer-restore

When one resident package serves requests with different steps/size/seed, mutate the
pipeline's config per request but snapshot it first and restore in a `defer` — otherwise a
cancelled or thrown request leaks its overrides into the next one. (Wrapper-side pattern,
but it surfaces app-side as "settings stick after cancel".)

## GPD — Gepard Demo (mlxengine-audio/PROD/gepard/Gepard Demo, first full golden-path build, 2026-07-24)

A from-scratch consumer app (AppKit main.swift shell + SwiftUI via NSHostingController) that
followed this skill's golden path verbatim; the whole first-run flow — folder pick →
useModelStore → register → needsDownload consent → prepare (ModelStateView showed real % +
MB/s) → run — worked on the first launch with zero engine-side surprises. Two small catches:

### GPD-1 — Xcode 26 template settings hide engine observables and block the store

New-project defaults bit twice: (a) `SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY = YES`
means a view touching `engine.preparation` / `engine.runProgress` MUST `import MLXServeCore`
itself — importing only MLXEngineUI/MLXToolKit gives "property not available due to missing
import of defining module" even though the type is re-exported enough to autocomplete; (b) the
template's `ENABLE_USER_SELECTED_FILES = readonly` silently caps the folder grant below what
`ModelStorageModel` needs — the picker works but bookmark writes/downloads fail. Set it to
`readwrite` AND ship a real `.entitlements` (CODE_SIGN_ENTITLEMENTS) for the app-scope
bookmark + network-client keys, which have no build-setting equivalents.

### GPD-2 — Debug (-Onone) builds misrepresent AR-loop TTS performance ~4×

The same run measured RTF 1.10 under Debug and 0.28 under Release (port's validated gen-RTF
0.24). Xcode compiles SPM dependency Swift at -Onone in Debug, and an autoregressive
per-frame decode loop is exactly the shape that dies at -Onone (kernel-dominated diffusion
packages care much less). Demo/benchmark runs of token/frame-loop packages: always Release;
never quote Debug RTF numbers in a registry or README.

## Measuring a consumer app's footprint: three traps that produce confident wrong numbers (LFM 2.5 Demo, 2026-07-24)

The LFM2.5 demo/reference app wired `MLXEngineTestKit.ValidationHarness` and immediately
contradicted the package's own CLI bench. All three causes were measurement, not the model:

1. **Floor must be read POST-LOAD, never post-warmup.** The package's `--mem-bench` ran a warmup
   generation and called the footprint after it "the floor." A post-run floor conflates resident
   weights with retained intermediates: it over-reads the floor *and* subtracts that inflation back
   out of `peak − floor`. Mis-declared the package 6.5/0.5 GB when the truth was 4.89/1.06 GB.
2. **`isolate: true` on EVERY scenario, not just the first.** Without it the previous scenario's
   `phys_footprint` has not returned to the OS when the next post-load floor is read. Observed:
   floors drifting 4.89 → 5.02 → 5.50 → 6.67 → 6.85 GB within one suite, with activation reading
   exactly 0.04 GB whenever the floor was inflated — i.e. the error is self-concealing, since the
   two numbers move in opposite directions and the total looks plausible.
3. **One PROCESS per declarable number.** Even isolated, back-to-back scenarios carry residue. A
   `<PREFIX>_SCENARIO=<name>` env knob on the `HeadlessAutorun` path plus a shell loop makes runs
   reproducible to two decimals; an in-process suite is a smoke sweep. A 1.75 GB "tool-call peak"
   that would have inflated the declaration turned out to be pure cross-scenario contamination.

**Use `engine.trimCaches()` for the harness's `clearCache` hook — do NOT link MLX into the app.**
The engine exposes `trimCaches()` / `gpuPoolSnapshot()` precisely so a consumer gets pool handling
and observability without the dependency; adding MLX to the app target is the wrong reference
pattern. With it wired, `retain` fell from 0.94 GB (bogus) to 0.14–0.25 GB and the pool read
`active == on-disk weight bytes` exactly — a clean, checkable signature of an mmap-backed load.

**Cross-check two harnesses.** The CLI-vs-in-app disagreement is what exposed trap 1; neither
harness alone would have caught it. When they disagree, suspect the measurement before the model.

## Benchmarking a consumer app: Release only, and read `usage` (2026-07-24)

**Throughput measured from a Debug build is ~3× low and must never be quoted.** The token loop,
sampling and logit processing are Swift; unoptimized they dominate decode. Measured on one
identical scenario in the LFM 2.5 demo app: **88.2 tok/s Debug vs 267.4 tok/s Release.**

The failure is silent and the number looks plausible, so guard it rather than remembering it: put a
`#if DEBUG` warning in the headless benchmark path, and point the README's example command at
`Build/Products/Release/`. **Footprint is unaffected** — floor and activation matched to 0.01 GB
across configurations — so a Debug run stays fine for memory work; it is specifically the tok/s
column that lies. Worth stating explicitly in any harness readout, because "I measured it in the
app" reads as authoritative regardless of configuration.

**Read `LLMResponse.usage` (contract 1.26.0); never estimate.** Before it existed, consumers derived
throughput from `characters ÷ 4 ÷ runSeconds` — which measured ~72 tok/s against a real 277 tok/s on
the same run, **3.8× low**, because a reasoning model's stripped `<think>` tokens were invisible to
it. Render a missing `usage` as "—", not as a backfilled estimate: `nil` means "this package doesn't
report usage" and is not zero. Packages adopt at different rates (LFM reports it; qwen and gemma do
not yet), so a multi-package readout will legitimately be mixed.

**Quote `decode` in cross-model comparisons**, not prompt throughput: a package holding a
`ChatSession` across turns for KV-cache reuse prefills only the new suffix, so its `promptTokens`
is not comparable with one that builds a fresh session per turn.

---

## LLM Voice Chat (2026-07-25) — first mic-bearing consumer

**VC-1 · Swift 6 `installTap` isolation SIGTRAPs on the first audio buffer.** The app died the
instant press-to-talk opened the mic. `AVAudioNode.installTap`'s block is not `@Sendable`, so a
closure literal written inside a `@MainActor` type inherits MainActor isolation; the compiler
emits an executor check in the closure PROLOGUE and CoreAudio's realtime thread traps on
`dispatch_assert_queue`. Cost four crash reports because the fix was repeatedly attempted on the
closure's *body* (drop `self`, mark `@Sendable`, nonisolated factory, AsyncStream relay) when the
defect was its *isolation*. **The tell:** a small, CONSTANT crash offset (+140/+192/+196) across
attempts = prologue = signature problem. **Fix:** install the tap from a `nonisolated static`
function. Full detail: `references/realtime-audio-apps.md` §1. **Latent in the Nemotron ASR Demo
today** (Swift 5 target hides it).

**VC-2 · A folder pick one level too high silently re-downloads the whole stack.** The store
picker returned `/Volumes/Satechi` instead of `/Volumes/Satechi/Models`; the engine happily built
a second `models--org--name` store there and re-fetched ~7.2 GB that already existed. Nothing
errored — the app simply looked slow. **Fix (now in the app):** after resolving the store root,
check it for `models--*` entries; if it has none but a *child* directory does, warn
"this folder has no models in it — did you mean `<child>`?" before loading. Warn, never block:
an empty folder is a legitimate first run.

**VC-3 · Code-only app launched windowless.** `NSApplication.delegate` is weak and the delegate
was held only by a closure local, so it deallocated at launch — no callbacks, no window, process
alive. Compounded by a missing `.regular` activation policy (no dock icon) and a window ordered
onto a non-active Space (`isVisible == true`, nothing on screen). See
`references/realtime-audio-apps.md` §6.

**VC-4 · COLD START owns the first turn; after that, reasoning does — prefill never did.**
The first-ever turn read `stt=74ms firstToken=2242ms think=907ms firstAudio=5740ms`, and the
obvious reading ("prefill is slow, shorten the prompt, add held-KV") was **wrong on all three
counts**. The falsifying check took a minute: the package's own CLI gate reaches its first
delta in 203 ms using a system prompt *six times larger*. So prompt length was never implicated
— the first turn was paying cold Metal pipeline compilation plus the TTS reference-clip encode
(~850 ms), costs a long-lived CLI had already absorbed.

**Fix:** warm up at load time — one throwaway 1-token generation and one very short synthesis
*with the real reference clip* (that also memoizes the speaker prefix). Result on the next
first-turn: `firstToken=156ms (llm 49ms) firstAudio=1747ms`, with steady state ~1.2 s.

**Then re-read the profile**, because the bottleneck moved: warm, `llm` dispatch→first-token is
~50 ms, so **held-KV session reuse would optimize a solved problem** and was cancelled on the
data. What owns time-to-first-audio now is `think` (0.6–1.4 s on an always-reasoning model),
which trades against answer quality — a product decision, not a performance fix.

**Transferable rules:** (a) never profile a voice loop on its first turn; (b) split the metric
so model time is measured from REQUEST DISPATCH, not from speech-end, or the transcript wait
and pipeline setup hide inside "the model is slow"; (c) when app and CLI numbers disagree by an
order of magnitude, the difference is environmental (warm caches, resident siblings), not
algorithmic.

---

## MFG — ML(X) Media Forge (Forge R0/R1 rebuild, 2026-07-26/27)

### MFG-1 · RESOLVED — `licenseEnforcement` defaults to `.advisory`, so an engine bump silently loosens a commercial app

**Symptom:** none. That is the issue — there is nothing to observe.

**Root cause:** contract **1.28.0** (engine **v0.37.0**) turned C7/C8 from *refusals* into *declarations*.
Before it, `register()` threw `licenseRejected` on a non-permissive package and blocking was simply the
built-in behaviour. After it, the default `licenseEnforcement: .advisory` **registers** a package with
non-commercial weights and merely records a `LicenseAdvisory`. The parameter is **defaulted**, so an app
that never mentions it keeps compiling and quietly changes posture the moment its engine pin moves.

**Resolution / pattern:**

1. Construct the engine in **exactly one place** and pass it explicitly:
   `MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)`.
2. Put the warning **at that call site**. Nothing catches a regression at compile time — deleting the
   argument builds fine.
3. Know what `.blocking` does to your error handling: it **throws `EngineError.licenseRejected(gate)`
   from `register()`**, and `engine.licenseAdvisories` stays **empty by design**. So "surface the license
   state" means surfacing the caught *registration failure*, not reading the advisory list. Record a
   per-package outcome rather than letting the first inference fail.
4. Runtime tell that enforcement has slipped: a `[License]` log line per advisory.

⚠️ **Pin the tag, not the contract.** The engine's git tags are a `0.x` series separate from the `1.x`
ContractVersion, and we planned against a contract that was pushed but **unreleased** — v0.36.0 is
contract 1.27.0 and has no `licenseEnforcement` symbol at all. Settle it with
`git show <tag>:Sources/MLXToolKit/ContractVersion.swift`, and **quote a contract version only alongside
the tag that carries it**.

### MFG-2 · PATTERN — a capability's real behaviour needs a channel to the receipt

Not an engine bug; a design smell worth naming because it recurred **four times in one release** across
different packages, each time as an *invisible* failure rather than an error.

| Case | What the caller was told | What happened |
|---|---|---|
| Upscale factor | the factor **requested** | a fixed-4× model returned 4× for a 2× request |
| Video quality | one scalar score | a p10 **percentile over a sample**, min well below the floor |
| Alpha ingest | frames decoded fine | the alpha stream was never demuxed; output silently opaque |
| Matte fallback | a matte, no error | `.best` had degraded to `.fast` on an under-memoried device |

Every one has the same shape: **the provider protocol returned a bare artifact** (`CGImage`, a file, a
`Double`) with no field for what actually occurred, so the honest answer was structurally undeliverable
however careful the implementation.

**Pattern when adding a provider protocol:** if a capability can degrade, substitute, or partially apply,
give the seam somewhere to say so — a result struct, or a `lastApplied*` on the adapter when the protocol
is fixed. A capability flag (`decodesAlpha`, `supportsStrength`) should **default to the pessimistic
value**, so a provider that has not been updated lands on "cannot" rather than silently claiming it can.
And prefer **measuring the artifact** over trusting a report: deriving the upscale factor from the output
pixels needed no protocol change and cannot be lied to by any provider, present or future.

## MVD — MLX MageVL Demo (mlxengine-think/PROD/MageVL, 2026-07-28)

### MVD-1 — a control bound to local `@State` when the value is package CONFIGURATION

The Video tab had a "Sampled frames" stepper and a "Visual token budget" stepper, both `@State`, both
feeding a local frame-gallery preview. The metrics harness had the same shape: scenarios named
`video 16f` / `video 32f` that varied a number used **only** in its own token re-derivation.

Neither reached the model. `frameCount` and `visualTokenBudget` live on `MageVLConfiguration`, which is
fixed at **register** time — so every run used the package defaults while the screen and the metrics
table both asserted otherwise. The metrics row was the worse case: `video 32f` reported a footprint and a
token count for a 32-frame run that never happened, and looked plausible because 16f and 32f differ only
slightly at a fixed budget.

**Test:** for every UI control and every benchmark scenario, ask *where does this value enter the
engine* — request field, or configuration? A configuration value has to `evict` + re-`register`; there is
no cheaper path, and binding it to local state produces confident wrong numbers rather than an error.

Once wired correctly the measurement inverted the intuition it was built to check: at a fixed token
budget, **doubling the frame count barely moves activation** (4.01 → 4.37 GB) because per-frame
resolution drops to compensate — 32 frames yields *fewer* visual tokens than 16. Doubling the *budget*
at fixed frames is what costs (→ 6.05 GB, run time 5.5 → 12.3 s). Keep a scenario that varies only the
budget, or the table shows the wrong lever.

### MVD-2 — a stale `peakActivationBytes` silently makes a package inadmissible

Same package, found by the same run. The manifest declared 14.5 GB activation, measured honestly — but
*before* video resolution was derived from `visualTokenBudget`. At the shipped defaults the real figure
is 4.0 GB, identical on bf16 and int8. The governor reserves what is declared (visible as
`reserve=14.50GB` on every `splitLogLine`), so bf16 declared 9.5 + 14.5 = 24 GB and would be refused on
machines that run it comfortably in ~14.

**Any change that alters how much work a request does — a default frame count, a token budget, a
resolution derivation — invalidates the declared footprint.** Re-measure in the same change. An
over-declaration fails *closed* and so never shows up as a bug report; it just quietly shrinks the set of
machines the package admits on.

### MVD-3 — measuring a sandboxed demo app with nobody at the keyboard

The models folder reaches a sandboxed app only through an `NSOpenPanel` pick (that pick *is* the grant),
so there is deliberately no path env-var: a sandboxed process cannot reach a folder it was merely told
about. That blocks autonomous measurement runs.

Working approach: keep the shipping app sandboxed, and accept a `*_MODELS` override **guarded by a real
readability probe** — it succeeds on an unsandboxed dev build and fails cleanly under the sandbox, so it
can never become a silent wrong answer. Then measure a re-signed scratch copy:

```bash
cp -R "App.app" /tmp/ && codesign -f -s - --entitlements dev-nosandbox.entitlements --deep "/tmp/App.app"
```

Numbers are otherwise identical — same engine, same package, same harness. What this does **not** cover
is the bookmark path itself, so a row validated this way stays 🟡 until one human GUI launch confirms the
grant persists across relaunch.

### MVD-4 — log an excerpt of the ANSWER, not just the megabytes

A metrics suite that reports only footprints proves the pipeline ran, not that it produced anything. An
empty, truncated or looping answer is exactly what a footprint table hides — and it is one line to close:
`[say] <scenario> chars=<n> · <first 220 chars>`. It caught nothing here (answers were specific and
correct on both quants), which is the point: the run is only evidence if a bad answer would have shown.

### MVD-5 — two engines: one shown, one driven (and the misdiagnosis it caused)

`DemoWindow` owned a `MageVLEngine` and bootstrapped it. `RootView` declared
`@StateObject private var engine = MageVLEngine()`. Those are **different objects**. The one the UI
rendered was never bootstrapped, so the header pill read `.needsFolder` on every launch no matter
how well the models-folder bookmark restored — and only woke when the settings sheet's Done
happened to call `modelStoreChanged()`.

Two engines also meant two `ModelStorageModel`s, so the storage panel and the header pill were
reading different objects and could disagree about whether a folder was granted. The panel showed
the right path, Ready, and a correct model count while the pill said "Choose folder".

**The expensive part was the misdiagnosis.** That symptom set — path restored, panel healthy, no
access, fixed by re-picking — is a near-perfect impression of `restoreBookmark()` resolving a
bookmark whose `startAccessingSecurityScopedResource()` then fails. It sent the investigation into
entitlements, `codesign -d --entitlements`, container preference plists, unified-log spelunking,
and App Translocation, and produced a defensive `MLXEngineUI` change released on a premise that was
simply not true.

**Check instance identity before you check the OS.** A `@StateObject` in a view initialiser is a
constructor, not a reference — if something else already owns that object, injecting it with
`@ObservedObject` is the fix, and `ObjectIdentifier(engine)` printed from both sites settles it in
one run. Cheap tests that would have caught it first:

- does the state the UI reads ever get its bootstrap called, on the instance the UI actually holds?
- do two views that should agree about the same model ever disagree? That is an identity smell, not
  a permissions smell.

Corollary for the component author: `ModelStorageModel` distinguished "restored and allowed in"
from "restored but locked out" only by whether a private `accessedURL` was nil, and said nothing.
When a component can end up in a state that *looks* configured but is not, it has to say so — an
absent signal gets replaced by an inferred one, and the inference can be wrong.

### MVD-6 — an in-process sweep UNDER-reads activation, and that is the dangerous direction

The measurement rule everyone quotes — *one process per declarable number* — is usually justified
by the FLOOR: the previous scenario's `phys_footprint` has not returned, so the next floor reads
high. True, and that is how it was originally found.

The same rule bites **activation** harder, and the other way. Measured on the same package, same
machine, same day:

| scenario | in-process sweep | one process per number | error |
|---|---|---|---|
| video 16f @ budget 4096 | 4.01 GB | **5.79 GB** | −31 % |
| video 16f @ budget 8192 | 6.05 GB | **14.19 GB** | −57 % |

A warm MLX buffer pool already holds allocations from the previous scenario, so the *incremental*
`phys_footprint` growth the sweep sees is much smaller than the real transient. The floor can look
perfectly stable while activation is understated by more than 2×.

**Under-declaring is worse than over-declaring.** An over-declared footprint wrongly excludes
machines — visible, annoying, harmless. An under-declared one gets the package *admitted* onto
hardware it will then OOM on. So: treat an in-process suite as a **lower bound on activation**,
never as evidence for a declaration. Declare from one-process-per-number runs only.

### MVD-7 — a bench that skips a production step reproduces a stale number forever

The stale footprint above survived a re-measurement because the measuring tool had quietly diverged
from the code it measured. `--mem-bench` called `preprocessFrames(...)` directly, while the
package's `run()` called `.budgeted(tokenBudget:frames:)` first and *then* preprocessed. So the
bench exercised a path production never takes, and every re-run faithfully reproduced the
pre-budget figure — 32,640 patches instead of 16,128.

That is the worst failure mode available to a benchmark: it does not error, it does not look
stale, and re-running it *confirms* the wrong number. The drift was invisible because both sides
were individually correct; only the fact that they no longer described the same request was wrong.

**When a request's cost changes — a default frame count, a token budget, a resolution derivation —
the declared footprint AND the harness that measures it both go stale.** Two checks worth making
routine:

- have the bench call the same seam production calls, not a lower-level one that happens to be
  convenient. If `run()` composes two steps, the bench composes the same two.
- when a number refuses to move, try to *reproduce the old value on purpose*. Here, running the
  fixed bench at `--budget 8192` returned 14.19 GB against 14.16 GB on record — which converted "the
  old number is wrong" into "the old number described a configuration that stopped being the
  default," and that is a different bug with a different fix.

## MFU — ML[X] Media Upscale V10 harness (ml(x)/ML[X]-MediaForge, 2026-07-30)

### MFU-1 — headless sandbox denial has a *progressively flaky* signature, and staging around it is a trap

A measurement harness inside a sandboxed product app (`ENABLE_APP_SANDBOX = YES` in the project) was
launched headless from a script. Two gates denied file reads with the same
`NSCocoaErrorDomain 257 "Operation not permitted"`: the App Sandbox itself (blocks `/Volumes/...`
AND `/private/tmp` alike), and TCC re-attribution — every rebuild re-signs the DerivedData app
ad-hoc, invalidating whatever removable-volume grants the app identity had earned interactively.
The tell was *progressive flakiness*: run 1 read the weights then lost the corpus; run 2 read
nothing. Nothing prompts, because nobody is at the keyboard (cf. MVD-3).

Two wrong fixes tried first, kept here because both looked reasonable:
- launching the same binary with the caller's sandbox disabled — irrelevant, the app's own
  entitlements govern;
- staging weights+corpus into `/private/tmp` — defeated by the sandbox anyway, and **a partial
  store copy is worse than none**: packages missing from the staged root silently fall back to the
  network mid-sweep and fail late.

The fix that held: build the harness copy with the sandbox off, into a session-private DerivedData —
`xcodebuild … -derivedDataPath "$SCRATCH/dd" ENABLE_APP_SANDBOX=NO build` — leaving the project
file shipping `YES`. Verify with `codesign -d --entitlements -` before trusting the first run.
MVD-3's re-sign-a-scratch-copy route is equivalent; the build-setting route wins when you are
rebuilding anyway (per-session DerivedData also stops two concurrent agents clobbering each other's
builds — which happened).

### MFU-2 — `xcodebuild -resolvePackageDependencies` will not move to a tag you pushed a minute ago

Existing pins that still satisfy the version requirement are treated as final. Deleting the
project's `Package.resolved` does nothing on its own — Xcode reconstructs it from
`SourcePackages/workspace-state.json`, and the cached git mirrors (in
`~/Library/Caches/org.swift.swiftpm/repositories/` AND `<DerivedData>/SourcePackages/repositories/`)
may not have the new tag yet. Force order: fetch/delete both mirror sets → delete
`workspace-state.json` + `Package.resolved` → re-resolve → grep the output for `pkg @ version`.
Before touching caches, resolve a scratch SPM package pinning `exact:` the new version — it
separates "Xcode cache" from "real graph conflict" in one step.

**AMENDED 2026-08-01 — the force order above is INCOMPLETE; add the manifest cache FIRST.**
Measured on the Moebius Demo dual-backend build: freshly-pushed tags stayed unresolved through the
entire procedure above — both mirror sets purged AND verified refetched (`git tag` in the mirror
listed the new tags), both `Package.resolved`s deleted, `checkouts/` purged. The stale view lived in
`~/Library/Caches/org.swift.swiftpm/manifests`, a **user-level cache no DerivedData purge touches**:

```
rm -rf ~/Library/Caches/org.swift.swiftpm/manifests
```

then resolve — correct version immediately.

⚠️ **Raising the consumer's `minimumVersion` to the new tag also "fixes" it, and that is a trap**:
it forces the floor above the cached version, masking the stale cache instead of clearing it. That
is how this gets misdiagnosed as "the resolver picks the lowest version in range" or "an Xcode bug"
— it is neither. Proven by lowering the floor back to the OLD value after purging manifests: still
resolved to the new tag, i.e. SPM picks the highest in range correctly once the cache is clean.
Bumping `minimumVersion` to the fix tag remains good hygiene for a different reason — a floor left
at an old version lets any consumer legitimately keep serving it — but do not mistake it for the
cure. Confirm the real thing with `git -C <DerivedData>/SourcePackages/checkouts/<pkg> describe --tags`.

## LTX Studio (2026-08-22): app-side `RunProgress.$sink` binding never fires at engine ≥ 0.48 — observe the engine-owned monitors instead

**Symptom.** A consumer wraps `engine.run(...)` in `RunProgress.$sink.withValue(onProgress) { ... }`
(the pattern older QuickStart scaffolds documented) and the closure never fires — no run-phase
events, a dead stepper — while the run itself completes fine.

**Why.** Since the engine took ownership of run-progress (v2 signal / preemption), `MLXServeEngine`
executes the package inside its **own unstructured `Task`** and binds its **own** sink around
`instance.run(request)` (MLXServeEngine.swift run wrapper: `let task = Task { ... RunProgress.$sink
.withValue(sink) { try await instance.run(request) } }`). A fresh unstructured Task does not inherit
the caller's task-locals, and even if it did, the engine's inner binding shadows any outer one. An
app-side `withValue` around `engine.run` is therefore structurally unreachable — not flaky, DEAD.

**The correct consumer pattern** (mirrors `ModelStateView`/`PreparationMonitor`):

```swift
// nonisolated lets on the engine — no await needed to grab them:
let prep = engine.preparation      // PreparationMonitor (@MainActor @Observable)
let run  = engine.runProgress      // RunMonitor        (@MainActor @Observable)
// SwiftUI: read run.report(for: .textToVideo) in body; nil == no run in flight.
// The engine clears the entry on EVERY exit (return/throw/cancel).
```

Correlate reports to stepper nodes per the package's plan (`plannedStages(...)` for LTX): match
`report.phase.rawValue` to the node's `phase`, disambiguate repeated phases (two-stage `denoise`)
with `report.stage` vs the node's `occurrence`, and keep the index MONOTONIC — the monitor exposes
only the latest report, so a view that recomputes naively can step backwards on repeated phases.

**Recognize it fast:** progress closure never called + run completes + `engine.runProgress` updates
in the debugger → you are on the dead pattern. Fix is a deletion, not a workaround: drop the
`withValue`, bind the monitors.

## LTX Studio (2026-08-22): SwiftUI `VideoPlayer` SIGABRTs on macOS 27.0 beta — wrap AppKit `AVPlayerView` instead

**Symptom.** The app aborts (SIGABRT, `abort() called`) the instant a result view containing
SwiftUI's `VideoPlayer` enters the hierarchy. Crash stack: `swift::fatalError` →
`getSuperclassMetadata` → `_swift_initClassMetadataImpl` → `_AVKit_SwiftUI
__swift_instantiateGenericMetadata` → `NSViewRepresentable._makeView`. Observed on macOS 27.0
beta 26A5416b (SwiftUI 8.0.84.1.406) at the worst possible moment: the first successfully
finished generation flipping the UI to its player card — 15 minutes of compute presented, then
abort (incident 2EFD86F5, 16:24:26).

**Why.** The Swift runtime fails to realize `_AVKit_SwiftUI`'s generic class metadata (superclass
demangle fails) on this OS/SDK pairing. Not app logic, not Metal API validation (the launch was
via `open`, where scheme-level Metal validation isn't even active), and nothing MLX — the engine
side had already returned and the file was safely stored.

**Fix (drop-in).** Avoid `_AVKit_SwiftUI` entirely — a 20-line `NSViewRepresentable` over AppKit's
`AVPlayerView` (`view.player = player`, `controlsStyle = .inline`) renders and plays the same MP4s
fine. Revisit `VideoPlayer` only after an OS/SDK move.

**Recognize it fast:** SIGABRT with `_AVKit_SwiftUI` + `getSuperclassMetadata` in the first 8
frames, fired on the first appearance of a video result view. The generation pipeline is
innocent — check the log for the `Generation done` line landing right before the crash.

## LTX Studio P2 sprint (2026-08-22/23): three supply-chain & verification gotchas

**1. Branch pins do not advance on resolve — edit the resolved revision directly.** A package
pinned to a BRANCH (`ltx-2-mlx-swift@main`) keeps its `Package.resolved` revision through every
`-resolvePackageDependencies`; work landed on the branch after your first resolve (here: the
LoRA `clearedFamilies` stamps) silently never arrives. The version-pin force-order (mirrors →
manifests cache) is the wrong tool — for a branch pin just write the new head SHA into
`Package.resolved`'s `state.revision` (get it from `git ls-remote <url> <branch>`) and resolve;
Xcode honors the edited revision. Distinct mechanism from the stale-TAG cache entry above.

**2. Terminal is TCC-blind to app containers — and it fails EMPTY, not loud.** `ls` on
`~/Library/Containers/<bundle-id>/Data/...` returns "Operation not permitted" — but piped through
`2>/dev/null | wc -l` it reads as an empty directory, and even bare `ls` output can mislead a
quick scan. Meanwhile a direct `stat <exact-file-path>` often succeeds. So: verify container
files by exact-path `stat`, never by directory listing; an id from the app's own log lines
(file names carry them) gives you the path. A "0 files" readout from Terminal is NOT evidence
of absence.

**3. Smoke fixtures must live under an existing sandbox grant.** A launch-argument file path
(`-SmokeImage /tmp/...`) silently no-ops in a sandboxed app — `NSImage(contentsOf:)` returns nil,
no prompt, no error dialog. Stage fixtures inside a folder the app already holds a
security-scoped grant on (the MODEL STORE is ideal: `<store>/_smoke/…`) and pass that path.
Corollary for thumbnails/detached work under Swift 6 default-MainActor isolation: hand Sendable
bytes (`Data`) across the task boundary and build `NSImage` main-actor-side.

## LTX Studio (2026-08-23): an upstream HF org migration breaks hardcoded WeightSource repos — diagnose by anonymous probe, unblock by hand-stage, fix in the package

**Symptom.** `engine.prepare(...)` on a fresh store fails with
`MLXHubMetadata.HubMetadataError error 0` (= `.httpStatus`) at the LISTING stage — before any
bytes move. Adding a verified HF token does NOT fix it.

**Cause.** The model's owner migrated the repo to a new org (here: `microsoft/Mage-Flow-*` →
`mage-flow-community/*`) and HF serves the old path a 401 **even authenticated** — no
API-level redirect for tree listings. A package with a hardcoded `WeightSource` repo string
is dead on fresh stores from that moment, silently, with an error that *looks* like an auth
problem. Tokens are a red herring: probe the suspect repo anonymously
(`curl …/api/models/<repo>/tree/main`) and probe the rumored new home — a 401-old / 200-new
pair is a migration, not gating.

**Triage order that worked:**
1. Probe old + new paths anonymously (curl, no token) — classifies gating vs migration in
   seconds and tells you whether the fix ADDS or REMOVES an auth requirement.
2. **Interim unblock, consumer-side, zero package changes:** hand-stage the needed files FROM
   THE NEW org INTO the store directory named for the DECLARED (old) repo
   (`models--<old-org>--<name>`), matching the config's globs, size-verified. The package's
   presence check short-circuits the listing entirely — the dead path is never consulted.
   This is the skill's hand-staged-store pattern applied as a shim.
3. **Durable fix belongs in the PACKAGE** (one-line repo constant + provenance rows +
   registry note), shipped with a migrate-by-RENAME instruction: after the bump the engine
   looks for `models--<new-org>--<name>`, and a rename of the store dir carries every byte —
   including your interim hand-stage — with no re-download.

**Adjacent trap from the same session:** engine-management panels aside, remember the app's
own smoke/probes may cache the OLD org name in fixtures or scripts — grep for the org string
once the bump lands.

## LTX Studio (2026-08-23): consumer-side verification of a generative surface — TCC-blind outputs, and the full-span control that separates "prompt ignored" from "prompt overpowered"

**Context:** first consumption of the port's `.videoEdit` (retake/extend). Operator reported
zero visible effect from retake prompts across multiple runs; app-side wiring (ask → request
prompt field) verified clean, so the question became which side owned the failure — and the
receipts had to be *frames*, not logs.

**Gotcha 1 — the harness cannot read the app's own outputs.** Sandboxed-app outputs live in
the app container, and macOS TCC blocks other processes (your shell, ffmpeg, even `cp` by
exact path) from reading another app's container: `Operation not permitted` with no prompt in
headless flows. Don't weaken TCC and don't bounce files through the user. **Pattern:** the
app already holds a security-scoped grant on its model-store folder — add a smoke launch arg
(`-…ExportOutputs`) that has the APP copy its library files into `<store>/_smoke/exports/`.
The harness reads them freely from there. Same trick as input fixtures (`_smoke/…` for i2v
images), run in reverse.

**Gotcha 2 — "the edit did nothing" has three different causes; one cheap probe splits them.**
Frame-grid the base vs. edited outputs at fixed timestamps (ffmpeg extract + hstack/vstack,
then LOOK at the grid). Near-identical frames with slight drift = the span re-denoised but the
prompt exerted no force — still ambiguous between (a) prompt embeds unused (hard bug) and
(b) context inertia (few-step distilled denoise + hard-frozen clean neighbors re-blended each
step leaves the prompt no room to introduce content). **The control that decides it: run the
same edit full-span** (start 0, duration = clip length) so nothing is frozen. Prompt obeyed →
conditioning path healthy, failure is context dominance (a tuning ask: more steps/CFG on
spans, soft-context anneal, feather). Prompt still ignored → hard bug in the prompt path.
Report whichever verdict with the grid + control receipts; that's a one-round ask instead of a
speculation thread.

**Timing corollary:** masked-span edits denoise the FULL clip geometry every time — cost is
geometry-driven, not span-driven. Don't infer per-span pricing from one noisy run; A/B it
(full-span ≈ short-span confirmed it here).

## LTX Studio P3 (2026-08-24): adding a REMOTE backend beside the engine — three live-wire lessons

**Context:** wiring the LTX cloud API as a second backend behind the same UI that drives
MLXEngine locally. The reference consumer (vendor's Electron app) shipped working code, and
its spec tables STILL didn't describe the wire.

**1. The shipping app's spec layer is not the wire — verify with one live call per path.**
The vendor app's capability tables list resolution rungs ("720p"); its handler silently
translates them to pixel strings ("1280x720") before POSTing, and the live API 400s on the
rung names. Same shape again for audio: nothing in the app says raw PCM is rejected — the
live 400 enumerates the accepted codecs (aac/mp3/vorbis/opus/flac). **Pattern:** transcribe
the reference client's payload-building code (not its spec/validation layer), then fire ONE
minimal live probe per endpoint before building UI on top. Budget for the 400s — each one is
a receipt; pin the request id in a ⚠️ comment.

**2. Same-probe-on-both-lanes classifies "defect" vs "family trait".** A capability that
disappoints locally (weak in-span prompt insertion on retake) earns a port bug report ONLY if
the reference service does better. Running the identical probe (same source, span, prompt)
against the cloud tier answered it: fails there too → family-wide trait, reframe the UI copy,
spare the port a doomed parity chase. Bonus: the comparison surfaced the REVERSE
differentiator (cloud retake re-renders the whole clip at a new resolution; the local masked
span-replace is the only geometry-preserving edit path) — worth a receipt to the port team.

**3. Absence-of-capability receipts need a cheap objective probe.** "Does the API honor a
seed?" = two identical requests + PSNR on matching frames (9 dB = ignored). "Is there an
OpenAPI surface?" = four 404s. Ten minutes of probes converted "maybe the API offers more"
into closed, citable verdicts — and killed a false record-honesty bug before it shipped
(records must not store a seed the wire ignored).

## LTX Studio (2026-08-25): AV-sync debugging — drift vs constant, the shift-ladder instrument, and the timebase-squeeze bug class

**Context:** local audio-conditioned generation (a2v) read as "lip sync off" while the cloud
lane with identical audio was tight. Resolved to operator-synced in one day, entirely with
consumer-side instruments.

**1. The perceptual shift ladder is a real measurement instrument.** Copies of the clip with
only audio timing shifted (±100/200/300 ms; then a finer ±50–200 rung set) — the operator
picks the best. It measured a ~200 ms offset, then its own falsification: "hard to choose,
aligns differently as the track progresses" = **drift, not constant** — the observation that
cracked the case. The port meanwhile tried an objective estimator (motion-energy × audio
envelope cross-correlation) and it FAILED validation against the ladder's known steps
(r ≤ 0.09) — talking heads carry too much non-mouth motion. Validate any instrument against
known shifts before trusting it; a disciplined perceptual ladder beats an unvalidated metric.

**2. The timebase-squeeze bug class.** If the conditioning span is derived from the MEDIA's
duration while generation runs on a quantized FRAME grid (8k+1 here), any off-grid input gets
its conditioning tokens squeezed → LINEAR sync drift (track 10.2 s onto a 10.04 s grid ≈
158 ms accumulated), invisible on short/coarse content (claps at 5 s passed) and glaring on
speech. **Consumer-side fence:** fit the media to the frame grid at ingest — and per the
operator, PAD UP with silence (never trim supplied content; trimming is only legitimate at
the hard envelope cap, where a longer track would re-introduce the squeeze at the clamp).

**3. Decompose before filing.** The measured 200 ms was squeeze-drift (~158, fixable at
ingest) + a vendor-shared +70 ms constant (token-probed identical on both arms — present but
perceptually marginal once the drift was gone: operator picked UNSHIFTED). Filing "200 ms
offset" as one bug would have sent the port hunting a number that no single mechanism owns.

**4. Doc-site beats app-internal tables, but the wire is lenient.** The service's official
docs contradicted the reference app's internal spec tables (duration floor 6 even-stepped vs
2–20-with-odds) — and the live API silently ACCEPTED out-of-spec values our earlier runs sent.
Offer the documented envelope in UI; treat past lenience as receipts, not contract.
