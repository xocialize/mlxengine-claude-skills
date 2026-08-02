# Topic 6 — Sandbox, entitlements & the offline-vs-live metallib boundary

Two environment facts trip up MLXEngine apps: what the **App Sandbox** requires, and why GPU inference
**only runs under the app build**, not from a plain `swift run`.

## Sandbox & entitlements

A sandboxed consuming app needs, at minimum:

```xml
<!-- App Sandbox on -->
<key>com.apple.security.app-sandbox</key>                    <true/>
<!-- HF weight downloads -->
<key>com.apple.security.network.client</key>                <true/>
<!-- The model store: user-picked folder + persisted access across launches (topic 1) -->
<key>com.apple.security.files.user-selected.read-write</key> <true/>
<key>com.apple.security.files.bookmarks.app-scope</key>      <true/>
```

Why each: network-client lets packages pull weights from Hugging Face; the two files entitlements are
what make the `ModelStorageModel` bookmark work (without them the folder pick succeeds for the session but
the grant can't persist, so every relaunch re-prompts). Metal/GPU needs **no** special entitlement.

### The container vs. the granted folder

Under the sandbox, `FileManager.homeDirectoryForCurrentUser` is the **app container**, not the real
home — so a package that naively reads `~/.cache/huggingface` finds an empty/unwritable path. This is
exactly why the model store exists: the engine stamps the **user-granted** folder onto each package
(topic 1), and a package reads weights from that root under the active security scope. If a package needs
to read pre-existing weights from a granted folder (e.g. an HF cache the user pointed at), it does so via
a `weightsRootOverride` + the security-scoped bookmark — the same pattern, not a broad entitlement.

**Keep the `ModelStorageModel` alive** for the app's lifetime — the security-scoped access lives on that
instance; if it deinits, writes under the store root start failing mid-session (topic 1).

## Running a sandboxed product app HEADLESS for measurement (V10-fix, 2026-07-30)

Launching a product app's binary from a script (bench harnesses, envelope sweeps) fails in ways that
do not reproduce from an interactive shell, because **two independent gates deny reads and neither
prompts anyone who can answer**:

1. **App Sandbox.** The product ships `com.apple.security.app-sandbox = true`, so the process cannot
   read `/Volumes/...` or even `/private/tmp` staging dirs regardless of TCC state. The failure is
   `NSCocoaErrorDomain 257 "Operation not permitted"` on paths your own shell reads fine.
2. **TCC re-attribution after rebuild.** Every rebuild re-signs the DerivedData app ad-hoc, which
   invalidates any removable-volume/folder grants the app identity had earned. Headless, the consent
   prompt either never fires or fires on a screen nobody is watching. The signature is *progressive
   flakiness*: one run reads weights then loses the corpus, the next run reads nothing.

**The fix is a build-time override, not staging copies:**

```bash
xcodebuild -project App.xcodeproj -scheme App -configuration Release \
  -derivedDataPath "$SCRATCH/dd" ENABLE_APP_SANDBOX=NO build
```

- `ENABLE_APP_SANDBOX=NO` on the command line — the project keeps shipping `YES`; only the harness
  build drops it. Verify with `codesign -d --entitlements - <app>` before trusting a run.
- **Own `-derivedDataPath` per session/agent** — two concurrent sessions sharing the default
  DerivedData clobber each other's builds and package resolutions.
- Staging weights into `/private/tmp` to dodge the sandbox is a trap: a partial store copy silently
  sends the un-staged packages to the network mid-run and fails late. Point at the real store and
  drop the sandbox instead.

### Freshly pushed tag not picked up by `xcodebuild -resolvePackageDependencies`

Resolution treats existing pins that still *satisfy* the requirement as final — it updates nothing.
Deleting the workspace `Package.resolved` is **not enough**: Xcode reconstructs it from
`DerivedData/<app>/SourcePackages/workspace-state.json`. To force a fresh-latest resolution:

1. `git fetch --tags` inside the cached mirrors (`~/Library/Caches/org.swift.swiftpm/repositories/<pkg>-*`
   and `<DerivedData>/SourcePackages/repositories/<pkg>-*`) — or delete them;
2. delete `<DerivedData>/SourcePackages/workspace-state.json` AND the project's `Package.resolved`;
3. re-run `-resolvePackageDependencies` and grep the output for the expected `pkg @ version`.

A plain-SPM probe package with `exact:` the new version resolves the graph independently — use it to
distinguish "Xcode cache" from "real dependency conflict" before touching caches.

## The offline-CLI vs. live-Xcode metallib boundary

The single most confusing thing for someone testing an integration:

> **GPU inference does not run from `swift run` / `swift test`.** The SPM command-line build can't load
> the MLX default metallib (`MLX error: Failed to load the default metallib`), so any code path that
> issues GPU evals fails there. Live model execution works **only under Xcode's build system** — the app
> target, or a tool/CLI built via `xcodebuild` that links the Metal stack.

Practical consequences for an app dev:

- **Unit-test the coordinator offline, the inference in-app.** The engine's own wiring — registration,
  routing, admission, eviction, the memory governor — is pure Swift with mock packages and tests fine via
  SPM (`swift test`). Real load/run must happen in the app (or an `xcodebuild`-built engine-driven CLI).
- **Don't expect a `swift run MyTool` to do GPU work.** If you want a headless smoke that actually
  loads + runs, build it through `xcodebuild` so it gets the metallib (the engine's own `RunX` CLIs do
  this). On this box the CommandLineTools `swift` is also unreliable — use
  `env DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift|xcodebuild …`.

## Metal API Validation gotcha (Debug scheme) — a recurring one

**Symptom.** The app `SIGABRT`s mid-inference with something like:

```
-[MTLDebugComputeCommandEncoder setBytes:length:attributeStride:atIndex:]:387:
  failed assertion 'bytes argument cannot be nil.'
```

**The tell:** the crashing class is `MTLDebug…` — those classes **only exist when Metal API Validation
is enabled** (the default for a Debug scheme). So the crash is the *diagnostic layer*, not your code and
not the package's logic.

**Root cause.** MLX legitimately dispatches **empty arrays** (a tensor with a zero-length dimension —
common in a gather/concat intermediate). Its `set_vector_bytes` then calls
`encoder.setBytes(vector.data(), length: 0, …)`, and `vector.data()` is `nullptr` for an empty vector.
That `setBytes(nullptr, length: 0)` is **benign at runtime** (zero length, ignored), but Metal API
Validation hard-asserts on the null pointer. **MLX and Metal API Validation are broadly incompatible**
around empty buffers — this is not specific to any one package (we first hit it via Kokoro TTS, but any
MLX-backed package can trip it).

**Fix.** Turn it off for the run scheme: **Edit Scheme → Run → Diagnostics → uncheck "Metal API
Validation."** This is the standard configuration for MLX-Swift apps, not a workaround — Release builds
already run with it off. Because the autogenerated scheme defaults it **on**, the toggle lands in
`xcuserdata` (per-user); to make it durable across machines/CI, commit a **shared** `.xcscheme` with
Metal API Validation disabled.

**Before blaming the package:** confirm the crash class is `MTLDebug…`. If it is, it's the validation
layer — flip the toggle and re-run before filing anything. (If it *still* crashes with validation off,
*then* it's a real empty-buffer/logic bug worth a package fix.)

## Recognizing the anti-pattern (review checklist)

- [ ] Missing `network.client` (downloads silently fail) or the two files entitlements (bookmark won't
      persist → re-prompt every launch).
- [ ] A package reading `~/.cache`/`homeDirectoryForCurrentUser` directly instead of the granted store
      root → "no models found" under the sandbox.
- [ ] Letting the `ModelStorageModel` go out of scope, dropping the security scope mid-session.
- [ ] Concluding "inference is broken" from a failing `swift run` — that's the metallib boundary, not a
      bug; run it in the app.
- [ ] A model that `SIGABRT`s in a Debug run inside an `MTLDebug…` encoder (e.g. `setBytes 'bytes
      argument cannot be nil'`) — that's Metal API Validation on a benign MLX empty buffer, not a bug;
      disable Metal API Validation in the scheme.

## Verify against source

`mlx-engine-swift/Sources/MLXToolKit/ModelStore.swift` (security-scope note on `writeMarker`),
`Sources/MLXEngineUI/ModelStorageSettingsView.swift` (bookmark entitlements in the doc comment),
`mlx-engine-swift/docs/getting-started.md`, and the engineering CLAUDE.md "Offline (CLI) vs live (Xcode
app) — the metallib boundary" section.
