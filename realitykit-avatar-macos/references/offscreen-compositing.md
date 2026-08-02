# Offscreen RealityKit → MTLTexture, and Metal compositing (macOS 15+)

For putting a RealityKit character into an existing Metal/CAMetalLayer compositor, over/under image
and video layers (CVPixelBuffer→MTLTexture sources). Two routes; RealityRenderer is the recommended
one. Both SDK-verified in the native `arm64e-apple-macos` interface (macOS 27 SDK).

## Route A (recommended): RealityRenderer — offscreen into your MTLTexture

Apple's purpose-built API. Abstract: *"A renderer that displays a RealityKit scene in an existing
Metal workflow."* Available **macOS 15.0+** (iOS 18, visionOS 1.0, tvOS 26) — non-beta,
non-deprecated. This is the right tool for the compositing scenario.

### Why it fits

- **Renders into your texture.** `CameraOutput.Descriptor.singleProjection(colorTexture:)` takes an
  `MTLTexture` you own — hand it a texture in your compositor's pool and RealityKit draws the
  character into it. `cameraSettings.colorBackground` can be a solid `.color(cgColor)` **or**
  `.outputTexture()` (render onto prepared/transparent content for over/under compositing).
- **You own the loop.** `updateAndRender(deltaTime:cameraOutput:whenScheduled:onComplete:actionsBeforeRender:actionsAfterRender:)`
  ticks the RealityKit simulation *and* renders in one call. `update(_ deltaTime:)` ticks simulation
  without rendering (e.g. to sub-step physics). Drive it from your compositor's clock (CVDisplayLink),
  not RealityKit's.
- **GPU sequencing hooks.** `MetalEventAction.wait(for:value:)` / `.signal(_:value:)` on an `MTLEvent`,
  passed as `actionsBeforeRender` / `actionsAfterRender`, let you fence the character render against
  other GPU work in the same frame (your video decode/upload, or MLX compute) — explicit ordering
  instead of hoping the driver interleaves well.
- **Content carries over unchanged.** *"All RealityKit APIs for loading resources, creating entities
  and adding components are compatible."* Your `SkeletalPosesComponent` / `BlendShapeWeightsComponent`
  character code migrates as-is. The only structural change: a registered `System`'s per-frame tick
  becomes **caller-scheduled** — you call your `tick()` (compose pose + blendshapes) right before
  `updateAndRender`, since there's no automatic scene-update loop when you own the renderer.
- Camera via an `activeCamera` Entity (a `PerspectiveCamera`); IBL via `lighting` (`ImageBasedLight`
  with `EnvironmentResource` + `intensityExponent`); `cameraSettings.antialiasing`, tone-mapping,
  and (macOS-only) EDR knobs. macOS 27 adds an `audioListener` entity.

### Shape of the integration

```swift
let renderer = try RealityRenderer()
renderer.entities.append(characterRoot)          // your loaded avatar
renderer.activeCamera = cameraEntity
renderer.lighting.resource = envResource

// per compositor frame (on your CVDisplayLink):
avatarTick(dt)                                    // compose ONE pose + ONE blendshape set
let out = try RealityRenderer.CameraOutput(.singleProjection(colorTexture: characterTex))
try renderer.updateAndRender(
    deltaTime: dt,
    cameraOutput: out,
    actionsBeforeRender: [.wait(for: videoUploadEvent, value: frameID)],
    actionsAfterRender:  [.signal(characterReadyEvent, value: frameID)])
// then your compositor blends characterTex over/under the video/image MTLTextures
```

`characterTex` is now a normal `MTLTexture` in your CAMetalLayer pipeline — composite it exactly like
your `CVPixelBuffer→MTLTexture` video sources. This is the path that fits an existing Metal
compositing workflow with zero change to how you blend layers.

### Gotchas

- No automatic System scheduling — **you** must tick per-frame behavior before each render. Miss it
  and the character freezes.
- Texture format/usage must match your compositor (`.renderTarget` usage; matching pixel format).
- Simulation determinism: pass a real `deltaTime`; springbone physics stepped via `update(_:)` wants
  a fixed sub-step accumulator (same as onscreen).

## Route B (onscreen): ARView.renderCallbacks.postProcess

If you stay **onscreen** but need a Metal hook to grab/composite the rendered frame:
`ARView.renderCallbacks` — `prepareWithDevice` (once; make textures/pipelines) and `postProcess`
(every frame). The `PostProcessContext` exposes `sourceColorTexture` (the fully rendered scene as an
`MTLTexture`), `targetColorTexture` (what your callback must fill), and the active `MTLCommandBuffer`.
WWDC21 §10075 demos compositing via Core Image, MPS, custom compute kernels, and SpriteKit overlays.

Constraints:
- **ARView-only, not RealityView.** The companion uses `RealityView`; to use postProcess you'd host an
  `ARView` in an `NSViewRepresentable` instead.
- The closure runs on **RealityKit's render thread**, not the MainActor — don't touch `@MainActor`
  state from it without hopping.
- **Native-macOS AppKit support is real** — SDK-verified in the `arm64e-apple-macos` RealityKit
  interface (`ARView.RenderCallbacks.postProcess` at ~line 363). The web docs' iOS-only platform
  badge is misleading; the earlier "iOS-only" claim was refuted 0-3 in verification, and the SDK
  confirms availability. (It is *ARView* that is the constraint, not the platform.)

## When to use which

- **Compositing into an existing Metal/CAMetalLayer pipeline** (video/image layers, your own blend
  passes) → **RealityRenderer**. It's built for exactly this, gives you the texture and the loop, and
  keeps your content code intact.
- **Onscreen RealityKit view + a post-effect or frame grab**, and you're OK hosting an ARView →
  postProcess.
- **Just showing the character with no compositing** → stay on `RealityView`; you don't need either.
