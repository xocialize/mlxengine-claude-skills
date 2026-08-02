---
name: realitykit-avatar-macos
description: Architect a real-time animated 3D character (USDZ/VRM avatar) viewer on macOS (Apple Silicon, macOS 15+/26/27) with RealityKit — animation driving, offscreen render-to-MTLTexture for Metal compositing, and frame pacing under concurrent GPU load. Use when a RealityKit avatar viewer is laggy or stuttering; when consolidating per-frame skeletal/blendshape writes; when deciding manual SkeletalPosesComponent writes vs native playAnimation/AnimationLibraryComponent vs IKComponent; when compositing a RealityKit character over/under image/video layers in an existing Metal/CAMetalLayer pipeline (RealityRenderer offscreen → MTLTexture, or ARView.renderCallbacks.postProcess); when a per-frame RealityKit System on the MainActor contends with Metal compute (MLX LLM/TTS inference); or when weighing RealityKit vs a custom Metal renderer vs SceneKit for a blendshape+skeleton+springbone character. Trigger phrasings — "RealityKit lag/stutter", "avatar viewer performance", "SkeletalPosesComponent every frame", "consolidate pose writes", "RealityRenderer", "render RealityKit to a texture", "composite RealityKit into Metal", "postProcess render callback", "RealityKit vs custom Metal renderer", "VRM/USDZ character on macOS", "lip-sync blendshapes per frame", "RealityView vs ARView macOS". Complements the app-level MLXCompanion avatar (AvatarRig/AvatarDriveSystem). NOT for visionOS-specific spatial UI, nor for the MLX-engine side (see mlx-swift-integration / mlxengine-implementation).
---

# RealityKit avatar viewers on macOS — animation, offscreen compositing, frame pacing

For a skinned USD/VRM character (skeleton + blendshapes + springbone physics) rendered in real
time on macOS while the GPU is also doing Metal compute (e.g. MLX inference). Grounded in Apple
docs, WWDC24 §10102 / WWDC21 §10075 / WWDC25 §288, and SDK-verified API availability (macOS 27 SDK,
2026-07). Reference implementation: MLXCompanion `AvatarRig`/`AvatarDriveSystem`.

## The decision (start here)

**Keep RealityKit. Do not rewrite to a custom renderer or SceneKit.** All three concerns — skeletal
animation, blendshapes, offscreen Metal compositing — are first-class, non-deprecated Apple APIs on
macOS 15+. The lag is almost always an *update-architecture* problem, not a stack problem.

```
Symptom                                        →  Fix (in priority order)
────────────────────────────────────────────────────────────────────────────────
Laggy / stutter, avatar only on screen         →  (1) Consolidate to ONE pose commit +
                                                      ONE blendshape commit per frame
                                                   (2) Move baked clips to native playAnimation
                                                   (3) Profile MainActor tick cost
Need to composite over/under video/image layers →  RealityRenderer offscreen → your MTLTexture
in an existing Metal/CAMetalLayer compositor        (content code carries over unchanged)
Staying on-screen but need a Metal grab hook    →  ARView.renderCallbacks.postProcess (ARView-only)
Character render vs MLX compute fight for GPU   →  Caller-driven RealityRenderer + MTLEvent fences;
                                                      command-queue priority; cap render Hz
```

Details: **[references/animation.md](references/animation.md)** (driving the skeleton + face),
**[references/offscreen-compositing.md](references/offscreen-compositing.md)** (RealityRenderer /
postProcess / Metal integration), **[references/frame-pacing-gpu.md](references/frame-pacing-gpu.md)**
(pacing + GPU contention + stack comparison).

## The #1 fix: one pose commit + one blendshape commit per frame

Apple's demonstrated cadence (WWDC24 §10102) is **at most one** consolidated `SkeletalPosesComponent`
write per frame — "This can be updated at most every frame." A viewer that runs several procedural
layers (baked clip sample → springbone → arm keep-out / IK → breathing idle) and does a separate
`entity.components.set(SkeletalPosesComponent…)` **per layer** is doing 2–4 component-set round-trips
per frame per skeleton. Each set crosses the Swift↔RealityKit-core boundary and re-uploads the pose.

**Fix:** chain the layers through one shared `[Transform]` working buffer, then commit once.

```swift
// tick(), once per frame:
var pose = restLocals                 // or read the current default pose once
sampleBakedClip(into: &pose, at: t)   // base layer
applyBreathing(&pose, at: t)          // additive idle
applyArmKeepOut(&pose)                // correction
applySpringbones(&pose, at: t)        // physics reacts to the FINAL body pose
commitPose(pose)                      // ← the ONE components.set this frame

func commitPose(_ p: [Transform]) {
    guard var c = entity.components[SkeletalPosesComponent.self],
          let id = c.poses.default?.id else { return }
    c.poses.default?.jointTransforms = JointTransforms(p)   // in-place; avoids the
    entity.components.set(c)                                 // per-frame SkeletalPose
}                                                            // reconstruction leak
```

This also fixes a correctness bug the multi-write pattern hides: **springbones should react to the
final body pose**, so they must run last in the chain, not as an independent write that races the
others. The blendshape side is already correct if it does one batched `weightSet` mutation per frame
(blink+gaze+visemes+emotion composed in memory first) — see animation.md.

## Non-obvious, load-bearing facts (SDK-verified, macOS 27)

- **`RealityRenderer` is native macOS 15.0+** (not visionOS-only). Abstract: *"a renderer that
  displays a RealityKit scene in an existing Metal workflow."* Caller drives it —
  `updateAndRender(deltaTime:cameraOutput:whenScheduled:onComplete:actionsBeforeRender:actionsAfterRender:)`
  renders into **your** `MTLTexture` via `CameraOutput.Descriptor.singleProjection(colorTexture:)`,
  with `MetalEventAction.wait/.signal(_:value:)` fences to sequence against other GPU work. The full
  SkeletalPoses/BlendShape content stack migrates **unchanged**; only the System tick becomes
  caller-scheduled.
- **`ARView.renderCallbacks.postProcess` IS available on native macOS AppKit** (SDK-verified in the
  arm64e-apple-macos interface; the web docs' iOS-only platform badge is misleading). It exposes
  `sourceColorTexture` / `targetColorTexture` / the active `MTLCommandBuffer`. **ARView-only, not
  RealityView**, and the closure runs on RealityKit's render thread (not MainActor).
- **SceneKit is soft-deprecated** (WWDC25 §288, macOS 26 SDK — critical-bug-only). Not a candidate;
  the VRM ecosystem itself is migrating SceneKit→RealityKit (VRMKit deprecates VRMSceneKit).
- **`IKComponent`** (full-body IK solver, macOS 15+) can replace a hand-rolled arm keep-out layer —
  fit permitting.
- **A custom pure-Metal renderer is viable but a large net-new bet** with thin validation
  (VRMMetalKit: GPU XPBD springbones @120Hz, 256-joint skinning, GPU morph targets — but ~1-week-old,
  self-reported production status, no independent benchmarks). Reach for it only if RealityKit fixes
  are exhausted AND GPU-compute isolation from MLX is the binding constraint.
- **Authoring a generated (vertex-colored, material-less) mesh to USDZ — three attributes RealityKit
  needs, each a distinct failure if omitted** (verified 2026-07-12 on TRELLIS→USDZ; a "holes" report
  is usually one of these, NOT missing geometry):
  1. **`doubleSided = true`** (a `UsdGeomGprim` attr, independent of any material). Without it, concave
     thin-shell geometry (bell sleeves, tiered skirts) is back-face culled — you see through to the
     background = apparent *holes*. The single most common false "hole".
  2. **Explicit per-vertex `normals`** (`CreateNormalsAttr` + `SetNormalsInterpolation(.vertex)`; the
     mesh handle must be `var` — `SetNormalsInterpolation` is mutating). Omit them and USD/RealityKit
     recompute normals from face winding; generated meshes with inconsistent winding then shade random
     faces **black** ("black patches" on otherwise solid surfaces). Carry the source glTF `NORMAL`
     through — don't pass `normals: nil`.
  3. **`primvars:displayColor`** (vertex interp) for color, since these meshes bind **no** material —
     RealityKit renders displayColor only when nothing is bound (a bound material overrides it, and
     Apple viewers drop glTF `COLOR_0` the moment any material is attached).
  Diagnose by inspecting the authored USD: `usdcat foo.usdc | grep -E 'doubleSided|normals|displayColor'`.
  Deeper root for the winding/topology mess (black patches + non-manifold) is that TRELLIS *decimates*
  rather than *remeshes* — authored normals are the correct workaround; remeshing is the structural fix.

## Migration path (lowest risk → highest)

1. **Consolidate writes** (above) — one pose commit + one blendshape commit per frame. Biggest win,
   smallest change, zero API migration.
2. **Adopt native clip playback** for the baked base layer: `AnimationLibraryComponent` +
   `playAnimation(_:transitionDuration:)` gives engine-driven cross-fade for free; keep procedural
   layers as additive writes on top. (animation.md)
3. **Evaluate `IKComponent`** to retire the custom keep-out layer.
4. **Only if compositing or GPU-contention demands it:** move to `RealityRenderer` offscreen into the
   existing CAMetalLayer pipeline. Content code is unchanged; you gain a caller-owned render loop and
   MTLEvent control points against MLX. (offscreen-compositing.md, frame-pacing-gpu.md)

> Living skill — extend as avatar/rendering gaps surface. Source of truth for API availability is the
> installed SDK's `.swiftinterface` (grep the `arm64e-apple-macos` interface), not web docs whose
> platform badges lag or mislead.
