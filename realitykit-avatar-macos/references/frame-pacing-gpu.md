# Frame pacing, GPU contention with MLX, and the stack comparison

## Keeping the viewer smooth while MLX compute shares the GPU

The companion runs MLX (Metal compute) LLM/TTS inference intermittently on the same GPU as the
RealityKit render. Bursty compute that isn't sequenced against the render is a prime stutter cause.
The web research surfaced no single silver-bullet doc here (the frame-pacing angle yielded fewer
hard-verified claims than the API angles), so this is engineering guidance, ordered by leverage:

1. **Own the render clock, sequence explicitly.** The strongest lever is Route A
   (RealityRenderer offscreen, see offscreen-compositing.md): a caller-driven loop on your
   `CVDisplayLink` plus `MetalEventAction.wait/.signal` fences lets you *order* the character render
   relative to MLX work in the frame instead of leaving it to the driver. This is the biggest reason
   to move off the auto-scheduled `RealityView` loop once contention bites.
2. **Cap the render cadence.** A conversational avatar rarely needs 120 Hz. Rendering at 30–60 Hz
   (skip frames on your CVDisplayLink) leaves GPU headroom for inference bursts and is visually fine
   for talking-head + idle motion. RealityKit's auto loop gives you less control here than a
   caller-owned loop.
3. **Don't fight on the MainActor.** The per-frame `tick()` runs on the MainActor. Keep it cheap:
   O(joints) pose composition, no allocation, no `await`. Heavy work (physics broadphase, envelope
   analysis) should be precomputed or off-actor, with only the final `[Transform]`/weight buffers
   applied on the MainActor. A slow tick stalls the whole UI, not just the avatar.
4. **Command-queue separation.** MLX uses its own Metal command queue; RealityKit uses its own. They
   already don't share a queue — the contention is for GPU *time*, not the queue object. Priority
   hints exist but are advisory; explicit `MTLEvent` fencing (lever 1) is the reliable control.
5. **Bound the MLX side too.** The engine's GPU buffer-cache cap (ENGINE-NEEDS N5, engine ≥0.21.0)
   keeps inference from ballooning resident memory during a render-heavy moment — a companion concern,
   not a RealityKit one, but it's part of the same "don't let compute starve the render" picture.

Measure before optimizing: a laggy avatar-only viewer is almost always the multi-write pose problem
(SKILL.md #1 fix), **not** GPU contention. Contention shows as stutter *correlated with* inference
bursts (a reply landing, TTS synthesizing) — if the lag is constant even when idle, it's the update
architecture.

### App-wide UI lag ≠ render lag (a distinct, common trap)

If the *whole app* feels sluggish — menus slow to open, seconds from a selection to it taking effect —
and not just the 3D view, the cause is usually **main-thread saturation**, not the renderer. The
RealityKit System `tick()` runs on the main thread every render frame; if it's heavy (e.g. multiple
full-skeleton FK passes) at 120 Hz, it starves SwiftUI event handling. Two cheap, high-leverage fixes,
verified on MLXCompanion (2026-07):
1. **Cap the tick to ~60 Hz.** `tick()` is invoked per render frame, but a talking companion doesn't
   need 120 Hz pose updates. Gate with a wall-clock accumulator (`guard now - last >= 1/60`); on a
   ProMotion display this halves the per-frame main-thread cost. Motion stays smooth (each pose shows
   for two display frames).
2. **Don't put a giant `Menu` in a monolithic body.** A `ForEach` over ~100 items inside a `Menu`,
   living in the same `body` as the RealityView + chat, rebuilds all items on every unrelated state
   change and is slow to open on a contended main thread — a classic "5–10 s from dropdown to
   activation." Replace with a lazy, searchable `List` in its own panel `View` so it (a) is lazy and
   (b) only re-renders on its own state. Splitting a monolithic view into isolated panel Views
   (each reading only its slice of an `@Observable` model) scopes SwiftUI invalidation so a chat
   keystroke doesn't re-evaluate the avatar or the animation list.
Instrument with `os_signpost` (`OSSignposter` interval around `tick()`/the procedural passes, an event
at clip activation) to confirm which of render cost vs queuing latency dominates before/after.

## Stack comparison (why RealityKit wins here)

| Stack | Status (macOS 26/27) | Fit for skeleton + 57 blendshapes + springbones | Verdict |
|---|---|---|---|
| **RealityKit** | Current, non-deprecated; macOS 15+ full API | SkeletalPoses + BlendShapeWeights + IKComponent + RealityRenderer all native | **Use it** |
| **SceneKit** | Soft-deprecated WWDC25 §288, critical-bug-only | Works, but no new features; VRM ecosystem is *leaving* it | **No** |
| **Custom Metal renderer** | Viable, demonstrated (VRMMetalKit) | GPU XPBD springbones @120Hz, 256-joint skin, GPU morphs >64 — technically covers it | **Only as last resort** |

- **SceneKit** is out: Apple soft-deprecated it (WWDC25 §288 "Bring your SceneKit project to
  RealityKit"), and the VRM community reflects the direction — VRMKit (actively maintained) marks
  `VRMSceneKit` deprecated in favor of `VRMRealityKit`. Don't start new work on it.
- **Custom pure-Metal renderer** (e.g. arkavo-org/VRMMetalKit, Apache-2.0, macOS 26+): a real,
  substantive VRM 1.0 renderer — springbones as XPBD Metal compute at fixed 120 Hz substeps with
  sphere/capsule colliders, GPU skinning to 256 joints, GPU morph targets (auto-selected >8 active,
  up to 64) that cover a 57-blendshape face. So the springbone layer *could* move entirely to GPU
  compute, freeing the MainActor. **But**: it's ~1 week past its 1.0 tag, 3 stars, self-reported
  "production," no published benchmarks, no third-party validation, and its own macOS app is
  "forthcoming." The claim that it was built for concurrent on-device AI-inference headroom was
  **refuted 0-3**. Treat it as a reference for *how* to build GPU springbones/skinning if you ever
  must, not as a drop-in — and only reach for a custom renderer after the RealityKit fixes are
  exhausted and GPU-compute isolation from MLX is proven to be the binding constraint.

## Bottom line

Keep RealityKit. Fix the update architecture first (one pose commit + one blendshape commit per
frame; native `playAnimation` for baked clips). Move to `RealityRenderer` offscreen when you need
Metal compositing or explicit GPU sequencing against MLX — the character code carries over unchanged.
Consider a custom Metal renderer only if, after all that, GPU contention with inference is still the
wall you're hitting.
