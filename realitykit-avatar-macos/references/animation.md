# Driving a RealityKit character: skeleton + face (macOS 15+)

Sources: WWDC24 §10102 "Compose interactive 3D content"; Apple docs for `SkeletalPosesComponent`,
`BlendShapeWeightsComponent`, `AnimationLibraryComponent`, `Entity.playAnimation`, `IKComponent`.
All verified 3-0 in adversarial checking. Platform: macOS 15.0+ (no deprecation).

## Skeleton — three sanctioned routes, one commit rule

Any skinned mesh imported from USD gets a `SkeletalPosesComponent` automatically. Apple names three
ways to drive it, in ascending manual-ness:

1. **Baked clips** — `AnimationResource` via `playAnimation` (engine evaluates the pose; see below).
2. **Manual pose** — read/write `jointTransforms` yourself from a registered `System.update` (or
   `RealityView`-driven tick). Apple's own WWDC24 demo mutates `jointTransforms` every frame; the
   docs explicitly sanction *"write your own pose to animate the skeletal model."* This is a
   first-class pattern, **not** a hack.
3. **IK** — `IKComponent` (full-body solver, per-joint constraints).

**The cadence rule (load-bearing):** *"This can be updated at most every frame."* One consolidated
pose write per frame per skeleton. Multiple `entity.components.set(SkeletalPosesComponent…)` calls in
a single frame (one per procedural layer) is the classic lag/stutter cause — each set crosses the
Swift↔core boundary and re-uploads the whole pose. Compose all layers in a `[Transform]` buffer, set
once. See SKILL.md "The #1 fix".

`SkeletalPosesComponent` is a `SkeletalPoseSet` (named poses, `Collection`-conforming, with a
`.default` pose and `BindTarget.skeletalPose(_:)`). The named slots are **not** documented as
procedural blend layers — don't expect them to auto-blend your idle over your clip. Blending is your
job (compose in the buffer) or the engine's (playAnimation transitions).

### In-place mutation avoids a per-frame leak

Rebuilding the pose each frame — `SkeletalPose(id:joints: Array(zip(names, xforms)))` — leaks under
RealityKit's per-frame reconstruction. Mutate `jointTransforms` in place instead:

```swift
comp.poses.default?.jointTransforms = JointTransforms(local)  // reuse names + id
entity.components.set(comp)
```

## Native clip playback (replaces manual baked-clip sampling)

If your base layer is a baked humanoid clip you sample by hand every frame, the native path removes
that work and gives you cross-fade for free:

- `AnimationLibraryComponent` (macOS 15+) — string-keyed `AnimationResource`s per entity:
  `animations["wave"]`, `defaultAnimation`, `automaticallyPlaysDefaultAnimation`.
- `entity.playAnimation(_:transitionDuration:blendLayerOffset:separateAnimatedValue:startsPaused:clock:handoffType:)`
  → `AnimationPlaybackController`. `transitionDuration` gives engine-driven cross-fade between clips
  with **no manual per-frame pose writes**. `blendLayerOffset` lets a second `playAnimation` blend on
  a separate layer.

**Pattern for a companion avatar:** let `playAnimation` own the base locomotion/gesture clip
(cross-fading between clips), and keep procedural layers (springbone, breathing, keep-out) as
additive writes composed on top in your one-commit tick. Caveat: an old forum report (Apple DTS
684833) noted a one-frame glitch with nonzero `transitionDuration` on some OS versions, reportedly
fixed — verify on your target OS.

**Fighting caveat:** don't drive the *same* joints with both `playAnimation` and hand-written pose
writes in the same frame — they fight. Partition: engine owns base clip joints; you own the
procedural-only joints, or you own everything (manual) — not both on the same joint.

## Face / blendshapes (57-shape lip-sync + emotion)

`BlendShapeWeightsComponent` (macOS 15.0+, weights 0–1) supports three drive paths:
procedural `FromToByAnimation`/`SampledAnimation`, USD blendshape animation via `playAnimation`, and
**direct per-frame `weightSet` mutation** — the right one for audio-envelope→viseme lip-sync.

Sanctioned flow: add the component **once** (initialized from a `BlendShapeWeightsMapping`), then
mutate weights through `weightSet` each frame — do **not** rebuild the component:

```swift
entity.components[BlendShapeWeightsComponent.self]!.weightSets[0].weights[i] = w
```

One batched write per frame with all 57 weights (blink + gaze + visemes + emotion) composed in memory
first **matches Apple's documented pattern** — this is already correct if you do it that way. Same
fighting caveat as skeleton: don't mix `playAnimation`-driven and hand-written weights on the *same*
shapes.

### Lip-sync latency

Audio-envelope→viseme driving is latency-sensitive but the per-frame `weightSet` mutation is cheap
and on the render cadence — the architecture doesn't bottleneck it. The latency budget lives in the
audio→envelope→weight mapping (keep it O(1) per frame, no allocation), not in RealityKit. The
research surfaced **no** architecture that beats direct `weightSet` mutation for this; it's the
recommended path regardless of onscreen vs offscreen rendering.

## IKComponent — retiring a custom keep-out layer

`IKComponent` + `IKResource`/`IKRig` solves full-body IK with per-joint constraints over the whole
skeleton at once (macOS 15+). A hand-rolled "keep hands out of the skirt / un-cross arms" correction
*may* map onto target-driven IK constraints — but avoidance/keep-out is not the same shape as
reach-a-target IK, so treat this as an engineering-fit spike, not a guaranteed swap. If it fits, it
removes a procedural layer (and its pose write) from your tick.
