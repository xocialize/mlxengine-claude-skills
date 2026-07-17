# Springbone / cloth physics

The `VRMC_springBone` data model, how `@pixiv/three-vrm` simulates it, how to synthesize bones for
generated garments/hair that have none, and how to convert VRChat PhysBone data (which XWear
stores). This is the "make the skirt and hair move, and stop it clipping" reference.

## Table of contents
- [Data model: chains, joints, colliders](#data-model-chains-joints-colliders)
- [Per-joint parameters](#per-joint-parameters)
- [Colliders and collider groups](#colliders-and-collider-groups)
- [The three.js runtime simulation](#the-threejs-runtime-simulation)
- [Synthesizing bones for a generated garment](#synthesizing-bones-for-a-generated-garment)
- [Stopping clipping (body colliders)](#stopping-clipping)
- [Tuning by cloth type](#tuning-by-cloth-type)
- [PhysBone → VRM conversion (XWear)](#physbone--vrm-conversion)
- [Auto-rigging tools](#auto-rigging-tools)

## Data model: chains, joints, colliders

`VRMC_springBone` organizes physics as **spring chains**. A chain is an **ordered array of glTF
nodes** in a parent→descendant hierarchy: `joints[n]` must be a parent or ancestor of `joints[n+1]`.

Consecutive nodes form **head–tail pairs**: the **head** joint rotates, and the **tail** (the next
node's position) is what actually swings, gets pulled by gravity, and collides. The **terminal
node** of a chain is only a tail position — it carries no parameters (it just needs to exist as a
node so the previous joint has a tail to track).

Chains are processed **root-to-descendants** — an ancestor joint updates before its children, so
motion propagates down the chain (shoulder → mid → tip of a ponytail).

Mental model for a skirt: each vertical "rib" of the skirt is one chain of 2–4 nodes running from
the waist down to the hem. The skirt mesh is skinned to these nodes; when they swing, the geometry
follows.

## Per-joint parameters

Each non-terminal joint carries:

| Field | Range | Meaning |
|---|---|---|
| `stiffness` | float ≥ 0 | Rigidity — force returning the joint toward its rest orientation. Higher = holds its shape, snaps back faster. |
| `dragForce` | `[0, 1]` | Deceleration / damping. Higher = more air resistance, less overshoot/wobble. |
| `gravityPower` | float | Gravity magnitude applied every frame. |
| `gravityDir` | vec3 | Gravity direction, canonically `[0, -1, 0]`. (Point it elsewhere for wind-like or stylized pull.) |
| `hitRadius` | float (m) | Radius of the joint's collision sphere — how "thick" the bone is for collision. |

`center` (optional, on the spring) — a node defining the reference space for inertia. Setting it to
a body node makes physics ignore the character's own locomotion (so a skirt doesn't fling out when
the avatar walks); leaving it null uses world space.

## Colliders and collider groups

Two collider shapes, defined in the **local coordinates of the node they attach to**:
- **Sphere** — `offset` (local center) + `radius`
- **Capsule** — `offset` (start point) + `radius` + `tail` (end point)

(Base VRM 1.0 has sphere + capsule only. `@pixiv/three-vrm` also implements a **Plane** collider,
but that belongs to the `VRMC_springBone_extended_collider` extension, not base 1.0 — don't emit
Plane colliders if you need portability.)

Colliders are bundled into **named collider groups**. A spring chain lists which collider **groups**
it interacts with — **a joint only collides with the groups its chain references.** This is the
crux of stopping clipping (below): the collider existing isn't enough; the skirt's chain must point
at it.

## The three.js runtime simulation

`@pixiv/three-vrm`'s `three-vrm-springbone` module: a central **`VRMSpringBoneManager`** holds the
joints/colliders/collider groups and drives them via **`update(deltaTime)`** each frame; per-joint
state lives in **`VRMSpringBoneJoint`**; colliders are `VRMSpringBoneColliderShape` subclasses
(Sphere, Capsule, Plane).

The per-frame update is **position Verlet** on each joint's tail:

```
inertia   = (currentTail − prevTail) × (1 − dragForce)      // momentum, damped
stiffness = rotate toward rest orientation (× deltaTime)     // restoring force
gravity   = deltaTime × gravityDir × gravityPower            // external force
nextTail  = currentTail + inertia + stiffness + gravity      // summed in one step
→ enforce bone-length constraint (keep tail at fixed distance from head)
→ resolve collisions against the chain's collider groups (push tail out by hitRadius + collider radius)
→ derive the joint's node quaternion from head→nextTail vs the rest pose
prevTail  = currentTail ; currentTail = nextTail            // velocity is implicit in history
```

Velocity is never stored explicitly — it's the difference between successive tail positions. The
three force terms are **summed in a single Verlet step**, not applied sequentially.

You rarely call this yourself; you call `vrm.update(delta)` in the render loop and three-vrm drives
the manager. Your job is authoring correct chains/colliders in the data.

## Synthesizing bones for a generated garment

A TRELLIS-generated skirt/hair mesh arrives with **no bones**. To give it physics you procedurally
author the chains:

**Skirt / cloak (sheet cloth):**
- Slice the hem into N vertical **strips** (e.g. 8–16 around a skirt; more = smoother, costlier).
- Each strip = one chain of **2–4 joints** from the waistband down to the hem. 2 for a short/stiff
  skirt, 3–4 for a long flowing one.
- Skin the strip's vertices to its chain (top verts → root joint, hem verts → tail), with falloff.
- Root joints are (near-)static; motion increases toward the hem.

**Hair (strands):**
- One chain **per strand/lock**, root at the scalp, 2–5 joints down the length (longer/looser hair
  → more joints).
- Bangs: short stiff chains; ponytail/twintails: longer chains with lower stiffness.

**Sleeves:** short chains hanging from the elbow/cuff, 2–3 joints; only worth it for loose/flared
sleeves.

Joint count is a cost/quality dial — every joint is per-frame work. Start low and add only where
motion looks stepped.

> There is no verified spec standard for strip/strand density or joints-per-chain — the numbers
> above are informed convention. Tune against real renders.

## Stopping clipping

A skirt clips the legs when its joints have nothing to collide with, or collide with the wrong
group. To fix:

1. **Place body colliders** — spheres/capsules on the pelvis/hips and upper legs (capsules down the
   thighs work well), sized to the body silhouette. Attach them to the relevant humanoid nodes so
   they move with the legs.
2. **Group them** — put those colliders in a named collider group (e.g. `"LowerBody"`).
3. **Reference the group from the skirt chains** — each skirt chain must list that collider group.
   *This is the step people miss:* the collider existing does nothing unless the chain points at it.
4. **Set `hitRadius`** on the skirt joints so the cloth surface sits outside the leg colliders
   (thicker hitRadius = more clearance, but too much makes the skirt stand off oddly).

Same pattern for hair vs shoulders/chest (chest/shoulder colliders in a group the hair chains
reference) and for a cloak vs the back/arms.

## Tuning by cloth type

Parameter **meanings** are spec-confirmed; the concrete **value ranges** below are convention —
starting points, not sourced presets. Tune with real motion.

- **Stiff skirt (pleated/structured):** higher `stiffness`, higher `dragForce` (less wobble),
  moderate `gravityPower`. Holds shape, small swing.
- **Flowing skirt (light fabric):** low `stiffness`, low–mid `dragForce`, moderate `gravityPower`.
  Large, slow swing.
- **Hair:** mid `stiffness` (keeps silhouette), mid `dragForce`, mid `gravityPower`. Bangs stiffer
  than a ponytail.
- **Cloak / cape:** low `stiffness`, higher `dragForce` (heavy fabric damps fast), higher
  `gravityPower` (weight). Big lazy motion.

General intuition: `stiffness` = how hard it returns to rest; `dragForce` = how fast wobble dies;
`gravityPower` = how heavy it feels; `hitRadius` = collision thickness/clearance.

## PhysBone → VRM conversion

XWear stores physics as VRChat-style **PhysBone** components (`PhysBoneParam`). Converting to
`VRMC_springBone` is a **heuristic mapping**, not exact — VRM SpringBone and PhysBone use different
integration math, so results won't match perfectly and need hand-tuning after.

**Proven default parameter mapping** (esperecyan's converter — byte-sourced; multipliers are that
tool's overridable defaults, targeting VRM 0.x `VRMSpringBoneParameters` but the concepts carry to
1.0):

```
VRM stiffness / StiffnessForce = PhysBone Pull    × 4.0
VRM dragForce  / DragForce     = PhysBone Spring   (1 : 1)
VRM gravityPower / GravityPower = PhysBone Gravity × 20.0
VRM hitRadius                  = PhysBone Radius   (direct, meters)
```

**⚠️ Two name traps — do not map by matching names:**
1. VRM `stiffness` maps from PhysBone **`Pull`**, *not* PhysBone `Stiffness`. VRChat's Advanced
   integration "Stiffness" (stay-at-rest) is a **different concept**; the default converter reads it
   but doesn't use it in the output.
2. PhysBone integration types differ: **Simplified** uses `Pull` + `Spring`; **Advanced** uses
   `Pull` + `Stiffness` + `Momentum`. Map from `Pull`/`Spring`, ignore Advanced-only fields unless
   you build a smarter converter.

**PhysBone specifics worth knowing:**
- `Gravity` is a **signed** value: positive pulls **down**, negative pulls **up**. Plus a
  `Gravity Falloff` (0–1) reducing gravity at rest (1.0 = no gravity in rest pose). VRM has no
  direct falloff analog — approximate or drop it.
- `Radius` = collision radius per bone in meters = direct `hitRadius` analog.

**Collider conversion** (PhysBone collider → VRM sphere/capsule):
- A **capsule** whose `height > 2 × radius` becomes **3 spheres** along the axis: center, and
  center ± `(height − 2·radius)/2`. (VRM capsule could represent it directly, but the reference
  converter decomposes to spheres.)
- Otherwise → **1 sphere**.
- **Plane** colliders **cannot be converted** — the reference tool warns and emits a degenerate
  sphere. Avoid relying on Plane colliders through this path.

The `ndmf-vrm-exporter` tool likewise converts PhysBone components into `VRMC_springBone` joints on
export — evidence PhysBone is a normal *source* for VRM spring authoring, not an incompatible system.

## Auto-rigging tools

**UniRig** — a neural auto-rigging framework that predicts a **skeleton hierarchy + per-vertex skin
weights** from an input mesh (autoregressive GPT-like transformer, skeleton-tree tokenization,
Bone-Point Cross Attention). Directly relevant to the **SkinTokens garment-rig mode** gap (skinning
a generated garment to a skeleton).

**But:** UniRig's prediction of **physics attributes** (e.g. per-bone stiffness for secondary
motion) was **"Coming Soon" / unreleased** as of ~2026-07. So even with UniRig producing a
skeleton, the spring **parameters** still need heuristics or hand-tuning (use the cloth-type table
above). Re-check UniRig for a physics-attribute release before assuming it's available.

## Authoritative sources (VRMC_springBone-1.0 spec + three-vrm), verified 2026-07-17

Spec: `vrm-c/vrm-specification/.../VRMC_springBone-1.0`. three-vrm impl + guides.

**Coordinate space / units (from the spec):**
- Joint positions simulate in **WORLD** coordinates; collider offsets/shapes are in the target
  node's **LOCAL** coordinates. All distances in **meters**.
- Sphere collider = offset + radius (local). Capsule = offset (start) + tail (end) + shared radius.
- Simulation = **Verlet**, three forces: inertia (prev velocity × `(1 − dragForce)`), stiffness
  (restore toward rest), gravity (`gravityPower` along `gravityDir`). Root→descendants order:
  compute next tail → constrain bone length → resolve collisions → update rotation.

**`center` node (locomotion-inertia control):**
- `center` MUST be the chain's **0th joint or an ancestor** of it.
- **Inertia is evaluated in center-space**; **gravity is always world-space**. Setting center on a
  moving parent makes the chain "shake too much when walking/running" go away.
- For HAIR: **center = Head** (rides head motion, no locomotion whip). For skirts: center = Hips.
  This validates the T3.4 profiles.

**⚠ Scale gotcha (three-vrm `spring-bones-on-scaled-models`): spring bones do NOT auto-scale.**
When a model or its bones are scaled by `s`, you MUST manually scale params or the hair reacts
too fast/slow and colliders misbehave:
```js
for (const j of vrm.springBoneManager.joints) { j.settings.stiffness *= s; j.settings.hitRadius *= s; }
for (const c of vrm.springBoneManager.colliders) { c.shape.radius *= s; c.shape.tail?.multiplyScalar(s); }
```
**Relevance to imported hair:** the compatibility gate admits donors at 82–120% of base height. If
imported-hair springs are authored at the donor's scale (approach A, preserve source rig) and worn
on girl-base, multiply `stiffness`/`hitRadius` and collider `radius`/`tail` by (baseScale/donorScale)
— OR author/synthesize the springs directly in girl-base's metric scale (approach B, `synthesize_hair`,
already in base meters → no correction needed). This is a decision input for the imported-hair-physics
work: synthesizing in base scale sidesteps the scale-correction entirely.

**SkinTokens** (VAST-AI-Research, MIT, arXiv 2602.04805) — the auto-rigging engine for our
**garment-rig lane (T3.3)**. GLB mesh in → **skeleton + per-vertex skin weights** out (FSQ-CVAE
weight tokenizer + grammar-constrained beam TokenRig on a Qwen3-0.6B backbone).
**✅ ALREADY PORTED TO SWIFT-MLX (on-device, NOT CUDA)** — `~/Development/mlxengine-3d/DEV/SkinTokensDev`,
package `mlx-skintokens-swift`, engine capability `meshRig`; memory `skintokens-autorig-candidate`
(mlxengine-3d scope). P0 Python oracle (CPU, SDPA-patched) + P1 Swift-MLX core done: S0→S2b parity
ladder PASSED and **first END-TO-END done 2026-07-10 — real GLB → rigged GLB**. Pipeline: SamplerMix
54k pts → Michelangelo encoder → SkinVAE cond → beam(10) TokenRig → FSQ decode → per-point skin →
**cKDTree propagate to the ORIGINAL glTF verts → inject JOINTS_0/WEIGHTS_0** (vertex identity
preserved — fits our extract/wear convention directly). **Route B conditions on OUR VRoid skeleton**
(`configs/skeleton/vroid.yaml` = the J_Bip bone list; `vroid` is a TRAINED cls token) → predicts
weights for the girl-base/N00 template = exactly what T3.3 needs. So the on-device neural garment
rigger EXISTS (complements the proximity-transfer path P2, likely exceeds it for OOD garments).
**Boundary:** skeleton + weights only — **NO physics / spring bones**, so spring params still come
from the cloth/hair profile tables above, and it does NOT help hair physics (head-anchor /
preserved-source-springs need no weight predictor).
