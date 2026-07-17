# Wear runtime — dressing items onto a base at runtime (three-vrm)

How CompanionCloset puts an item onto the loaded companion in the browser viewer
(`VRMDeveloper/src/three/createViewer.js` + `src/lib/`). This is the READ/wear side, distinct
from the authoring pipeline. Built + proven 2026-07-17.

## The three item tiers (same file, different rigging problems)

An imported avatar's contents split into fundamentally different handling — do NOT treat them
the same (the #1 source of "hair exploded / outfit backwards" bugs):

| Tier | Rigging | Extractor | Notes |
|---|---|---|---|
| **Garment** | multi-bone body skin, driven by the base rig | `extractItems.js` | tops/dresses/skirts/bottoms/shoes |
| **Hair** | donor hair spring bones, rest-corrected into base-head frame (rigid 1.0 fallback) | `hairAnchor.js` | source hair springs sit on the HEAD — the one bone whose base rest differs (permutation) → naïve wear explodes; corrected + re-baselined |
| **Accessory** | rigid 1.0 bind to ONE named anchor bone | `axf/anchor_item.py` | hats/brooches/ties; complex assemblies compose externally |

## The wear puppet (garments)

An extracted item is a standalone skinned GLB carrying a **full copy of the base node
hierarchy** (so its skin `joints` / spring `node` indices stay valid). Instead of rebinding,
the viewer **puppets** it: each frame, copy the base bone locals onto the item's name-matched
bones, then run the item's OWN spring manager for garment-only chains (hem/sleeves) the base
doesn't have. Worn items are children of `vrm.scene` (spin/centering carry them).

**This works only when the item already shares the base's coordinate space and rest pose** —
i.e. VRM 1.0 items and same-lineage extracts. It does NOT retarget. See the traps.

## Drop-to-closet import routing (`importDropped.js`)

Drop any wearable-bearing file → extract → closet (IndexedDB) → auto-wear:
- `.vrmxw` / `.glb` standalone wearable → the whole file is one item.
- `.vrmxa` / `.vrm` (VRM 1.0) outfitted avatar → `detectItems` → garments (puppet) + hair
  (head-anchor); a BARE avatar (no items) loads as the model instead.
- `.xwear` → VRoid zip; unpacking not wired (needs deobfuscator lane).

## Compatibility gate (`compatibility.js`)

"VRMs come in all shapes and sizes." The wear path does NOT scale, so a donor built too
differently distorts. Gate BEFORE extracting on the humanoid skeleton:
- **head-bone height vs base** (girl-base head Y ≈ 1.395) — the decisive signal. Doll 0.96 m
  (56%) and outliers → **incompatible**, marked with reason, not force-fit.
- **torso build ratio** (headY/hipsY ≈ 1.5) — catches non-standard humanoids.
- Bands: [0.90,1.12] fits · [0.82,1.20] marginal (warn) · outside → incompatible.

Diagnose with the **donor-reference method**: load the donor VRM on its own and capture
front/side/back (three-vrm renders it correctly, incl. rotateVRM0) as ground truth, THEN
compare the applied result. Never judge an applied item with no reference.

## Hair head-anchor (`hairAnchor.js`) — the traps that make it work

Bind ALL hair verts rigid (1.0) to a single joint (the head) so it can't come apart. Then two
corrections make placement + orientation right:

1. **Bind against the BASE's head rest frame, not the source's.** `IBM = inverse(baseHeadWorld)`.
   girl-base's rig is **NOT normalized — its head bind is an axis permutation** (X→Y, Y→Z, Z→X).
   Binding against the source head applies that permutation as a rotation and swirls the hair
   over the crown. (`BASE_HEAD_WORLD` is hardcoded for girl-base, the fixed base.)
2. **Translate verts by (baseHeadPos − sourceHeadPos)** so the hair sits on the base head while
   keeping its authored world orientation.

Result: `v(t) = headBone(t) · baseHeadRest⁻¹ · (v_src + Δ)` — at rest the hair keeps source
orientation at the base head; during animation it follows the head. Head-anchoring is source-rig-
independent (only needs the head bone) so it also absorbs cross-avatar head position automatically.

## Hair PHYSICS — preserving the donor's spring bones (`extractHairRigged`, F-drop3 complete)

`extractHairRigged` makes imported hair **sway** instead of tracking rigidly, browser-native, by
**preserving the donor's own hair spring bones** rather than collapsing to the single-joint rigid
bind (which stays the fallback when a donor has no hair springs). The recipe — every skin influence
must contribute exactly `T(Δ)` at rest so the mesh reproduces the rigid F-drop3 framing, then springs
deviate under motion:

- **Head-weighted verts** → IBM `= baseHeadWorld⁻¹ · T(Δ)` (the rigid anchor's IBM — absorbs the
  axis-permuted base head).
- **Hair-chain verts** → keep donor IBMs; re-author each chain **root's** local under the head to
  `baseHeadWorld⁻¹ · T(Δ) · bindWorld_root` (descendants keep donor locals — a pure translation
  preserves relative transforms), so every hair bone lands at `T(Δ)·bindWorld_i`.
- **Everything else** (back-hair verts weighted to body spring bones, etc.) → redirect to the head
  slot (rigid). No dependency on un-driven body bones ⇒ can't explode.
- Carry the donor `VRMC_springBone` hair chains **verbatim** (node indices stay valid via the
  full-hierarchy copy; tuning preserved), `center = null` (world space, like `hair.glb` — maximises
  head-turn swing), **no colliders** (`hair.glb` proves none needed; a synthesized skull sphere shoves
  close side-locks out).

**Two runtime pieces make it work (both in `createViewer.wearItem`):**
1. **Re-baseline at wear** — the springs' rest is captured at load in the donor head frame, but the
   puppet then drives the head to girl-base's permuted frame → first-frame jump (the original
   explosion). After the item is added, puppet-copy the base locals once, `root.updateWorldMatrix
   (true,true)`, then `springManager.setInitState()` + `reset()` → rest re-captured at the base pose.
   No-op for same-lineage items (`hair.glb`, garments).
2. **Exclude spring bones from the puppet** — the puppet pairs bones **by name**, and girl-base ships
   its own 12 `J_Sec_Hair*` chains, so a donor's same-named hair chains get snapped to girl-base's
   hair rest and the physics is overridden (side-locks stick out at head height). Skip any bone in
   `springManager.joints[].bone` when pairing; its parent (the head) is still puppeted.

Verified on donor 9128: rest error 0.0 m vs F-drop3 framing; joints swing ~51° (Looking Around) and
flare/lift under Cross Jumps; no explosion; `hair.glb` unregressed. See
`vroid-xwear-interop/docs/imported-hair-physics.md`.

## Traps (each cost a debugging session)

- **VRM 0.x detection**: `isAvatar` must check BOTH `VRMC_vrm` (1.0) AND `VRM` (0.x). Missing 0.x
  makes a whole avatar (face+body+hair) wear as one "standalone wearable" → a rotated head.
- **VRM 0.x orientation**: 0.x faces −Z vs +Z bases, different rest pose → items wear 180°
  backwards. The puppet can't fix it; the correct fix is an LBS rebake into the base bind pose
  (`axf/dressup_compose.py`, R=diag(−1,1,−1)). In-app 0.x is GATED to the Python lane.
- **Head axis permutation** (above) — girl-base's head rest is not identity.
- **Puppet name-collision on spring bones**: the puppet pairs item↔base bones by NAME, and girl-base
  carries its own `J_Sec_Hair*` chains — so an imported hair item's same-named chains get puppet-driven
  to girl-base's hair rest, overriding their physics (side-locks stick out at head height). Exclude the
  item's own `springManager.joints[].bone` from the pairing; spring joints belong to the spring manager,
  never the puppet. General rule for any item whose spring-bone names might exist on the base.
- **Spring rest captured before the puppet moves the head**: worn hair chains hang off the head; their
  spring rest is sampled at load in the donor frame, then the puppet drives the head to the base frame
  → first-frame jump/explosion. `springManager.setInitState()` after one puppet pass re-baselines the
  rest at the base pose. Harmless for same-lineage items.
- **Cross-avatar fit**: even a correctly-oriented VRM1 garment fits loosely if the donor body
  differs — inherent, hence the compatibility gate.
- **Outfit persistence**: `loadFile` resets the worn set on every model load; persist on
  wear/remove actions, NOT via a `wornItems` watcher (the watcher clobbers the save on reset).
