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
| **Hair** | rigid 1.0 bind to the HEAD bone | `hairAnchor.js` | hair spring bones don't exist on a bald base → puppet explodes it |
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
orientation at the base head; during animation it follows the head. Rigid (no strand physics
yet — head springs are a later nicety). Head-anchoring is source-rig-independent (only needs
the head bone) so it also absorbs cross-avatar head position automatically.

## Traps (each cost a debugging session)

- **VRM 0.x detection**: `isAvatar` must check BOTH `VRMC_vrm` (1.0) AND `VRM` (0.x). Missing 0.x
  makes a whole avatar (face+body+hair) wear as one "standalone wearable" → a rotated head.
- **VRM 0.x orientation**: 0.x faces −Z vs +Z bases, different rest pose → items wear 180°
  backwards. The puppet can't fix it; the correct fix is an LBS rebake into the base bind pose
  (`axf/dressup_compose.py`, R=diag(−1,1,−1)). In-app 0.x is GATED to the Python lane.
- **Head axis permutation** (above) — girl-base's head rest is not identity.
- **Cross-avatar fit**: even a correctly-oriented VRM1 garment fits loosely if the donor body
  differs — inherent, hence the compatibility gate.
- **Outfit persistence**: `loadFile` resets the worn set on every model load; persist on
  wear/remove actions, NOT via a `wornItems` watcher (the watcher clobbers the save on reset).
