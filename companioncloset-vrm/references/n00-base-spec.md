# N00 canonical base spec + conformance (what "compatible" means)

The **N00** humanoid is the canonical base body/skeleton CompanionCloset targets — a shared skeleton +
bind-pose that makes cross-source outfit fit ≈identity and defines the "Compatible Item" output contract.
Ground truth = a real **VRoid Studio N00 nude export** (`Spec.vrm`); shippable version = a CC0 MakeHuman
build. Full contract: `~/Development/vroid-xwear-interop/COMPATIBLE-ITEM-SPEC.md`. Checker:
`conformance/conformance_check.py` (dual-aware 0.x/1.0; per-facet PASS/WARN/FAIL + MVP/Full level).

## Canonical layout (from the real VRoid N00 nude export)
- **Two-mesh split:** `Face (merged)` (7 primitives) + `Body (merged)` (1 primitive = nude `Body_SKIN`,
  no fused outfit). Mirror this split when authoring.
- **Material taxonomy — pattern `N00_000_00_<Part>_00_<TYPE>`, 8 MToon:** Body→**SKIN**, Face→**SKIN**,
  FaceMouth→FACE, FaceBrow→FACE, FaceEyeline→FACE, EyeIris→**EYE**, EyeWhite→EYE, EyeHighlight→EYE. The
  `SKIN/FACE/EYE` suffix drives segmentation + MToon-slot assignment.
- **Rig:** 54-bone VRM humanoid (`J_Bip_*` naming in VRoid). ~1.58 m for the VRoid N00 (anime, ~6.5-head);
  CC0 MakeHuman build is realistic-slim ~1.19 m (proportion gap = art, not conformance).
- **Expressions:** full **14 presets** (emotions + 5 visemes + 3 blinks + neutral). A base with 0 morphs
  can't blink/lip-sync — the biggest functional gap to close on any base that lacks them.
- **Physics:** `VRMC_springBone` (nude base has minimal chains). **No** `VRMC_node_constraint` on the nude base.

## Conformance facets (conformance_check.py)
`2.0` integrity (dangling-ref + POSITION min/max — catches morph-target dropped `targets`), `2.1`
container (VRM 1.0 `VRMC_vrm`), `2.2` humanoid (≥ required bones mapped), `2.3` materials (MToon),
`2.4` segmentation (distinct meshes/materials + firstPerson flags), `2.5` scale/units (~human meters),
`2.6` skinning, `2.8` provenance (source+license+permitted-uses — MUST for shipped items), `2.9`
expressions (preset→morph binds valid; catches stale node indices after a merge). MVP = core facets pass;
Full adds provenance + firstPerson + 1.0 container.

## Licensing discipline (persistent constraint)
- **VRoid / VRoid-Hub output** (incl. `Spec.vrm`, the 8298 corpus) = **NOT** modifiable/redistributable →
  dev/reference ONLY (gather facts, don't ship derived geometry).
- **Commercial marketplace FBX** (e.g. TurboSquid Standard) = incorporation into interactive products,
  NOT redistribution-as-extractable-model → shipping avatars as `.vrm`/`.axf` files is the extractable
  zone → likely prohibited. Marketplace/VRoid assets are DEV/TEST only (with a valid purchase).
- **Shippable** = CC0 / explicitly redistribution-permissive sources only (MakeHuman = CC0). Shipped item
  = derivative of (license-clean base) + (own-generated textures). Stamp source+license on every component.

## The recurring wall
Many "base" sources have **no separable nude body** — the outfit is fused into the body mesh (VRoid 8298's
kimono, stock Mixamo characters). That's why we author our own N00 base. When a source's body mesh is
fused, extraction yields a clothed body, not a nude one; look for a `.blend`/DCC source with separate objects.
