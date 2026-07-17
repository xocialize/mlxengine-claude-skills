# Ingesting VRM 0.x (the read path reality)

CompanionCloset **authors** VRM 1.0, but most VRM content in the wild is **0.x** — so if you
*ingest* existing avatars (loading third-party `.vrm`, studying a corpus, migrating assets), you hit
0.x constantly and it stores MToon in a completely different place. This reference is the 0.x→1.0
material bridge.

## Table of contents
- [Why this matters (corpus evidence)](#why-this-matters)
- [Where 0.x puts MToon: VRM.materialProperties](#where-0x-puts-mtoon)
- [The 0.x ↔ 1.0 slot map](#the-0x--10-slot-map)
- [Reading a 0.x material](#reading-a-0x-material)
- [Springbone location differs too](#springbone-location-differs-too)

## Why this matters

In a sampled corpus of 24 VRoid-exported `.vrm` files, **20 were VRM 0.x and only 4 were 1.0**. So
"we're VRM 1.0 only" is true for what you *generate*, and false for what you *load*. A reader that
only understands `VRMC_materials_mtoon` will see a 0.x model as a pile of `KHR_materials_unlit`
materials with no shade, no matcap, no toon parameters — because in 0.x none of that lives on the
glTF material.

## Where 0.x puts MToon

- **VRM 1.0:** each glTF material carries `extensions.VRMC_materials_mtoon` with its own texture
  slots and factors. Self-contained per material.
- **VRM 0.x:** the glTF material is a minimal (often `KHR_materials_unlit`) fallback. **The real
  MToon definition lives in a top-level array `extensions.VRM.materialProperties[]`**, parallel to
  `glTF.materials[]` (same order/index), using **Unity shader property names** (`_`-prefixed).

Each `materialProperties` entry has:
- `shader` — `"VRM/MToon"` (the toon material) or `"VRM_USE_GLTFSHADER"` (plain/unlit passthrough).
- `textureProperties` — `{ "_MainTex": <texIndex>, ... }` mapping Unity slot → glTF texture index.
- `floatProperties` — scalar params (`_Cutoff`, `_ShadeShift`, `_ShadeToony`, `_BumpScale`,
  `_OutlineWidth`, `_UvAnimScrollX/Y`, `_ZWrite`, …).
- `vectorProperties` — colors and per-texture tiling/offset (`_Color`, `_ShadeColor`,
  `_EmissionColor`, `_RimColor`, `_OutlineColor`, …).

## The 0.x ↔ 1.0 slot map

Verified against real files (usage counts from the 24-model corpus in parentheses):

| Purpose | VRM 0.x (`VRM.materialProperties`) | VRM 1.0 (`VRMC_materials_mtoon` / glTF core) |
|---|---|---|
| base / lit color | `_MainTex` + `_Color` (532×) | `pbrMetallicRoughness.baseColorTexture` + `baseColorFactor` |
| shade color | `_ShadeTexture` + `_ShadeColor` (529×) | `shadeMultiplyTexture` + `shadeColorFactor` |
| normal | `_BumpMap` + `_BumpScale` (474×) | glTF `normalTexture` (+ `scale`) |
| emissive | `_EmissionMap` + `_EmissionColor` (461×) | glTF `emissiveTexture` + `emissiveFactor` |
| matcap / sphere-add | `_SphereAdd` (453×) | `matcapTexture` + `matcapFactor` |
| outline width | `_OutlineWidthTexture` (16×) | `outlineWidthMultiplyTexture` |
| rim | `_RimColor` / `_RimLift` / `_RimFresnelPower` (floats/vecs) | `parametricRim*Factor` / `rimMultiplyTexture` |
| cutoff | `_Cutoff` (float) | `alphaCutoff` (glTF) + `alphaMode: "MASK"` |
| toon shading shape | `_ShadeShift`, `_ShadeToony`, `_ShadingGradeRate` (floats) | `shadingShiftFactor`, `shadingToonyFactor` (+ `shadingShiftTexture`) |
| UV scroll animation | `_UvAnimScrollX/Y`, `_UvAnimRotation` | `uvAnimationScrollXSpeedFactor` / `…YSpeedFactor` / `…RotationSpeedFactor` |

Two structural notes:
- **Naming is `_ShadeTexture` in 0.x but `shadeMultiplyTexture` in 1.0** (and the XWear/MToon10
  Unity slot is `_ShadeTex`) — three names, one concept. See the slot map in
  [mtoon-texturing.md](mtoon-texturing.md#slot-map).
- 0.x `_ShadeShift`/`_ShadeToony` roughly correspond to 1.0 `shadingShiftFactor`/`shadingToonyFactor`
  but are not a 1:1 numeric transfer — the shading model was reparameterized between versions.

## Reading a 0.x material

To resolve material *i*'s real textures/colors in a 0.x file:
1. Read `glTF.materials[i]` for the fallback (base color factor/texture, alpha, unlit flag).
2. Read `extensions.VRM.materialProperties[i]` for the MToon truth: `shader`, `textureProperties`
   (→ glTF texture indices → images), `floatProperties`, `vectorProperties`.
3. Map Unity slots to your internal MToon via the table above.
4. If `shader == "VRM_USE_GLTFSHADER"`, treat it as a plain glTF material (no toon shade/matcap) —
   these are the simple flat-color parts.

Textures still reference the mesh's `TEXCOORD_0` (single UV set is the norm — only 2 of 24 corpus
models used a second UV). So the bake-onto-existing-UV rule from
[mtoon-texturing.md](mtoon-texturing.md#baking-onto-existing-uvs) holds for 0.x too.

## Springbone location differs too

Same split as materials:
- **1.0:** `extensions.VRMC_springBone` (chains/joints/colliders — see
  [springbone-physics.md](springbone-physics.md)).
- **0.x:** `extensions.VRM.secondaryAnimation` (`boneGroups` + `colliderGroups`), with different
  field names (`stiffiness` [sic], `dragForce`, `gravityPower`, `hitRadius`, `bones`, `colliders`).
  Same physical model, older schema. Every corpus model had springbone in one form or the other.

If a migration/ingest tool is on the roadmap, budget for translating both materials **and**
`secondaryAnimation`→`VRMC_springBone`, not just geometry.
