---
name: companioncloset-vrm
description: >-
  Reference and how-to for working with VRM 1.0 avatars, outfits, and hair in the CompanionCloset
  system — the three.js / @pixiv/three-vrm runtime plus the VRoid XWear (.xwear) garment pipeline.
  Use this WHENEVER the task touches: VRM / .vrm / VRM 1.0 files and their glTF extensions
  (VRMC_vrm, VRMC_springBone, VRMC_node_constraint, VRMC_materials_mtoon); MToon / MToon10
  materials and texturing (base/shade/normal maps, _MainTex/_ShadeTex/_BumpMap, shade-color,
  color spaces); springbone / spring bone / cloth / hair physics (joint chains, colliders,
  Verlet simulation, auto-synthesizing bones for generated garments); VRChat PhysBone → VRM
  conversion; the XWear / .xwear / VRoid garment format; the TRELLIS→XWear write path or Companion
  Closet read path; skinning generated garments to a humanoid skeleton; or generating/baking
  textures (e.g. from Klein or an image-gen model) onto VRM meshes. ALSO covers the built pipeline:
  the AXF / .axf decomposable avatar format (components, variants, compose→VRM, producer contract);
  extracting a garment/hair/accessory from an FBX (FBXItemKit native reader, targeted multi-mesh
  extraction, UV/normal/material decode, Blender coordinate/unit traps); retargeting an outfit onto
  the canonical N00 base; spring-bone SYNTHESIS from dynamic (cloak/skirt/hair) bones + colliders +
  cloth/hair/stiff/rigid profiles; the N00 base spec + conformance checker + dev-vs-shippable
  licensing; and Klein image-EDIT retexturing of a garment's UV atlas. ALSO the RUNTIME wear side:
  our owned `.vrmxa` (avatar) / `.vrmxw` (wearable) formats (xwear demoted to an export tier); the
  canonical fixed girl-base composition; dressing items onto a base via the puppet-rig model;
  drag-drop import-to-closet + auto-wear; the proportion compatibility gate for donor VRMs; hair
  head-anchoring; and "why did the imported hair explode / the head rotate / the outfit go
  backwards". Trigger even when the user
  says only "the avatar's skirt clips through the legs", "why is my outfit black in shadow",
  "how do I retexture this VRM", "convert this PhysBone rig", "get this garment out of the FBX", "the
  wing droops", "why did the outfit land at the origin", or names Companion, Closet, XWear, AXF, N00,
  Klein garments, or TRELLIS garments without saying "VRM". Prefer this over generic 3D/glTF advice
  for anything anime-avatar / VRoid / VRM / AXF shaped.
---

# CompanionCloset VRM

Working knowledge for VRM 1.0 avatars — outfits, hair, textures, and physics — as used by the
CompanionCloset apps (**Companion** and **Closet**, both three.js / `@pixiv/three-vrm`) and the
**VRoid XWear** garment pipeline that feeds them.

This skill exists because VRM has three sharp edges that generic glTF/3D knowledge gets wrong,
and each one silently breaks generated content:

1. **MToon is multi-slot, and shade defaults to black.** A garment textured with only an albedo
   renders **fully black in shadow**. You must produce a shade map or set a shade color. → [references/mtoon-texturing.md](references/mtoon-texturing.md)
2. **Springbones are a data model you can author, not a black box.** Generated skirts/hair have no
   bones; you synthesize joint chains + colliders. PhysBone→VRM mapping has name traps. → [references/springbone-physics.md](references/springbone-physics.md)
3. **XWear is Unity-space; VRM/glTF is not.** Every read/write crosses a basis change (negate X,
   flip winding, tangent.w). → [references/xwear-pipeline.md](references/xwear-pipeline.md)

CompanionCloset **authors** VRM **1.0** (the split `VRMC_*` extensions). But most VRM content in the
wild is **0.x**, so anything that *ingests* existing avatars will meet 0.x constantly — where MToon
and springbone live in entirely different JSON. If you're loading/migrating third-party `.vrm`, read
[references/vrm-0x-ingest.md](references/vrm-0x-ingest.md). Author 1.0; be ready to read 0.x.

## How the pieces fit

**BUILT + PROVEN creation path (the MVP, use this for sourcing garments):**
```
license-clean garment in an FBX (a separable rigged object)
  → FBXItemKit native extract: targeted mesh + UV/normal split + material/texture + bone hierarchy
  → coordinate/unit normalize (Blender cm/m + Z-up→Y-up traps — the #1 bug source)
  → retarget onto canonical N00 base (≈identity) + spring-bone SYNTHESIS from dynamic bones
  → embed texture (MToon) → compose into the AXF avatar → VRM 1.0 (three-vrm)
  → Klein image-edit the garment's UV texture to restyle (prompt-driven, consistent)
```
See fbx-garment-pipeline.md, avatar-exchange-format.md, klein-retexture.md, n00-base-spec.md.

**Generation path — BUILT through the item stage (2026-07-16), rigging still gated:**
```
Klein text→image (item conventions) → TRELLIS.2 1024_cascade → decimate/Y-up/class-scale
  → pbr_to_mtoon (synth shade) → UNRIGGED .item.glb + provenance sidecar     [BUILT: generate_item.py]
  → rigging [GATED, T3.3]: proximity weight-transfer from the skinned base (fitted items)
    or SkinTokens auto-rig (loose) → springs → AXF component / .xwear
```
Also BUILT on the same stack: **re-texture an existing garment from one reference image with
bit-identical UVs** (~30 s; composes with Klein), whole-outfit multi-material restyle (~45 s),
mesh→SLat encode (bit-exact in Swift), and RePaint region-editing. Details, entry points, and
the hard-won gotchas: [references/trellis2-generation.md](references/trellis2-generation.md).
Ground-truth rigging corpus for the T3.3 gate: `vroid-xwear-interop/base/garments/`
(license-clean pixiv sample garments with per-vertex weights).

The two neutral representations: **AXF** (the decomposable master avatar) and the FBX-extract **item
JSON** (positions/uvs/normals/indices/joints/weights/boneNames/humanoidBones/inverseBinds/boneParents/
boneWorldBinds/materials). Legacy `Garment` (GLTFKit) still maps 1:1 to an AXF `outfit` component.

## Licensed dev fixtures (2026-07 — use these, not VRoid-corpus files)

`vroid-xwear-interop/base/` holds the license-clean canonical assets (each has a `.license.json`
sidecar that must travel with derivatives; provenance: pixiv sample-girl lineage, VRM Public
License 1.0 + VRoid Hub all-allow grant — modification/redistribution/commercial OK):

- **`girl-base.vrm`** — VRM 1.0 nude dev mannequin (complete body, 57 `Fcl_*` face morphs, all
  18 expression presets, 12 hair springbone chains). Built by `axf/graft_girl_base.py` — which
  also documents two cross-version rebind traps (VRM0↔VRM1 180° facing flip not encoded in bind
  data; non-normalized 1.0 bind rotations → anchor rebinds on bind translations only).
- **`garments/{tops,bottoms,shoes}.glb` + `.weights.json`** — ground-truth skinned garments on
  the full J_Bip rig (tops carries authored `VRMC_springBone` chains + thigh colliders). Score
  SkinTokens garment-rig output against the `.weights.json` influences; use as VRMDressKit /
  XWear round-trip fixtures.
- **`girl-base-bindpose.json`** — measured 54-bone bindpose; matches canonical n00 within
  ~15 mm (uniform height offset only), so VRoid-ecosystem garments fit natively.

## Routing — read the reference for your task

| Task | Read |
|---|---|
| VRM file structure, glTF extensions, node references, `VRMC_node_constraint` | [references/vrm-data-model.md](references/vrm-data-model.md) |
| Texturing outfits/hair, MToon slots, shade maps, color spaces, what Klein must output | [references/mtoon-texturing.md](references/mtoon-texturing.md) |
| Spring/cloth physics: data model, three.js sim, auto-synthesis, PhysBone conversion, tuning | [references/springbone-physics.md](references/springbone-physics.md) |
| The `.xwear` format, `Garment` intermediary, groups/segmentation, read/write paths | [references/xwear-pipeline.md](references/xwear-pipeline.md) |
| Loading/migrating existing (0.x) VRMs: where 0.x hides MToon, the 0.x↔1.0 slot map | [references/vrm-0x-ingest.md](references/vrm-0x-ingest.md) |
| VRoid texture resolution, atlas-per-category, layer naming, the `.vroid` sample | [references/vroid-texture-templates.md](references/vroid-texture-templates.md) |
| **AXF** — our decomposable whole-avatar format: container, components, compose, producer contract | [references/avatar-exchange-format.md](references/avatar-exchange-format.md) |
| **Getting a garment from an FBX** onto the base: native extract, coord/unit traps, retarget, spring **synthesis**, profiles | [references/fbx-garment-pipeline.md](references/fbx-garment-pipeline.md) |
| **N00 base spec + conformance:** material taxonomy, mesh split, 14 expressions, licensing, checker facets | [references/n00-base-spec.md](references/n00-base-spec.md) |
| **Klein retexture:** image-edit the garment's UV atlas, two-tier strategy, UV mask, consistency, cost | [references/klein-retexture.md](references/klein-retexture.md) |
| **TRELLIS.2 lane:** generate items (image→MToon GLB), re-texture w/ UV preservation, whole-outfit restyle, encode, RePaint region-edit, gotchas | [references/trellis2-generation.md](references/trellis2-generation.md) |
| **Wear runtime:** dressing items at runtime — the puppet-rig model, drop-to-closet import routing, compatibility gate, hair head-anchor, and the traps (VRM0 detect, head axis-permutation, exploding hair) | [references/wear-runtime.md](references/wear-runtime.md) |
| **Owned formats:** `.vrmxa` (avatar) / `.vrmxw` (wearable) specs, the canonical girl-base composition, xwear as an export tier | [references/owned-formats.md](references/owned-formats.md) |

Load only what the task needs — each reference is self-contained.

## The things people get wrong (quick answers)

**"The outfit goes black in shadow."** MToon's `shadeColorFactor` defaults to `[0,0,0]`. In
shadow the shader lerps toward that black. Fix: give the material a `_ShadeTex`/`shadeMultiplyTexture`
**or** set a non-black `_ShadeColor`/`shadeColorFactor` (convention: base albedo darkened ~20–40%
and hue-shifted toward blue/purple). Details + color-space table in the MToon reference.

**"The skirt clips through the legs."** Spring joints only collide with the collider groups
*assigned to their chain*. You need body colliders (spheres/capsules on hips/thighs) **and** the
skirt's chain must reference that collider group. Layout + collider recipes in the springbone reference.

**"Converted PhysBones behave wrong."** VRM `stiffness` ≈ PhysBone **`Pull`**, *not* PhysBone
`Stiffness` (VRChat's "Advanced Stiffness" is a different concept — never map by name). The two
systems use different integration math, so conversion is approximate and needs hand-tuning. The
proven default multipliers and collider decomposition are in the springbone reference.

**"Imported hair explodes / a rotated head appears / the outfit is backwards."** These are the
runtime WEAR traps, not authoring bugs: (a) VRM 0.x avatars fail a `VRMC_vrm`-only "is avatar"
check and wear whole (face+body+hair) → rotated head; detect `VRM` too. (b) 0.x faces −Z → garments
would wear **180° backwards** through the puppet. HANDLED (F-ingest, 2026-07-17): the drop front
door **auto-converts** the 0.x donor to a self-consistent VRM 1.0 (`vrm0to1Full.js`, incl. the 180°
geometry flip + garment springs migrated), then extraction runs the **generalized clean anchor**
(R=I) that also handles native 1.0: each humanoid bone gets `IBM'=baseBind⁻¹·T(Δ)` (translation-only
re-seat onto girl-base, orientation cancelled — head-weighted verts get the base-head anchor for
free, so accessory verts on the head stay upright; **same-lineage ⇒ Δ=0 ⇒ identity**), carried
garment/tail/ear spring chains keep their donor IBM with the chain root re-authored under its
humanoid ancestor (the `extractHairRigged` recipe, P generalized from head → hips/head), and
non-humanoid non-spring joints redirect to the nearest humanoid ancestor. So converted 0.x drops
land forward + fitted, ears upright, tail behind, AND garment springs sway. The earlier
metadata-only F-drop2 path (which dropped springs, R=diag) is superseded; its raw-0.x branch is
retained `@deprecated`. Remaining gap = cross-avatar **shell fit** (tight boots/stockings shaped for
the donor interpenetrate girl-base; fix body-side via pre-scale + skin-mask, not placement — F-fit).
See vrm0-inapp-conversion.md §F-ingest. (c) Hair's source spring bones don't exist on a bald base →
the puppet explodes it; head-anchor it instead, binding against the BASE head rest frame (girl-base's
head bind is an axis permutation — binding against the source head swirls the hair). Full model +
fixes in the wear-runtime reference.

**"Can I drag a random VRoid model in and dress my base in its clothes?"** Only if its build is
close to the base — the wear path doesn't scale, so a doll (0.96 m) or a tall model distorts. A
proportion **compatibility gate** (head height + torso ratio) marks incompatible donors rather
than force-fitting. VRMs come in all shapes; that's expected. See wear-runtime.md.

**"The FBX garment landed at the origin / exploded / is rotated."** Blender FBX export is
unit/axis-inconsistent: `Cluster.Transform` (meshBind) is in a different unit than `TransformLink`, or
identity, and geometry vs transforms can differ by 100× or a −90°X. Fix: derive meshBind from the **mesh
Model node's world transform** (not `Cluster.Transform`), re-export with `bake_space_transform=True,
axis_up='Y'`, and scale positions **and** IBM translation by the same cm→m factor. Full recipe in
fbx-garment-pipeline.md §2.

**"The wing/cape droops like a wet rag."** Spring classification keys on **bone names** (`Cloak.*` →
cloth), so a non-cloth mesh riding cloth-named bones gets cloth gravity and collapses. Fix: pass a spring
**profile** (`stiff` = holds shape but flutters, `rigid` = holds exact rest shape) to `axf_springs`.
Long-term, infer the profile from the mesh's role/material. See fbx-garment-pipeline.md §4.

## Runtime vs package-time physics

Spring physics can live in **either** place, and CompanionCloset may use both:
- **Package-time synthesis** — author `VRMC_springBone` chains + colliders when writing the
  `.xwear`, so outfits ship with physics baked in. Portable, works in any VRM runtime.
- **Runtime (three.js)** — `@pixiv/three-vrm`'s `VRMSpringBoneManager.update(dt)` drives the
  Verlet simulation each frame from whatever chains the loaded model carries.

They are not exclusive: synthesize at package time so the data exists, and the three.js runtime
simulates it at load. The springbone reference covers both the authored data model and the runtime.

## Source provenance

Content distilled from two adversarially-verified deep-research passes against primary sources
(`vrm-c/vrm-specification`, `@pixiv/three-vrm` API docs, vrm.dev, VRChat PhysBone docs) plus the
validated `vroid-xwear-interop` format spec, then **checked against real assets**: a 24-model
VRoid `.vrm` corpus (texture-slot usage, 0.x-vs-1.0 split, shade values) and a commercial VRoid
texture set (resolution, layer naming). Where a value is art convention rather than spec-sourced
(e.g. exact shade-darkening amounts, per-cloth-type tuning ranges), the reference says so — treat
those as starting points to tune, not fixed truth.

The **AXF / FBX-garment / N00 / Klein** references (avatar-exchange-format, fbx-garment-pipeline,
n00-base-spec, klein-retexture) are distilled from the **built-and-render-verified** pipeline: a real
magical-girl dress + hair + wings extracted from an FBX, retargeted onto a VRoid nude N00 base, spring-
synthesized, textured, and **Klein-restyled** — all rendered in three-vrm and passing the conformance
checker. Those are working-code facts (FBXItemKit, `axf/*.py`, `conformance_check.py`), not just research.
Repo source of truth: `~/Development/vroid-xwear-interop/` (+ `FBXItemKit/`, `AXFKit/`). Dev-vs-shippable
licensing (VRoid/marketplace = dev-only; CC0 = shippable) is a hard constraint — see n00-base-spec.md.

**Known open gap (narrowed):** VRoid texture **resolution (2048²)**, **PNG format**, **atlas-per-
category** layout, and **layer naming** are now verified (see
[references/vroid-texture-templates.md](references/vroid-texture-templates.md)). What remains open is
the precise **0–1 UV island positions** per garment element — read those from the atlas PNGs or the
`.vroid` mesh; the vendor guide is an application workflow, not a UV map. Don't assume a fixed template.
