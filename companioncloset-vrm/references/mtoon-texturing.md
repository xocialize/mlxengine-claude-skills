# MToon texturing (MToon / MToon10)

How VRM's toon material works, what a texture-generation model (Klein) must output per garment,
and how to bake externally-generated maps onto VRM meshes. This is where generated content most
often goes visibly wrong.

## Table of contents
- [The core model: lit ↔ shade interpolation](#the-core-model-lit--shade-interpolation)
- [The black-shade trap](#the-black-shade-trap)
- [Slot map: spec ↔ three-vrm ↔ MToon10 (XWear)](#slot-map)
- [Color spaces — get this wrong and colors shift](#color-spaces)
- [What Klein must output per garment](#what-klein-must-output-per-garment)
- [Deriving a shade map from an albedo](#deriving-a-shade-map-from-an-albedo)
- [Alpha, cutoff, outline](#alpha-cutoff-outline)
- [Baking onto existing UVs](#baking-onto-existing-uvs)

## The core model: lit ↔ shade interpolation

MToon is the `VRMC_materials_mtoon` glTF extension. Unlike a PBR material (one albedo lit
continuously), MToon does **cel/toon shading**: it computes how lit each pixel is, then
**interpolates between two colors** by that lit-ness:

- **Lit (base) color** = `baseColorFactor` × `baseColorTexture` — the color under direct light
- **Shade color** = `shadeColorFactor` × `shadeMultiplyTexture` — the color where light doesn't reach

Within each slot, **texture × color is a multiply**. The shading term (a smoothed `dot(N,L)`,
optionally offset per-texel by `shadingShiftTexture`) drives the lerp, producing the characteristic
hard-ish anime light/shadow boundary rather than a smooth gradient.

So a garment's appearance is governed by **two** color inputs, not one. This is the single most
important thing to internalize about VRM texturing.

## The black-shade trap

`shadeColorFactor` **defaults to `[0,0,0]` (black)**. If a material supplies only a base
color/albedo and nothing for the shade, then everywhere the surface is in shadow the shader lerps
toward **black** — the outfit looks like it has huge black patches or goes fully black when lit
from the side.

This is the #1 failure mode for generated garments, because image-gen and 3D-gen pipelines
naturally produce an albedo and stop. **Two valid fixes:**
1. Emit a `shadeMultiplyTexture` / `_ShadeTex` (a shade map — see [deriving one](#deriving-a-shade-map-from-an-albedo)), or
2. Set a non-black `shadeColorFactor` / `_ShadeColor` (a flat shade tint).

For quality, (1) beats (2); for a fast default, (2) is enough to stop the black-out.

## Slot map

The same concept has three names depending on layer. Keep this table handy — the XWear pipeline
uses the `MToon10` (Unity property) names, three.js uses its own, the spec uses glTF terms.

| Purpose | Spec (`VRMC_materials_mtoon`) | `@pixiv/three-vrm` | MToon10 (XWear `_`) |
|---|---|---|---|
| Base / lit color | `pbrMetallicRoughness.baseColorFactor` + `baseColorTexture` | `color` + `map` | `_Color` + `_MainTex` |
| Shade color | `shadeColorFactor` + `shadeMultiplyTexture` | `shadeColorFactor` + `shadeMultiplyTexture` | `_ShadeColor` + `_ShadeTex` |
| Normal | glTF `normalTexture` (+ scale) | `normalMap` + `normalScale` | `_BumpMap` |
| Emissive | `emissiveFactor` + `emissiveTexture` | `emissive` + `emissiveMap` | (emissive props) |
| Shading boundary offset | `shadingShiftTexture` (R channel) | `shadingShiftTexture` | — |
| Rim / matcap / outline / uv-anim | rim/matcap/outlineWidth/uvAnimationMask textures | same | — |

Note MToon **reuses glTF core slots** for base color, normal, and emissive — it does *not* define
its own normal or emissive map. Only the toon-specific slots (shade, shadingShift, rim, matcap,
outline, uvAnimationMask) live under `extensions.VRMC_materials_mtoon`.

## Color spaces

Sampling a texture in the wrong color space shifts every color subtly (or badly). Confirmed
per-slot behavior:

| Texture | Color space |
|---|---|
| `baseColorTexture` (base/albedo) | **sRGB** (converted to linear on sample) |
| `shadeMultiplyTexture` (shade) | **sRGB** |
| `normalTexture` (`_BumpMap`) | linear (normal data) |
| `shadingShiftTexture` (uses R channel) | **linear** |
| `outlineWidthMultiplyTexture` (uses G channel) | **linear** |
| `uvAnimationMaskTexture` | **linear** |
| `baseColorFactor` (the factor, not a texture) | linear |

Rule of thumb: **color you see (base, shade) = sRGB; data masks (normal, shading-shift, outline,
uv-mask) = linear.** When exporting PNGs from a gen pipeline, tag/encode the color maps as sRGB and
the data maps as linear, or the runtime will mis-interpret them.

## What Klein must output per garment

To render correctly under MToon10, the minimum viable set is:

1. **`_MainTex`** — base albedo, **sRGB**. (Required.)
2. **`_ShadeTex`** — shade map, **sRGB**. Without this (or a non-black `_ShadeColor`), the garment
   blacks out in shadow. (Effectively required — see the trap above.)
3. **`_BumpMap`** — tangent-space normal, **linear**. (Optional but recommended for cloth detail;
   omit → flat shading, still valid.)
4. **`matcapTexture` / `_SphereAdd`** — a sphere-mapped additive highlight. **Not niche:** in a
   24-model VRoid corpus this was the *5th-most-bound* texture slot (used on skin, eyes, metal/leather
   accents for anime sheen). Additive, sampled by view-space normal (not UV). Omit for matte cloth;
   include for anything that should catch a highlight (leather, metal, silk, skin). Historically the
   `_SphereAdd` slot in 0.x.

Plus non-texture params the writer sets: `_Color` (usually white `[1,1,1,1]` so the albedo shows
unmodified), `_ShadeColor` (white if a shade *texture* carries the tone, or a tint if not),
`_AlphaMode` + `_Cutoff` (see below).

If Klein can only produce one map today, produce `_MainTex` **and** have the XWear writer set a
sensible non-black `_ShadeColor` as a stopgap — never ship albedo-only with default shade.

> **Verified against real models.** Across 24 VRoid `.vrm` files, every MToon material binds
> multiple slots together (`_MainTex` + `_ShadeTexture` + `_BumpMap` + `_EmissionMap` + `_SphereAdd`
> is the common set), every material carries a **non-black** shade (confirming the trap is real and
> universally avoided), all shade textures sample `texCoord: 0`, and `MASK` at `alphaCutoff 0.5` is
> the usual cutout setting. The multi-slot model isn't theoretical — it's what shipped content does.

## Deriving a shade map from an albedo

The spec confirms the multiply relationship and the black default; it does **not** define numeric
recipes. The following is **art convention** — a starting point to tune, not sourced truth — and it
**depends on the material**: shadow color is not universally "darker and bluer."

**Cloth / fabric (cool shadow):**
- Start from the base albedo.
- **Darken** to roughly 60–80% value (multiply luminance by ~0.6–0.8).
- **Hue-shift toward blue/purple** slightly (cool the shadow), often nudging saturation up a touch —
  fabric shadows read cool and slightly more saturated, not just darker.

**Skin / face (warm shadow) — this is the correction real data forced:**
- Skin shade is **warm and only lightly darkened**, *not* cooled. Real VRoid values sampled from
  shipped models: a 1.0 face `shadeColorFactor ≈ [0.93, 0.62, 0.71]` (warm pink, lightly down),
  a 0.x face `_ShadeColor ≈ [0.97, 0.81, 0.86]` (barely darkened, warm). Hue-shifting skin toward
  blue makes it look sickly/dead. Keep skin shade in the warm family, subtle.

**Either case**, a flat-tint equivalent (set `_ShadeColor`/`shadeColorFactor` to that color, skip the
texture) is the cheaper, lower-quality fallback.

So a texture-gen model should **branch on material role**: cool-shift garments, warm-shift skin.
Tune per art direction; treat the numbers as defaults to iterate on with real renders.

## Alpha, cutoff, outline

MToon supports the three glTF alpha modes:
- **OPAQUE** — no transparency (`_AlphaMode` 0)
- **MASK / cutout** — hard alpha test at `alphaCutoff` / `_Cutoff` (`_AlphaMode` 1). Use for hair
  cards, lace, cutouts — cheap, sorts correctly.
- **BLEND / transparent** — true alpha blending (`_AlphaMode` 2), plus a `transparentWithZWrite`
  boolean (default `false`) controlling whether it writes depth. Use sparingly (sorting cost).

Outline is an MToon feature (`outlineWidthMode`, `outlineWidthMultiplyTexture` in the G channel,
outline color). Relevant if generated garments should carry the anime ink outline; the outline
width map is **linear**.

## Baking onto existing UVs

MToon textures are standard glTF index-based references with `texCoord` (default `0`) selecting the
mesh's `TEXCOORD_0`. So the intended workflow is: **assign your generated map to the correct MToon
slot; it samples the mesh's existing UV set.** No re-UV needed — the map bakes onto the atlas the
mesh already carries.

⚠️ **Open gap:** VRoid's default UV template regions (where "tops" vs "sleeves" vs "collar" land in
0–1 UV space), standard resolutions, and layer separation were **not** established by research.
Before targeting specific UV regions with a generated map, determine the layout empirically from a
real export — inspect `uv1` on the XWear sample garment meshes, or the `.vroid` separated texture
layers. Don't assume a fixed template.
