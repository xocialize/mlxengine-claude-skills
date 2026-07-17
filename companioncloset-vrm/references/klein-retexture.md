# Klein retexture (image-edit the garment's UV texture) — PROVEN

The committed CompanionCloset creation MVP: restyle a license-clean base garment's texture with **Klein**,
keeping its shape/UVs. Proven end-to-end with the real model — controllable by prompt, consistent by seed.

**Klein = FLUX.2-klein-4B** (Black Forest Labs, **Apache-2.0**, the shippable 4B tier; 9B variants are
Non-Commercial). At `~/Development/mlxengine-image/` (weights `weights/FLUX.2-klein-base-4B`, ~22 GB).
`diffusers.Flux2KleinPipeline` — its `__call__` takes an **`image` param = native Kontext-style edit**
(conditions on the input image, applies the prompt). Ref invocation: `/tmp/klein_edit.py`.

## The loop (each half proven separately, then together)
1. **Edit** the garment's real UV **atlas** (its `_MainTex`/diffuse) with Klein: `pipe(image=atlas,
   prompt=style, num_inference_steps=28, guidance_scale=3.5, generator=seed)`. Output = restyled atlas PNG.
2. **Re-embed** (trivial): drop the new PNG into `axf_springs ... texdir` — it maps straight onto the
   existing UVs. No geometry/rig change.

## Key findings
- **Base-model Klein edits the fragmented UV atlas WELL — a LoRA is NOT a prerequisite.** Feared the
  ~30-island atlas was out-of-distribution; instead Klein **respected island boundaries + layout** while
  doing real **style transfer** (gold filigree → *silver* filigree — a hue-shift can't do that). Maps back
  cleanly (no smear/misalign on the worn garment).
- **Controllable + consistent:** same seed + same input image → **structure held**; the **prompt drives
  the style** (sapphire/silver vs crimson/gold came out structurally identical, stylistically distinct).
  The guidance prompt is the primary art-direction lever. Guidance ~3.5, 28 steps.
- **Two-tier retexture strategy:**
  - **Simple attributes** (eye color, accent recolor) → cheap **`axf/retex_material.py`** hue-retint of
    the specific material's texture (set hue + boost sat, preserve value/detail/alpha). **<1 s**, no diffusion.
    `retex_material.py in.vrm out.vrm EyeIris 135 1.6` → green eyes.
  - **Complex restyle** (garment material/pattern/style transfer) → **Klein**.
- **Cost:** ~13–27 min per 1024² edit on an M-series Mac (image-conditioned float32/MPS ~33–57 s/step;
  thermal/memory variance). The unlock = the MLX/quantized Klein port (int4 DiT ~2.35 GB fits a 16 GB Mac).

## UV island mask (`axf/uv_mask.py`) — the refinement lever
Rasterizes the garment's UV triangles from the item JSON into a texture-space mask (filled + dilated +
overlay-on-atlas for alignment; FBX V-flip already applied). Feeds all three refinement paths if the base
model ever needs help: **masked inpaint** (edit only inside islands, no seam bleed), **layout-conditioned
LoRA** (island layout as a control channel → far less training data), **render-space bake** (edit a
coherent render, project back through the UVs). Currently optional — base Klein is good enough at MVP quality.

## Color-space contract (when generating/deriving MToon slots — see mtoon-texturing.md)
baseColor + shadeMultiply = **sRGB**; shadingShift / outlineWidth / uvAnimMask = **linear**;
shadeColorFactor defaults **black** → always supply a non-black shade (warm-darkened for skin, cool for cloth).
