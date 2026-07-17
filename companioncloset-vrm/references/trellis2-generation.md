# TRELLIS.2 generation & re-texturing lane (BUILT, 2026-07-16)

The image→3D / re-texture / region-edit capabilities that feed the closet. Everything here
is **built and verified locally** (roadmap of record:
`~/Development/vroid-xwear-interop/TRELLIS2-CLOSET-ROADMAP.md`, Phases 0–2 + T3.1/T3.2 done).
Two stacks, same model (microsoft/TRELLIS.2-4B, MIT; weights mirrored gated at
`xocialize/trellis2-mlx` incl. encoders + 1024 flows):

- **Python oracle** — `~/Development/mlxengine-3d/DEV/TrellisDev/trellis2-apple` on the venv
  `~/Development/mlxengine-3d/.venv` (Metal backends: flex_gemm/mtlgemm sparse conv+attn,
  cumesh/mtlmesh, mtldiffrast, mtlbvh). Scriptable; the lane reference. Slow at 1024_cascade
  (~25 min/item).
- **Swift engine** — `~/Development/mlxengine-3d/WIP/mlx-trellis2-swift` (module TRELLIS2 /
  Trellis2Kit). Every stage parity-gated vs the oracle (SW1–SW8); res512/1024/1536 wired;
  the production-speed path (~2 min res512 e2e).

## Capabilities → entry points

| Capability | Entry point | Notes |
|---|---|---|
| **Generate item** (image → closet-ready MToon GLB) | `vroid-xwear-interop/axf/generate_item.py <img> <class> <outdir>` | 1024_cascade default; sentinel-face filter; Y-up; class-height scale (dress 1.15 m, top 0.62, skirt 0.52, shoes 0.14, hair 0.38); emits `.item.glb` + provenance sidecar. Items are UNRIGGED (rigging = T3.3) |
| Conditioning images | `axf/item_image_prompts.py` + `TRELLIS2-ITEM-IMAGE-CONVENTIONS.md` | ONE item/image (multi-item fuses; no decomposition), ghost-mannequin worn shape, white bg, front ¾, no cel-shading prompts. Klein test corpus: `/Volumes/Satechi/TrellisRedux/Corpus/t31/` |
| **Re-texture existing garment** (keeps UVs!) | oracle `Trellis2TexturingPipeline.run(mesh, image)`; Swift `TexturingPipeline` | ~30 s warm at res512/tex1024 (**standardize on that** — res1024/tex2048 measured WORSE + 18× slower). UVs bit-identical → output stays Klein-editable. Requires the mesh to have ONE atlas |
| Whole-outfit restyle (multi-material) | `mlxengine-3d/DEV/TrellisDev/texturing/retex_outfit.py` | **Generate once, bake per material**: one tex-SLat generation over the concatenated outfit (style-consistent), then ~0.5 s original-UV re-projection per material; composite ORIGINAL albedo alpha (cutout frills keep shape). 10-material outfit ≈ 45 s |
| PBR→MToon | `axf/pbr_to_mtoon.py in.glb out.glb [--role cloth|skin|hair]` | Synthesized shade TEXTURE (cloth: value×0.7 + cool hue lerp; skin: warm ×[0.930,0.624,0.708], never cooled); alpha OPAQUE-or-MASK@0.5, never auto-BLEND (layered garments alpha-sort badly); metal/rough reported then dropped |
| Encode (mesh→SLat) | oracle `encode_shape_slat`; Swift `DualGridConvert` + `ShapeSlatEncoder` | CPU C++ converter (no CUDA anywhere); round-trip fidelity ~1.7 voxels @1024³ on a layered outfit; Swift bit-exact (SW7) |
| **Region edit** (RePaint) | `mlxengine-3d/DEV/TrellisDev/texturing/edit/repaint_probe.py` | Preserve-outside (½-voxel chamfer) / regenerate-inside an allowance box; clean seams single-pass. Growth is allowance-SHAPED, not semantic (global DINOv3 cond) — bulk adds & re-rolls yes, "this buckle here" no (compositional path for that). Latent cell ≈ 5.3 cm @512 / 2.7 cm @1024 on a body — so NO direct eyes/nose/mouth editing: those live in TEXTURE (Klein) + parametric morphs; geometry re-dream also destroys Fcl_ blendshapes (topology). EXCEPTION — soft skull-scale head shaping (jaw/cheeks/cranium) is legitimate **on a T4.3 reference head before the N00 shrinkwrap** (roadmap T2.2 strength dial): the wrap output inherits N00 topology + morphs, so blendshape safety comes from the wrap, not the edit |

## Gotchas (each cost real debugging — check here first)

- **Sentinel faces:** raw SLat decode emits ~0.1% `-1` face rows — filter
  `(faces >= 0).all(axis=1)` before ANY postprocess (hardened cumesh raises on them; older
  builds segfaulted; numpy silently mis-indexes via negative indexing).
- **`ATTN_BACKEND=sdpa` env** required for the image→3D pipeline on Mac (dense SS stage
  defaults to CUDA flash_attn). Sparse backends auto-probe (`CONV=flex_gemm`).
- **Pipeline registry bug:** `trellis2.pipelines.from_pretrained` does a `globals()` lookup
  that bypasses its own lazy `__getattr__` → KeyError. Import pipeline classes directly.
- **Tex decoder needs the ENCODER's subdivision record** (`pred_subdiv=false`): upstream
  carries it invisibly on the SparseTensor spatial cache; any port must capture per-level
  child-occupancy masks at encode (Swift: `encodeCapturingMasks`) and re-align rows to the
  decoder's C2S coord order per level.
- **`concat_cond` is the NORMALIZED shape SLat** ((x−mean)/std); decoder input is the
  DE-normalized latent. Normalization constants live in `texturing_pipeline.json`.
- **Swift `sampleSLat` coords are batch-prefixed [N,4]** — passing [N,3] traps inside RoPE
  with a confusing reshape error. `GridSample3d.sample` also takes [N,4].
- **Swift print is block-buffered under pipes** — `setvbuf(stdout, nil, _IONBF, 0)` or
  crashes eat your gate output.
- **Torch upgrade ⇒ rebuild all four pedronaugusto Metal packages**
  (flex_gemm/cumesh/mtlbvh/mtldiffrast) with `--no-build-isolation` (mtlgemm additionally
  has a PEP 517 silent-wrong-ABI trap). PR#2 correctness patch archived at
  `mlxengine-3d/vendor/mtlmesh-pr2-metal-correctness.diff` — reapply on rebuild.
- **TRELLIS output is unit-cube normalized + model-space Z-up.** Always restore real-world
  scale (class table above) and export Y-up (`yUp: true` in Swift generate; `to_glb` handles
  the swap in the oracle). Record the transform (provenance sidecar).
- More field hazards (GPU watchdog on Ultra Macs, metallib min-OS, float-atomics M3+…):
  `mlxengine-3d/DEV/TrellisDev/METAL-STACK-FIELD-NOTES.md`.

## What TRELLIS.2 does NOT do (verified — don't re-research)

No text conditioning (front with Klein text→image). No official editing/variation ("round-trip
editing" on the web is a third-party myth — but OUR RePaint probe demonstrated the real thing).
No part decomposition (multi-item images fuse; P3-SAM designated for the T4.4 probe, dev-tool
only — Tencent license excludes EU/UK/SK). No rigging (T3.3: proximity weight transfer from the
skinned base for fitted items, SkinTokens for loose; ground-truth corpus at
`vroid-xwear-interop/base/garments/`). Hunyuan3D is not a replacement (license + watertight
bias + Paint re-unwraps, destroying UV composition).
