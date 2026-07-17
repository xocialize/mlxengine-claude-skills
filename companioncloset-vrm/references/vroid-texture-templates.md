# VRoid texture templates & layer conventions

What VRoid's clothing/skin textures look like on disk — resolution, the atlas-per-category layout,
and the layer-naming convention — so a generated texture can target the right regions. Grounded in a
real commercial VRoid texture set (WonderfulP "Fantasy RPG Equipment", stable-ver, 2021) plus the
24-model `.vrm` corpus.

## What's verified vs still open

**Verified from real assets:**
- **Resolution: 2048×2048.** The composited atlases (clothes, body, cloak) are all 2048². Textures
  are **PNG** (1,142/1,142 images across the corpus were PNG — assume PNG, not JPEG).
- **One atlas per VRoid material category**, plus separable per-part layers.
- **Layer-naming convention** (below).
- **Explicit shade layers** ship separately — VRoid separates lit and shade, and skin-base shade
  comes in tone variants.

**Still open (narrowed, not closed):** the precise **0–1 UV island positions** — where "sleeve" vs
"collar" vs "hem" land in the atlas. The vendor "HowToUseVRoidTexture" guide is a **VRoid Studio
application workflow** (import each PNG into the Face/Clothing tabs, save custom items), **not** a UV
map. Read the actual UV islands from the atlas PNGs themselves or the `.vroid` mesh — don't assume a
fixed template. This is the residual of the original UV-template research gap.

## The atlas-per-category layout

A VRoid outfit's textures split by the editor's clothing categories (VRoid tabs: Face 顔 / Hair 髪型 /
Body 体型 / Clothing 衣装 / Accessory アクセサリー / Look ルック). On disk, a color variant folder holds one
composited atlas per category:

```
<variant>/
  clothes_00all.png        # 2048² — the Clothing-layer atlas (jacket, scarf, belt, gloves…)
  body_00inner_all.png     # 2048² — the Body/skin-layer atlas (on-skin: tops, leggings, gloves, base)
  pants.png                # bottoms
  shoes.png                # footwear
  parts/                   # the separable source layers that composite into the atlases
```

The `_00all` files are the **flattened result**; `parts/` holds the **pre-composite layers**. To
retexture one element without disturbing others, edit the relevant `parts/` layer and recomposite —
this is the file-level analog of the "retexture the tops group without touching the body" workflow.

## Layer-naming convention

From `parts/`, VRoid's layer names are `<category>_<NN><name>` (the `NN` orders compositing):

- **body_**: `body_10gloves`, `body_40tops`, `body_60shortpants`, `body_70leggings`,
  `body_80bodybaseshadow-{light,medium,olive}`
- **clothes_**: `clothes_10gloves`, `clothes_20scarf`, `clothes_30waistbelt`, `clothes_50jacket`,
  `clothes_40shadow`
- **pants_**: `pants_harness`, `pants_shoes_parts`
- **cloak (per element × color)**: `cloak_upper_<color>`, `cloak_under_<color>`, `hood_<color>`,
  `shoulder_<color>` — a multi-part garment split into named elements, each provided in multiple
  colorways (e.g. black, khaki).

Two conventions worth reusing:
1. **Shade layers are explicit and separate** (`clothes_40shadow`, `body_20/30/50shadow`,
   `body_80bodybaseshadow-*`). This is VRoid confirming the MToon shade slot at the asset level — the
   generated-texture pipeline should likewise produce a shade layer, not just albedo. See
   [mtoon-texturing.md](mtoon-texturing.md#deriving-a-shade-map-from-an-albedo).
2. **Skin base ships in tone variants** (`-light`/`-medium`/`-olive`). This is the asset-level form of
   the `_SKIN` passthrough (see [xwear-pipeline.md](xwear-pipeline.md#segmentation)) — the on-skin
   base adapts to the wearer's skin tone rather than baking one tone in.

**Internal (`.vroid` `data.bin`) naming — confirmed by extraction.** Pulling the 45 embedded PNG
layers out of a real `.vroid` (signature scan, validated) surfaced VRoid's internal asset names:
- Part/map naming: `N00_000_00_<Part>_00_<map>.texture` where `<map>` suffix = `spe` (sphere/matcap),
  `nml` (normal), and the composited atlases are `<cat>_00all` (e.g. `body_00all-<color>`,
  `all_in_clothes`). Confirms the `_SphereAdd`/matcap and normal slots from
  [mtoon-texturing.md](mtoon-texturing.md#slot-map).
- Merge/part keys: `MergeKey.PartGroup.N00.<Part>` (e.g. `.Face`, `.BaseHair`, `.N00_007_01_Tops`) —
  the part-group identity used to compose layers.
- 43 of 45 layers are transparent overlays (the separated part layers); the `_00all` atlases are the
  flattened composite. The composited garment atlas directly shows the 0–1 UV island layout — the
  most reliable source for the still-open precise-UV-region question is inspecting these atlases.

## The `.vroid` as parameter ground-truth

The set ships a `.vroid` project (ZIP → `data.bin` protobuf + `meta.json`). Per the vendor readme,
"you can check the parameters by opening the `.vroid` file" — so it holds the **real MToon material
values and spring-bone parameters** for this outfit (including the cloak's cloth physics).

**Schema-recovery status (Phase 0 de-risk, GO):** `data.bin` is Google.Protobuf, package `vroid_core`
(schema recoverable from VRoid Studio's IL2CPP metadata — descriptors ship as base64 `FileDescriptorProto`s).
Enough is known to plan the importer: geometry is stored as **`RawVertexAttributes`** (packed/raw form,
`num_vertices` uint32 + packed data) — **not** the structured `repeated Vector3` `VertexAttributes`
form (that's the editor representation). A naive "decode as Vector3/packed-float" fails; target the
`Raw*` messages. Full decode needs the complete descriptor via **Il2CppDumper** on `GameAssembly.dylib`
(the descriptor chunks are lexicographically sorted in metadata, so the cctor concat-order is required)
→ then SwiftProtobuf. This is the storefront-integration path; details in the interop repo's
`VROID-FORMAT-PLAN.md`. Until then, use the separated PNG **texture layers** (extractable by signature
scan, no schema) and treat `data.bin` params as a Phase-1 target, not a casual read.

## As a worked example

The Fantasy RPG cloak is a ready-made end-to-end test of the whole skill: multi-part segmentation
(`cloak_upper`/`under`/`hood`/`shoulder`) × colorways (retexture path) × a real cloth-physics cloak
(springbone synthesis + tuning, see [springbone-physics.md](springbone-physics.md#tuning-by-cloth-type))
× 2048² MToon layers with explicit shade (Klein output contract). If you want a concrete target to
validate the pipeline against, this is it.
