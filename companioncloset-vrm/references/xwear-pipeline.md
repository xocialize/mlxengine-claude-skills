# XWear (`.xwear`) garment pipeline

The VRoid XWear format, the neutral `Garment` intermediary, outfit/hair segmentation conventions,
and the read/write paths that connect XWear to CompanionCloset. This is the "how outfits get in and
out" reference.

Authoritative source: the `vroid-xwear-interop` repo (`~/Development/vroid-xwear-interop/`) — format
SOLVED + VALIDATED byte-exact on 5 real garment meshes; Swift `XWearKit` reader+writer built. When
implementing, read `XWEAR-FORMAT-SPEC.md` and `TRELLIS-TO-XWEAR-PIPELINE.md` there for byte-level
detail; this reference is the conceptual map.

> **Provenance.** The XWear spec was reverse-engineered from pixiv's serializer `.cs`
> (`~/Documents/VRM/xwear_source/`: `BinaryUtil`, `XResourceMesh`, `MeshInfoReader/Writer` — commonly
> available via the Unity XWear plugin) plus the shipped plugin. Two details worth carrying: the mesh
> binary is `.NET BinaryWriter` (LE, `float32`, 7-bit-length strings), and `XResourceMesh.IndexFormat`
> is Unity `UInt16` **or** `UInt32` — a reader must branch on it, indices aren't always 32-bit.

## Table of contents
- [What .xwear is](#what-xwear-is)
- [The Garment intermediary](#the-garment-intermediary)
- [Segmentation: groups and the Inner-as-geometry trick](#segmentation)
- [Materials (MToon10)](#materials-mtoon10)
- [Coordinate basis change (the sharp edge)](#coordinate-basis-change)
- [Read path (Companion Closet)](#read-path)
- [Write path (TRELLIS → XWear)](#write-path)
- [The two open capabilities](#the-two-open-capabilities)

## What .xwear is

`.xwear` is a **ZIP** archive — pixiv/VRoid's portable dress-up garment format (package
`com.pixiv.vroid.xwear`). Layout:

```
Body/
  XItem.json/XItem.json    # material definitions (JSON) — MToon10 shader properties
  XResources/<guid>        # item manifest (JSON, no extension): humanoid map, PhysBone components, skeleton
Mesh/
  <guid>                   # one per garment group; custom .NET-BinaryWriter binary
Textures/
  <guid>.png               # RGBA PNGs, referenced by GUID from materials
```

The mesh binary is little-endian, `float32`, with .NET 7-bit-length-prefixed strings — read arrays
in the exact order the spec lists (vertices, normals, tangents, colors, uv1–uv4, boneWeights,
bindPoses, submesh indices, blendshapes, optional trailer). Use the validated reference reader
(`reference/xwear_mesh_reader.py` in the repo) as ground truth.

## The Garment intermediary

Both directions pivot on one neutral in-memory representation — a **skinned `Garment`** (expressed
with GLTFKit mesh/skin/material types):

```
Garment {
  meshes: [ SubMeshGroup {           // Tops, Bottoms, Shoes, … (XWear "groups")
    positions normals tangents uv
    submeshes: [[Int]]               // triangle index lists
    boneWeights: [(w0..3, b0..3)]    // 4 influences/vertex → humanoid bones
    blendShapes?: [...]
  }]
  skeleton: HumanoidMap              // 54-bone VRM humanoid + bind poses
  physBones?: [SpringBoneChain]      // cloth physics (skirt/cloak/hair)
  materials: [MToon]                 // + textures
}
```

XWear read/write is just (de)serialization of this; VRMDressKit composes it onto a base body;
TRELLIS/SkinTokens produce it. Keeping everything routed through `Garment` is what lets the same
outfit flow to both the three.js runtime and the Swift tooling.

## Segmentation

A costume archive carries **multiple garment groups**, each its own mesh + material(s):
`Tops`, `Bottoms`, `InnerTop`, `InnerBottom`, `Shoes`. This is the concrete segmentation that makes
outfit swapping/retexturing tractable — swap a group, keep the rest.

**The Inner-as-geometry trick (important):** `Inner*` groups (gloves, leggings, inner layers) are
promoted to **real geometry** rather than painted into the body skin texture. This is how XWear
avoids the tone/matting artifact that baked-VRM export creates — when you paint gloves into the body
skin, you fight the wearer's skin tone; as separate geometry with their own material, they composite
cleanly over any skin tone. A **Body `_SKIN`** material with empty `_MainTex`/`_ShadeTex` + a normal
map is the transparent skin-passthrough that lets the wearer's own skin show through.

Material names mirror VRoid conventions (`…_SKIN`, `…_CLOTH`, e.g.
`N00_000_00_Body_00_SKIN(Clone)`), useful for identifying a group's role programmatically.

## Materials (MToon10)

XWear materials use the **`VRM10/MToon10`** shader. Properties are typed JSON entries
(`ShaderFloatProperty` → `Value`, `ShaderColorProperty` → `Color`, `ShaderTextureProperty` →
`TextureGuid` → `Textures/<guid>.png`). Key slots: `_MainTex`, `_ShadeTex`, `_BumpMap`, `_Color`,
`_ShadeColor`, `_AlphaMode` (0 opaque / 1 cutout / 2 transparent), `_Cutoff`.

This is the same MToon model as the VRM runtime — see [mtoon-texturing.md](mtoon-texturing.md) for
the slot semantics, color spaces, and the black-shade trap. The XWear writer is where Klein's
generated textures land in `_MainTex`/`_ShadeTex`/`_BumpMap`.

## Coordinate basis change

**XWear stores Unity-native values: left-handed, Y-up, Z-forward, meters.** VRM/glTF is
right-handed. Every read and write **crosses this boundary** and must apply the standard basis
change:
- **negate X** on positions/normals/etc.
- **flip triangle winding** (reverse index order per triangle)
- **adjust tangent.w** (handedness) accordingly

Do this at the glTF boundary; keep XWear-side data Unity-native. Getting this wrong yields
inside-out meshes, wrong-facing normals, or mirror-flipped garments — the classic symptom is a
garment that looks right but lights/culls backwards. Validate visually against a known-good sample.

## Read path

`.xwear` → unzip → parse mesh binary + JSON manifests + PNG textures → `Garment` → GLTFKit model →
**VRMDressKit** fits/composes onto a base body → renders in **Companion Closet** (three.js /
`@pixiv/three-vrm`). Apply the Unity→glTF basis change during decode. Map each mesh's bone indices
to humanoid bones via the manifest's `XResourceHumanoidMap`.

## Write path

TRELLIS effort emits a spec-valid `.xwear` from a generated `Garment`:

| XWear needs | Source | Status |
|---|---|---|
| positions/normals/tangents/uv/submesh indices | TRELLIS.2 `imageTo3D` garment mesh (glTF) | ✅ exists |
| boneWeights + bindPoses (4 influences → humanoid bones) | SkinTokens `meshRig` | ⚠️ **gating** (see below) |
| `XResourceHumanoidMap` (54-bone humanoid) | base-body skeleton the garment is rigged to | derive from target body |
| MToon materials + textures | Klein-generated garment textures | ✅ |
| PhysBone chains (optional) | authored or synthesized for skirts/cloaks/hair | ⚠️ deferred (see below) |
| ZIP assembly (Body/Mesh/Textures) | `XWearKit` writer | built |

Emitting XWear puts generated outfits into the whole VRoid/BOOTH ecosystem **and** earns VRoid
Studio's auto-fit for free — the payoff for targeting this format rather than a bespoke one.

## The two open capabilities

These are the live engineering gaps in the generative-garment path (both flagged in the interop repo):

1. **SkinTokens garment-rig mode (gating).** A generated garment must be skinned to the **supplied
   base-body humanoid skeleton** (so a sleeve follows the arm bone, a skirt the hips) — *not* an
   invented skeleton. SkinTokens `auto` mode generates a new skeleton; XWear needs the garment
   weighted against the given humanoid bones (target = base-body bones). This gates the whole path;
   de-risk first. UniRig (see [springbone-physics.md](springbone-physics.md#auto-rigging-tools)) is
   the relevant class of tool, but confirm it can target a *supplied* skeleton, not only invent one.

2. **Spring-bone synthesis for generated skirts/cloaks/hair (deferred).** Generated geometry has no
   bones; author `VRMC_springBone` chains + colliders procedurally — see
   [springbone-physics.md](springbone-physics.md#synthesizing-bones-for-a-generated-garment). Can be
   synthesized at package time (baked into the `.xwear`) or built at load in three.js; CompanionCloset
   may do both.

## Related: .vroid

VRoid's native `.vroid` project format (ZIP + Protocol Buffers) is a **richer** source with
parametric params and **separated transparent texture layers**. Import is **parked** — only its
easy high-value part (extracting the separated PNG layers via signature scan, no schema needed) is
worth grabbing for the retexture path; the mesh protobuf RE is deferred. See `VROID-FORMAT-PLAN.md`
in the interop repo. Its clean separated layers are also the best available source for answering the
open UV-template question in [mtoon-texturing.md](mtoon-texturing.md#baking-onto-existing-uvs).
