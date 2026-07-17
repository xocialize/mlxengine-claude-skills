# AXF — Avatar eXchange Format (our decomposable whole-avatar format)

AXF is CompanionCloset's own **complete-avatar** format: a decomposable avatar (separable
body/face/hair/outfits, layered textures, humanoid-relative rig, content-addressed assets) on an
**open glTF/VRM substrate** we control and can redistribute. It matches VRoid `.vroid`'s decomposability
but is open, so it's the master format; VRM is a **disposable compose target** (render artifact).

Full spec: `~/Development/vroid-xwear-interop/AVATAR-EXCHANGE-FORMAT.md`. Python impl: `axf/axf_tool.py`
(segment/extract/compose), `axf/axf_retarget.py`, `axf/axf_springs.py` (spring-aware bridge). Swift:
`~/Development/AXFKit/`. Conformance: `conformance/conformance_check.py`.

## Why it exists
- **VRM** is a baked snapshot: composing body+outfit fuses meshes, atlases textures, binds springs → no longer decomposable.
- **`.vroid`** IS decomposable but proprietary, protobuf, and license-locked (can't modify/redistribute).
- **XWear** solved single-component transfer (one garment). AXF promotes that to the **whole avatar**: body+face+hair+N outfits, each still separable/swappable/retexturable, sharing one skeleton.

## Container (ZIP)
```
manifest.json          # components, variants, skeleton ref, meta
skeleton/humanoid.json # shared VRM 54-bone humanoid: bone→node, rest pose, bind matrices
skeleton/skeleton.gltf.json + its buffer
components/<id>.json   # body | face | hair | outfit | accessory — one per component
assets/<sha256>        # content-addressed blobs (mesh buffers, PNG textures) — dedup automatic
```
`manifest.variants` = named alternative compositions (`default`/`nude`/`dressed`/...) — the
decomposability payoff. `compose(manifest, variant) → VRM 1.0` selects components, resolves assets,
binds skins to the shared humanoid, aggregates expressions/firstPerson/lookAt, emits a standard VRM.

## Data-model musts (learned the hard way)
- **Expressions are REQUIRED, not optional** (§4.5). A companion avatar must blink + lip-sync → the
  face component carries morph targets for the **14 VRM presets** (`neutral happy angry sad relaxed
  surprised` + visemes `aa ih ou ee oh` + `blink blinkLeft blinkRight`) and an expression map;
  `compose()` **remaps morph-target binds to the merged node indices** (`_remap_expressions`) — a
  compose that skips this leaves binds pointing at stale node indices (conformance facet 2.9 FAIL).
- **Materials carry a `kind`** — `SKIN | FACE | EYE | CLOTH | HAIR` + canonical `N00_000_00_<Part>_00_<TYPE>`
  naming → deterministic segmentation + MToon-slot assignment (see n00-base-spec.md).
- **firstPerson + lookAt** are compose-time VRM fields (head/face/hair = `thirdPersonOnly`).
- **Morph-target `targets` accessors MUST survive merge** — dropping them crashes three-vrm bounds
  computation ("Cannot read properties of undefined (reading 'min')"). The integrity facet catches it.

## Producer contract (headless Blender, component-first) — see AVATAR-EXCHANGE-FORMAT.md §8
Guarantee on emit: **meters, Y-up, +Z-forward**; skeleton = canonical N00 at the fixed bind pose
(`canonical-n00/n00-bindpose.json`); MToon color spaces + non-black shade + N00 naming; morph targets
for the 14 presets; firstPerson/lookAt; provenance (CC0/permissive only for shippable). Two emit paths:
component-first (Blender→glTF→wrap into assets/+component json) or VRM-then-segment (`axf_tool extract`).

## Canonical N00 base
The shared skeleton+bind-pose that makes cross-source retarget ≈identity. Derived from the VRoid **N00**
spec. `spec-dev.axf` (from a real VRoid nude `Spec.vrm`) is the current **dev base** (VRoid-licensed →
dev/reference only). CC0 MakeHuman base (`canonical-n00-cc0.axf`) is the shippable scaffold. See
n00-base-spec.md. Cross-body outfit fit is ≈identity only when both share this bind pose.

## Related
XWear garment = 1 AXF `outfit` component (map 1:1). VRM = compose target + ingest source (segment a VRM
→ components). `.vroid` = the capability reference + a license-restricted ingest source.
