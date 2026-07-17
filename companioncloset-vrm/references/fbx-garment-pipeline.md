# FBX garment → onto the humanoid base (native extraction + retarget + spring synthesis)

The **built, proven** creation path (supersedes the aspirational TRELLIS→SkinTokens→XWear plan for
sourcing base garments): take a rigged garment from an FBX, extract it natively, retarget onto the
canonical N00 base, synthesize physics, texture it. Validated end-to-end on a magical-girl dress + hair
+ wings onto a VRoid nude base, rendered in three-vrm with working cloth physics.

Tools: `~/Development/FBXItemKit/` (native Swift FBX reader — zero deps), `axf/axf_springs.py`
(spring-aware bridge), `axf/uv_mask.py`. Item JSON = the handoff (`ItemExporter`).

## 1. Native FBX extraction (FBXItemKit) — the reader contract
Bounded FBX-binary subset only (GlobalSettings, Geometry, Model, Deformer/Cluster skin, Material/
Texture, Connections graph). The value is the VRM-compat emitter, NOT the byte-parsing.
- **Targeted multi-mesh extraction:** `ItemExtractor.extract(scene, targetMesh:"aHat")` selects ONE
  named mesh from a character FBX and **scopes the skin to that geometry's own Skin deformer**
  (Geometry←Skin←Cluster chain) — else every mesh's clusters pile onto one vertex array and scramble.
  Stock characters are usually **fused single-mesh** (clothing = texture, not geometry) → the separable
  garment often lives only in the `.blend` as a distinct object; export just that object.
- **UV + vertex splitting:** `extractGeometryUV` decodes `LayerElementUV`/`LayerElementNormal`
  (ByPolygonVertex|ByControlPoint × Direct|IndexToDirect; **FBX V → glTF 1-v**) and **splits vertices
  on (controlPoint, uv, normal)** so the result is glTF-ready (one uv+normal per position). Skin remaps
  per-control-point → split via `cpMap`.
- **Materials/textures:** Model→Material (OO) + Texture→Material (OP) → baseColorTexture filename
  (external `RelativeFilename`; embedded Video/Content not yet handled).
- **Bone hierarchy + world binds:** `SkinExtractor` returns `parents` (Model→Model OO chain) +
  `worldBinds` (Cluster.TransformLink) — needed for ancestor-fallback + spring synthesis.

## 2. Coordinate/unit normalization — THE #1 bug source (Blender FBX is inconsistent)
- **meshBind for the IBM must come from the mesh Model node's WORLD transform**, not `Cluster.Transform`.
  Blender writes `Cluster.Transform` in a *different unit* than `Cluster.TransformLink` (meters vs cm),
  and may leave it identity. The Model node's `Lcl Translation/Rotation/Scaling` (composed up parents)
  is spec-correct and **unit-consistent with the bone TransformLinks**, so `IBM = inverse(TL)·meshModelWorld`
  cancels cleanly (det≈1, meter-scale) AND bakes any -90°X Z-up→Y-up conversion for free.
- **Geometry-vs-transform unit split:** Blender may export geometry in cm but transforms as identity
  (or vice-versa) with a 100× object scale hiding on the node. **Fix at export:** re-export with
  `bake_space_transform=True, apply_scale_options='FBX_SCALE_ALL', axis_up='Y', axis_forward='-Z'` so
  geometry AND skeleton land in the same Y-up space (may still be cm → bridge at `scale=0.01`).
- **Scale positions AND IBM translation by the same factor** (cm→m) so bone-local coords stay
  meter-scale (`axf_springs` scales IBM cols 12/13/14 with positions).

## 3. Retarget onto canonical N00
LBS bind retarget: `v_A = A_bind[bone] · IBM_source[bone] · v` maps garment geometry from source-bind
into N00-bind space. `A_bind` = N00 humanoid bone world transforms. Works ≈identity when source and N00
share the bind pose; residual misfit for proportionally-different bodies (dedup POSITION accessors —
VRoid meshes share one POSITION across primitives, else the deform compounds per-submesh → explosion).

## 4. Spring-bone SYNTHESIS from dynamic bones (`axf_springs.py`) — the proper fit
Classify every source bone: **humanoid** (bind to base node) | **spring** (name matches
`cloak|skirt|hair|cloth|ribbon|tail|breast|scarf|cape|sleeve|string` → synthesize a node) | **rigid**
(twist/finger helper → bind to nearest humanoid ancestor). **Skin EVERY vertex to a node — no geometry
baking.** Skinning identity (meters): `node.world · IBM_j · v = correct rest`, because synth nodes sit
at `RigidDelta[anc]·sourceWorld[j]` (`RigidDelta[anc]=A_bind[anc]·inv(sourceWorld[anc])`) and IBM_j =
source IBM. Synth nodes parent under the humanoid ancestor's node (Cloak→upperChest, Skirt→hips), local
TRS decomposed from `inv(parentWorld)·synthWorld`. Emit `VRMC_springBone` chains (linear ancestor→child;
branches split, anchored by shared parent). Add body colliders (torso spheres + leg capsules) in one
group referenced by every spring so skirts collide instead of clipping.

**Spring PROFILES (`cloth|hair|stiff|rigid`, CLI 6th arg):** classification is bone-NAME-based and
misfires when a non-cloth mesh rides cloth-named bones — e.g. a **wing** weighted to `Cloak.*` droops
under cloth gravity. `stiff` (stiffness 1/grav .03) holds shape but flutters; `rigid` skips springs (bind
dynamic bones rigidly to ancestor → holds exact rest shape). Long-term: infer profile from mesh
role/material (`wing`→stiff, `hair`→hair, garment→cloth).

## 5. Texture + re-embed
`axf_springs` carries TEXCOORD_0/NORMAL, resolves the texture (search `.fbm` + `Texture/` dirs), embeds
the PNG (bufferView image + sampler + texture), builds MToon (baseColor + shadeMultiply, non-black shade).
Retexture = swap the PNG and re-run (`axf_springs ... texdir`). See klein-retexture.md.

Run (single): `python3 axf_springs.py <item.json> <base.axf> <out.vrm> 0.01 "<texdir1:texdir2>" [profile]`.

## 6. Multi-item compose (full avatar in one pass)
`bridge_multi(items, base_axf, out_vrm, scale)` composes MANY items (dress + hair + wings + …) onto one
avatar: base composed ONCE, each item via `_add_item` with its own profile + texdirs + material name,
body colliders added ONCE and referenced by every item's springs, expressions remapped ONCE. `bridge()`
is now a 1-item wrapper. Run: `python3 axf_springs.py multi <base.axf> <out.vrm> <scale>
item.json,profile,texdir1:texdir2 ...`. Proven: Klein-blue dress(cloth)+bob hair(hair)+feathered
wings(stiff)+green eyes(`retex_material.py`) = one finished anime avatar, 87 spring chains, conformant.
Shared textures **dedup by sha256(png)** across items (content-addressed cache); each item mesh gets a
`firstPerson` role (hair→`thirdPersonOnly`, else `auto`) → clears the segmentation firstPerson WARN.
