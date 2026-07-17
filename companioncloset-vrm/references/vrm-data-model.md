# VRM 1.0 data model

How a `.vrm` is structured, and how its extensions reference glTF data. All VRM 1.0.

## Table of contents
- [.vrm is glTF 2.0 + extensions](#vrm-is-gltf-20--extensions)
- [The four extensions](#the-four-extensions)
- [Index-based references](#index-based-references)
- [VRMC_node_constraint](#vrmc_node_constraint)
- [FirstPerson (per-mesh visibility)](#firstperson-per-mesh-visibility)

## .vrm is glTF 2.0 + extensions

A `.vrm` file is a **glTF 2.0 binary** (`.glb` container — JSON chunk + binary buffer chunk). It
is a valid glTF; all VRM-specific meaning lives in glTF **extensions**. A plain glTF viewer renders
the meshes; a VRM runtime additionally interprets the extensions for humanoid rig, materials,
physics, and constraints.

This matters for the pipeline: retexturing or swapping an outfit is **rewriting extension JSON +
buffer data and re-indexing**, not restructuring a scene graph. The mesh topology and UVs are
ordinary glTF; the avatar semantics are a layer on top.

## The four extensions

**`VRMC_vrm`** (core, required) — the avatar definition. Components:
- `meta` (required) — title, author, license
- `humanoid` (required) — maps human body bones (hips, spine, leftUpperArm, …) onto glTF **nodes**.
  This is the retargetable skeleton definition.
- `firstPerson` (optional) — per-mesh camera visibility (see below)
- `expressions` (optional) — blendshape-driven expressions (VRM's morph/expression system)
- `lookAt` (optional) — eye-gaze / head-aim configuration

**`VRMC_springBone`** — secondary-motion physics (hair, cloth, accessories). Full treatment in
[springbone-physics.md](springbone-physics.md).

**`VRMC_node_constraint`** — roll/aim/rotation constraints on nodes (twist bones, secondary bones).
Detailed below.

**`VRMC_materials_mtoon`** — the toon shading model. Full treatment in
[mtoon-texturing.md](mtoon-texturing.md).

Plus standard Khronos extensions a VRM commonly uses: `KHR_materials_unlit`,
`KHR_texture_transform`, `KHR_materials_emissive_strength`.

These are **four sibling glTF extensions**, each with its own spec, not sub-objects of one blob.
(This is the key structural difference from legacy VRM 0.x, which packed everything into a single
monolithic `VRM` extension. CompanionCloset is 1.0-only, so you work with the split form.)

## Index-based references

Everything cross-references by **integer index** into the glTF arrays — the same convention glTF
uses internally:
- `humanoid` bones → glTF **node** indices
- spring joints, colliders, constraints → glTF **node** indices
- materials (incl. MToon) → glTF **material** indices
- textures → glTF **texture** indices, which reference **image** + **sampler** indices
- a texture reference carries an optional `texCoord` (default `0`) selecting the primitive's
  `TEXCOORD_n` UV attribute

**Consequence for retexturing:** an MToon texture slot points at a glTF texture that samples the
mesh's *existing* `TEXCOORD_0`. So a newly generated map is baked onto the mesh's existing UV
layout — you don't re-UV, you replace the image the index resolves to. (See the UV caveat in
[mtoon-texturing.md](mtoon-texturing.md) — VRoid's specific UV template regions are an open gap to
determine empirically.)

## VRMC_node_constraint

Extends a glTF **node** with a constraint. Root object requires `specVersion: "1.0"` and exactly
**one** `constraint` object. There are exactly **three** constraint types (a node has one):

**Roll** — transfers twist (roll) from a source bone; used for twist/roll bones in limbs so a
forearm twist distributes smoothly.
- `source` (int, required, ≥0) — node index providing the rotation reference
- `rollAxis` (string, required) — `"X"`, `"Y"`, or `"Z"`
- `weight` (number, optional) — default `1.0`, range `[0.0, 1.0]`

**Aim** — orients the node so a chosen local axis points toward the source's world position;
used for look-at-like secondary bones.
- `source` (int, required) — target node index
- `aimAxis` (string, required) — six signed enum values: `"PositiveX"`, `"NegativeX"`,
  `"PositiveY"`, `"NegativeY"`, `"PositiveZ"`, `"NegativeZ"`
- computes the minimal rotation from the node's rest orientation to align the aim axis toward the
  source position, in world space
- `weight` (optional) — default `1.0`, `[0.0, 1.0]`

**Rotation** — copies the source node's local rotation onto the destination (scaled by weight).
- `source` (int, required)
- `weight` (optional) — default `1.0`, `[0.0, 1.0]`

> Note: an early research claim about a `@pixiv/three-vrm` class named `VRMRotationConstraint`
> was **refuted** — do not rely on that specific class name. Author against the extension spec
> (`vrm-c/vrm-specification/VRMC_node_constraint-1.0`); check the current three-vrm
> `three-vrm-node-constraint` module for the actual runtime class names if you need them.

**Relevance to CompanionCloset:** node constraints matter only if generated garments carry
twist/secondary bones you want auto-driven. For most outfit/hair work you won't author these — but
you must **preserve** any that exist when reading and rewriting a VRM, or limb twisting breaks.

## FirstPerson (per-mesh visibility)

`VRMC_vrm.firstPerson` annotates each mesh with a visibility flag controlling first- vs
third-person camera rendering. Four values:
- `Auto` — runtime auto-splits by relationship to the head bone
- `Both` — always visible
- `ThirdPersonOnly` — hidden from the first-person camera; **recommended for head and hair** so
  they don't obscure the view in first person
- `FirstPersonOnly`

This is also concrete evidence that VRM avatars are **partitioned into distinct meshes** (head,
hair, body, garments) with independent material assignment — the structural basis for swapping and
retexturing outfits independently. Segmentation conventions in [xwear-pipeline.md](xwear-pipeline.md).
