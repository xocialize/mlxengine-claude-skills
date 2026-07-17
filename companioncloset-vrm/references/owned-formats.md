# Owned formats — `.vrmxa` and `.vrmxw` (xwear is an export tier)

Decided 2026-07-17. The platform uses its OWN openly-documented avatar + wearable formats
instead of building on VRoid's proprietary `.xavatar`/`.xwear`. Specs of record live in the
`vroid-xwear-interop` repo — this is the pointer + the essentials.

| Ext | Name | Container | Spec (repo) | Replaces |
|---|---|---|---|---|
| **`.vrmxa`** | VRM-eXtended **Avatar** | VRM 1.0 GLB | `base/CANONICAL-COMPANION-BASE.md` | loose `.vrm` |
| **`.vrmxw`** | VRM-eXtended **Wearable** | skinned glTF GLB (no `VRMC_vrm`) | `base/WEARABLE-SPEC.md` | loose `.glb` items |

Umbrella: `FORMATS.md`. Both are valid GLB (content-load regardless of extension) — the
extension signals **spec conformance**, not just "a VRM/GLB". Not `.vrma` (that's VRM Animation).

## Why our own

`.xwear` is a VRoid Studio format (proprietary zip, no freely-implementable spec). We already
produce/consume the equivalent capability natively (skinned garment + anchored accessory +
springs + MToon, portable across bases). Owning the spec lets us document/version/tool it.

## The canonical avatar (`.vrmxa`) — girl-base

**Not a generic VRM — a FIXED composition** (`base/CANONICAL-COMPANION-BASE.md`): exactly 2
meshes (Body 1 prim / Face 7 prims + 57 `Fcl_` morphs), 8 materials with the canonical texture
set + alpha/render-queue layering (iris/eyeline BLEND q−2, highlight q−1 — harvested from the
VRoid Studio export; flattening to OPAQUE breaks the eyes), 54-bone humanoid + eye bones, 18
presets + 52 PerfectSync customs, **NO physics** (base ships as a bald mannequin). Hair,
clothing, accessories are wardrobe layered on top — never part of the base.

## The wearable (`.vrmxw`) — portability mechanism

A standalone skinned GLB that **embeds the base rig hierarchy** so its skin/spring node indices
stay valid on any conformant `.vrmxa` (same trick `.xwear` uses). Two tiers: garment
(proximity-skinned) + accessory (anchor-bound). Item-owned springs + colliders. Provenance in
`asset.extras`. Ground-truth `.weights.json` sidecar for rig scoring. See wear-runtime.md for
how these get worn.

## Export tiers (from the app)

native `.vrmxa`/`.vrmxw` → plain `.vrm` (a .vrmxa IS a VRM 1.0) → `.xwear`/`.xavatar` compat
(lossy where our spec exceeds theirs; roadmap F-xw, via the `xwear_source` reference lane).

## Migration status

Specs are authoritative; the physical file rename (base/*.vrm→.vrmxa, garments/*.glb→.vrmxw +
all path refs) is sequenced as roadmap **F-fmt2** (all-or-nothing, run on a quiet tree). Helper:
`axf/migrate_extensions.py` (dry-run/apply). Only the conformant bases + n00-reference migrate;
upstream MASTERS stay generic `.vrm`.
