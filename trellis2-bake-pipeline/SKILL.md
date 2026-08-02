---
name: trellis2-bake-pipeline
description: The TRELLIS.2 Swift mesh+bake stage (mlx-trellis2-swift ≥v0.8.0) — MeshBake's dual-grid→remesh→simplify→unwrap→rasterize→bake→inpaint chain, the parallel-xatlas default unwrap, the provenance fallback backend, and the measurement/debugging methodology that got the unwrap from 11 h to 24.8 s. Use whenever working on MeshBake, uvUnwrap/uvUnwrapParallel, UV atlases, chart segmentation, texture baking, GLB export quality, o-voxel double-shell artifacts (shards / seam cracks / dark flecks / streaks), Simplify collapse behavior, bakeab/unwrapbench harnesses, or planning the TRELLIS workflow metrics harness. Trigger phrasings — "MeshBake", "unwrap backend", "uvUnwrapParallel", "bake artifacts", "seam cracks", "atlas utilization", "bakeab", "unwrapbench", "TRELLIS bake quality", "double shell", "chart segmentation slow", "metrics harness for trellis". Encodes hard-won debugging lessons (viewer mode-splitting, oracle comparison, metric-vs-renderer gaps) — read BEFORE re-deriving any of them.
---

# TRELLIS.2 bake pipeline (mlx-trellis2-swift)

The mesh+texture stage that turns decoder voxel output into a textured GLB.
Everything here was measured, not assumed — sources: `UV-UNWRAP-METAL-PLAN.md`
(mlxengine-3d root, the full campaign record) and the `xatlas-bottleneck-profile`
memory. Released as **v0.8.0** (app consumes `upToNextMajor 0.7.0` from GitHub).

## Pipeline shape (MeshBake.run)

```
decoder feats → DualGridMesh.extract (raw fine double-shell, ~2.4M faces, NON-manifold,
                winding NOT unified — never trust face-normal-orientation metrics on it)
→ remeshDualContouring(256)   (watertight; FUSES components into one giant blob)
→ simplify(targetFaces)        (QEM + normal-flip guard; can still WALL-STITCH → non-orientable patches)
→ unwrap                       (default: uvUnwrapParallel; flag: provenance)
→ UVRasterize (5-pass jittered ≈ conservative) at 2× supersample → box-average down
→ per-texel: BVH closest-point remap to RAW shell → GridSample3d trilinear (baseColor + metallic/roughness)
→ dilateInpaint TO COMPLETION (never leave black texels — mips average them in)
→ GLTFExport (doubleSided:true REQUIRED for thin shells; baseColor + metallicRoughness PNGs)
```

## Unwrap backends

- **`.xatlas` (default) = `Mesh.uvUnwrapParallel()`**: BSP-partition faces by centroid
  into ~16 balanced buckets → concurrent `Atlas` per bucket at a SHARED
  `texelsPerUnit` → merge parameterizations into ONE atlas via `addUvMesh`
  (pack-only). 64.8–510.6× over single-instance at chart quality within ~5%
  (superlinear merge loop → partitioning pays beyond core count). Union repack
  packs tighter than single-run (coverage 0.54→0.68–0.75). Auto-falls back to
  direct `uvUnwrap()` under ~64k faces.
  - Per-COMPONENT partition is useless: the DC remesh fuses everything into one
    ~70% component. Spatial BSP is the working decomposition.
- **`.provenance`** (`Trellis2Configuration.unwrapBackend = "provenance"`): 6-way
  normal/grid tags → CC → per-chart axis projection → pack-only. Same speed
  class now; slightly worse error (2.2% vs 1.6% bad); kept as fallback and for
  fabric/hair-class content where seams hide.
- xatlas has exactly ONE cheap "provided route": `AddUvMesh` (bring your own
  UVs, it packs). `ParameterizeFunc` hook does NOT skip the cost (segmentation
  is 99.98%). ChartOptions tuning maxes at −30%. Forking its C++ merge loop:
  don't — it is algorithmically serial.

## Sharp edges (each cost a debugging session — do not rediscover)

1. **UvMeshDecl welds coincident UVs** (no positions exist to disambiguate) —
   overlapping chart UVs FUSE into folded islands → atlas-crossing streak
   artifacts. Always offset charts/buckets into disjoint UV regions before
   `addUvMesh`. `faceMaterial` alone does NOT prevent the weld.
2. **Zero-UV-area faces get DROPPED by xatlas, their verts left UNPACKED at raw
   input coords** → normalize near origin → corner-anchored streak slivers.
3. **Simplify wall-stitching**: collapses can seam the o-voxel shell's two walls
   into locally NON-ORIENTABLE patches (~10k winding-bad edges at 300k faces).
   Global `unifyFaceOrientations` cannot fix non-orientable. Full xatlas
   silently repairs via per-chart winding normalization — any pack-only path
   must do the same (flip faces opposing the chart's signed axis).
4. **The normal-flip collapse guard** (Simplify.swift) prevents shard
   poke-through; it does NOT prevent wall-stitching (see 3).
5. **doubleSided:true is mandatory** in exports AND in any debug viewer's
   material overrides — thin-shell o-voxel geometry shows hole/backface shards
   without it. A viewer that replaces materials and drops `side` will fake
   artifacts that don't exist in the GLB.
6. **Inpaint to completion**, not N rings: black texels poison the mip chain.
7. **Raw dual-grid winding is not unified** — dot(final-normal, raw-normal)
   "wrong-wall" metrics on it are coin flips.
8. **fast_simplification (Python lane) floors ~264k** on dense MC meshes;
   trimesh `fill_holes` never achieves watertightness (~7.5% of boundary edges)
   and runs TWICE in postprocess_cpu.py.
9. **cwd resets between Bash commands** — always explicit `cd` for git publish.

## Measurement methodology (reuse for the workflow metrics harness)

- **Harnesses (in-repo, keep green):**
  - `unwrapbench` — stage-split unwrap timing on mesh files (bench-only PLY/GLB
    readers), `--parallel N`, `--provenance`, `--golden` round-trip, winding audit.
  - `bakeab` — end-to-end bake A/B vs **voxel ground truth** (surface samples →
    baked color via UVs vs trilinear attr-volume truth), per-face-size bad-rate
    buckets, texel-class diagnostics (inpainted/wall-flip), diagnostic
    color-coded GLBs, cross-backend probe (read bake B at bake A's bad points),
    atlas PNG dumps. Args: `octant | target N | atlas N | remesh N | only <backend>`.
  - **three.js A/B viewer** (recreate from plan-doc history; scratchpad is
    volatile): the decisive instrument. **Mode-splitting isolates defect layers:
    unlit = texture content; flat-lit gray = geometry/winding; normals view =
    orientation; mips OFF = filtering; FrontSide vs DoubleSide = holes vs flips.**
- **Meta-lessons:** scalar/mip-0 metrics MISS renderer-visible defects (thin-line
  populations, mip interactions) — always end with eyes on a lit render. The
  official TRELLIS.2 space GLB is the reference oracle (structure-compare with
  trimesh: double-wall %, islands, material stack). Ground-truth evals that
  share machinery with the bake are circular exactly where the machinery is
  wrong. State the xatlas LANE (pypi vs cumesh vs Swift) with any timing; xatlas
  wall time varies >15× on nominally identical inputs.

## Known state & next targets (for the workflow metrics harness)

At v0.8.0, res512 engine e2e ≈ 753 s of which MeshBake ≈ 25 s — the bake is no
longer the bottleneck; flow sampling (SLat DiTs) dominates. Bake-internal
remaining costs: supersampled GridSample3d + CPU inpaint/downsample loops
(seconds). Known quality gaps vs oracle: none material (baseColor+MR at 2048
matches); geometry tier (512 vs 1024/1536) drives detail. Reference numbers
live in UV-UNWRAP-METAL-PLAN.md tables.
