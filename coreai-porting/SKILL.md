---
name: coreai-porting
description: Port PyTorch models to Apple Core AI (.aimodel) for on-device inference on Apple silicon, with the Neural Engine as a first-class target. Covers torch.export capture, coreai-torch lowering, missing-lowering triage and custom lowerings, fp16/fp32 precision failures, ANE eligibility (rank<=5, einsum folding), and — critically — PROVING compute-unit placement rather than requesting it. Also carries the CoreAI-vs-MLX fit journal for deciding which runtime (or both) a model belongs on. Trigger phrasings — "port to CoreAI", "CoreAI export", ".aimodel", "coreai-torch", "TorchConverter", "run on the ANE", "Neural Engine", "ANECCompile", "is this ANE-eligible", "CoreAI or MLX", "which runtime for this model", "AIProgram", "save_asset", "E5RT". Invoke eagerly — CoreAI ports fail SILENTLY: a requested compute unit can fall back without raising, and the specialization cache then caches the fallback. Runs BEFORE `coreai-swift-integration` (which packages the finished asset). For the vendored Apple reference material see `Skill("working-with-coreai")`.
---

# Core AI porting

Ours, not Apple's. `Skill("working-with-coreai")` is the vendored Apple skill — API surface,
official docs map, onboarding protocol. **This skill is what we measured that the docs don't
say**, and it is where every new CoreAI finding lands.

> **Why this skill exists.** Until 2026-08-29 our CoreAI findings were appended to Apple's
> vendored `working-with-coreai/SKILL.md`, whose own header says local edits are silently lost
> on refresh. They now live here. Never write CoreAI findings into the vendored file.

**Related:** `Skill("coreai-swift-integration")` (packaging a finished `.aimodel` into Swift /
MLXEngine) · `Skill("working-with-coreai")` (Apple's reference) · `Skill("model-authoring")`
(re-structuring architecture for ANE) · `Skill("model-compression-exploration")` (quant /
palettization sweeps) · `Skill("mlx-porting")` (the MLX sibling — read together when deciding
which runtime a model belongs on).

---

## Evidence status convention

Every claim in this skill and its references carries one of these. **Do not promote a claim a
tier without a measurement.**

| Tag | Meaning |
|---|---|
| **MEASURED** | We ran it, on named hardware and OS, with the numbers recorded |
| **INHERITED** | Taken from upstream docs/code and NOT independently verified — treat as a lead |
| **ASSUMED** | Plausible, unmeasured. An open question, never a basis for a design decision |
| **OPEN** | We know we don't know. Named so it can be closed later |

Our own history is the argument for this: an upstream comment presented `patch_nearest_upsample`
as blanket hygiene; MEASURED on SRVGG it was neutral on GPU and **16% slower on ANE**. Inherited
recipes get A/B'd, not adopted.

---

## The two rules that cost us the most

**1. A preference is not a placement.** Requesting a compute unit does not mean you got it, and
the runtime never volunteers what it did. There are two failure modes and only one is loud.
→ `references/placement-and-residency.md`. Read this before any benchmark.

**2. Don't accept first-working.** The objective is learning what is *optimal*, not what runs.
Every graph rewrite, precision choice, and tile geometry gets at least one alternative measured
against it, and the loser is recorded with its numbers. A rewrite that "works" and a rewrite
that is *better* are different findings. → `references/measurement-protocol.md`.

---

## Pipeline

```text
0. FIT        CoreAI, MLX, or both?              → references/mlx-vs-coreai-fit.md
1. EXPORT     torch.export → coreai-torch        → references/export-recipes.md
2. PRECISION  fp32 proves the graph, fp16 ships  → references/precision.md
   COMPRESS   quantize / palettize / prune        → references/compression.md
3. ELIGIBLE   will the ANE even accept it?       → references/ane-eligibility.md
4. PLACE      prove where it ran                 → references/placement-and-residency.md
5. MEASURE    parity, latency, energy, memory    → references/measurement-protocol.md
6. PACKAGE    Swift / MLXEngine / publish        → Skill("coreai-swift-integration")

   op has no lowering?                     → references/custom-lowerings.md
```

When a step fails and the error is opaque — which is the normal case —
→ `references/debugging-methodology.md`.

---

## Export recipe that works

MEASURED (macOS 27.0, `coreai-core==1.0.0b2`, `coreai-torch==0.4.1`), used for both shipped
ports. Run it in an isolated `uv` venv — `coreai-torch` pins torch to 2.11.x.

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(dummy,))          # static shapes only for ANE
ep = ep.run_decompositions(get_decomp_table())
program = (TorchConverter()
           .add_exported_program(ep, input_names=["x"], output_names=["out"])
           .to_coreai())
program.optimize()
program.save_asset(Path("model.aimodel"))               # wants a Path, not a str
```

> ⚠️ **`program.optimize()` is not free.** Upstream `coreai-torch#49` reports it silently
> miscompiling broadcasting-significant axis moves, and `#33` was a segfault in it. **Export both
> ways and parity-check each** — see `references/known-upstream-defects.md`.

`save_asset` writes a **directory**, and `main.mlirb` is roughly checkpoint-sized (MEASURED:
431 MB fp16 / 862 MB fp32 for a 226M UNet). Gitignore the exports dir before the first
`git add -A`, not after the push is rejected.

Details, missing lowerings, and per-family graph preparation →
`references/export-recipes.md`.

---

## First-load cost is a design constraint, not a detail

MEASURED: E5RT specialization on first load per (model × machine) ranged **~8 s** (SRVGG, 1.4M)
to **254 s** (Moebius UNet, 226M). OS-cached after (then ~0.2–1.1 s). Budget it at the
package's `load()`/prepare seam and warn the user the first time — never inside the first
user-visible inference.

The same cache is also a diagnostic hazard: **it caches failure-then-fallback**. See
`references/placement-and-residency.md`.

---

## Reference map

| File | Read it when |
|---|---|
| `references/placement-and-residency.md` | Before ANY benchmark or ANE claim. **The frontier area — we have the least experience here.** |
| `references/ane-eligibility.md` | The graph won't compile for ANE, or you're designing for it |
| `references/precision.md` | fp16 parity is bad, or mixed precision won't legalize |
| `references/compression.md` | Quantization / palettization / pruning via `coreai-opt` |
| `references/export-recipes.md` | Capture or lowering fails; per-family graph prep |
| `references/known-upstream-defects.md` | **Before every port** — the harvested failure-mode catalogue from `apple/coreai-torch` issues |
| `references/case-eomt-capture.md` | A `GuardOnDataDependentSymNode` capture failure, root-caused and fixed — includes the grep-the-graph-dump technique |
| `references/case-deformable-conv.md` | A complete worked triage-ladder pass — decompose-in-PyTorch beating a custom lowering, and how scale changed the ANE verdict |
| `references/custom-lowerings.md` | An op has no lowering — `register_torch_lowering`, `TorchMetalKernel`, the supported-ops list, composite ops |
| `references/runtime-api.md` | Python runtime sharp edges, toolchain versions, asset layout |
| `references/measurement-protocol.md` | Parity thresholds, benchmark discipline, the A/B rule |
| `references/debugging-methodology.md` | The error is opaque, redacted, or moves between runs |
| `references/mlx-vs-coreai-fit.md` | Deciding which runtime a model belongs on |
| `references/gradebook.md` | The sealed-port protocol and what each port taught |

## Scripts

`scripts/placement.py` — canonical placement helpers. Builds verified `SpecializationOptions`,
times a lane, scans stderr for `ANECCompile` / `ane_validation_message`, and flags lanes whose
latencies agree (two lanes with equal latency are the same lane). Run it standalone to print
this machine's placement surface. **Use it instead of hand-rolling options** — it exists because
three separate API traps each produce plausible mislabeled data.
