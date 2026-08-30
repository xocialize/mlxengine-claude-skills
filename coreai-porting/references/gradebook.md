# The gradebook — sealed-port protocol and per-model results

The purpose of the collection is **skill**, not throughput. Ported models are the evidence that
pays for the skill content. This file records the protocol and what each port actually taught.

---

## Sealed-oracle protocol

LibreYOLO's `libreyolo/export/coreai.py` is 771 lines of solved problems. Reading it first hands
us the answers and teaches us nothing; never reading it wastes the best grading key available.
So: **seal it, attempt it, then grade against it.**

1. **Seal.** Do not open `libreyolo/export/coreai.py` for this family. Take the PyTorch model
   from `libreyolo/models/<family>` — a clean seam, since the model source is not the export
   path.
2. **Attempt.** Port from the model source and the CoreAI docs alone.
   **Write down the raw error text BEFORE you know the cause.** That text is the single most
   valuable thing a skill can carry, and it is unrecoverable once you know the answer.
3. **Unseal and diff.** Then read their rewrite for that family.
4. **Grade three ways:**
   - **They caught it / we missed it** → a gap we would have shipped. Highest-value content.
   - **We caught it / they missed it** → our finding. Candidate upstream issue or PR.
   - **Both, differently** → measure both. That comparison is content too.
5. **Bank it.** Reference file here + a typed AgentBridge receipt.

**Sequencing note:** the *harness* diff comes after the first sealed port, not before — otherwise
their `REL_TOL` and sensitivity-margin design anchors ours and we never derive the reasoning.

**Discipline cost, stated honestly:** this is slower per model, and the temptation to peek peaks
exactly when a port is stuck — which is when the learning is worth the most. **Log the stuck
time.** It is a real measurement of difficulty and belongs in the receipt.

---

## Grading log

*One entry per sealed port. Empty until Phase 1 completes.*

| Model | Sealed? | We missed | They missed | Both, differently | Stuck time |
|---|---|---|---|---|---|
| *(pending — Phase 1: `realesrgan`)* | | | | | |

## Findings ahead of the first port (environment/toolchain research, 2026-08-29)

| # | Finding | Status |
|---|---|---|
| 1 | `ComputeUnitKind.cpu/.gpu/.neural_engine` are **staticmethod factories, not constants** — the uncalled attribute raises `Invalid ComputeUnitKind` | MEASURED → `placement-and-residency.md`, `scripts/placement.py` |
| 2 | `ComputeUnitKind.available_kinds()` order is **non-deterministic across processes** — indexing it selects a different unit each run | MEASURED (6 runs) → same |
| 3 | **Fallback off the ANE cannot be forbidden** — `allowed_compute_unit_kinds` stays all 3 for any preference; `cpu_only()` is the only restriction primitive | MEASURED → same |
| 4 | `SpecializationOptions.is_supported()` returns True **without** `USE_OS_COREAI=1`, contradicting its own docstring | MEASURED |
| 5 | The `avg_pool2d` off-by-one is **still live in coreai-torch 0.4.2**; LibreYOLO's shim covers only 0.4.1 and declines silently | MEASURED, minimal repro → `runtime-api.md` |

Findings 1–3 are the same class as the mislabeled-ANE-table incident, but at **API level**, and
finding 2 is non-deterministic — it would defeat a careful person who spot-checked once.

---

## Ports completed before the protocol existed

Not sealed — no grading key existed at the time. Recorded for completeness.

| Model | Date | Outcome | Where the findings live |
|---|---|---|---|
| Real-ESRGAN SRVGG (1.4M) | 2026-07-31 | Shipped to `coreai-community` as `Real-ESRGAN-CoreAI` | AB-R-0047; `precision.md`, `measurement-protocol.md`, `mlx-vs-coreai-fit.md` |
| Moebius UNet (226M) | 2026-08-01 | ANE blocked; filed `apple/coreai-models#138` | AB-L-0028; `ane-eligibility.md`, `debugging-methodology.md`, `precision.md` |

### Source locations

| What | Where |
|---|---|
| Real-ESRGAN Swift package + model card | `Development/mlxengine-image/PROD/coreai-realesrgan-swift` |
| Real-ESRGAN export script | `srvgg_export.py` (in that repo); also `Development/coreai-probe/scripts/` |
| Moebius CoreAI work + notes | `Development/mlxengine-image/WIP/moebius-m0/coreai/`, `COREAI-NOTES.md` |
| Upstream repro for #138 | `moebius-m0/coreai/repro_upstream.py` (public weights) |
| Probe harness, banked results | `Development/coreai-probe/` (`scripts/`, `results/srvgg`, `results/edsr`) |
| Programme context | GAP-PROGRAM V13 / V13-P / V13-E; `Development/mlxengine-todo/` |
| Vendored Apple skill (pristine) | `~/.claude/skills/working-with-coreai/` |
| Pre-migration backup of the appendix | `CoreAI/coreai-collection/.working-with-coreai.SKILL.md.pre-phase0.bak` |

**Re-test `#138` per macOS update** before assuming the ANE door is still closed.
