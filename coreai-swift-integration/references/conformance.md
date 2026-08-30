# Conformance — the gate that does not exist yet

## Honest status

`Skill("mlx-swift-integration")` carries a **C0–C13 conformance gate**: a checklist a package
must pass to be engine-pluggable, derived from many shipped MLX packages and hardened by
packages failing it.

**CoreAI has no equivalent.** We have shipped **one** package. A gate invented from one example
would encode that example's accidents as requirements — which is worse than no gate, because it
would look authoritative.

So this file is a **derivation plan**, not a gate.

---

## Candidate axes, from the one package we have

Observations from `coreai-realesrgan-swift` that *might* generalize. Each needs corroboration
from at least two more packages before it becomes a numbered requirement.

| Candidate | Why it might be load-bearing | Corroborated? |
|---|---|---|
| Static-shape geometry declared as a **build-time** property | The inverse of the MLX contract; a consumer must not assume runtime injection | 1 package |
| First-load E5RT cost declared and paid at the **prepare seam** | 8 s → 254 s measured range; a product-visible constraint | 1 package |
| Placement **proven**, not requested, before any perf claim in the manifest | Silent GPU fallback is the default failure | 1 package |
| Asset vendored as a **directory** resource (`.copy`), or downloaded to the shared store | `.aimodel` is a directory; `.process` is wrong | 1 package |
| Input contract fully specified — name, shape, dtype, layout, range | Static assets reject mismatched dtype; no negotiation possible | 1 package |
| Memory reported as **resident + peak**, not one number | 19 MB vs 0.86 GB on the same run | 1 package |
| Registered via injection seam, never by a hard dependency from a lower-floor host | SPM refuses 26→27 dependencies | 1 package |

---

## Derivation plan

1. Ship **3–4** CoreAI packages across different model classes (Phase 2–3 of the program).
2. After each, re-read this table and mark which candidates held, which broke, and which were
   accidents of the first package.
3. Only then number them into a gate.
4. **Re-audit every earlier package against the gate.** A gate that has never failed anything
   has not been tested — it has only been written.

---

## What the MLX gate has that we cannot yet copy

Listed so the gaps stay visible rather than being quietly assumed away:

- **QuantFootprint** — CoreAI's memory shape is fundamentally different (on-die activations), so
  the MLX field does not transfer. What the engine should be told is **OPEN**.
- **MAT gate** (auto-materialization / WeightSourcing) — no CoreAI analogue attempted.
- **CAN gate** (cancellation) — unknown whether an in-flight CoreAI inference can be cancelled
  at all. **Untested.**
- **Capability vs mode vs specialty** routing — undesigned for a two-runtime capability.
