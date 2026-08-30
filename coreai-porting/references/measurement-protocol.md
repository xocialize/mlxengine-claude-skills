# Measurement protocol — and the rule against first-working

---

## The A/B rule

**The objective is learning what is optimal, not what runs.** Every graph rewrite, precision
choice, and tile geometry gets at least one alternative measured against it, and the loser is
recorded **with its numbers**. "It works" and "it is better" are different findings, and only
one of them is worth putting in a skill.

### The case that proves it

**MEASURED, 2026-08-01, Real-ESRGAN SRVGG, same checkpoint, 128² tile, M5 Max, median of 20
post-warmup.**

Upstream's coreai-models comment says MPSGraph rejects nearest-mode `coreai.interpolate` and
routes it to BNNS/CPU, so `patch_nearest_upsample` (repeat_interleave for nearest-×2) reads like
blanket hygiene. Measured:

| Lane | Unpatched | Patched | Verdict |
|---|---|---|---|
| GPU | 2.19 ms | 2.17 ms | neutral |
| ANE | 5.79 ms | **6.74 ms** | **16% SLOWER** |

The patched export genuinely drops the op (`strings main.mlirb` finds no interpolate) — the
rewrite *works*. It just doesn't pay for that architecture. Plausibly `repeat_interleave`
materialises the scaled tensor explicitly where the native op is handled better.

**Honesty note carried forward:** the same patch's value for the Moebius UNet is **UNMEASURED**.
It was applied there from the upstream comment and never compared against an unpatched export,
so "load-bearing" was an INHERITED claim, not a finding. Export both ways and probe each lane
before believing it.

> An inherited recipe is a hypothesis. Measure it per port, per lane.

---

## Benchmark discipline

- **Median of 20 post-warmup**, not mean, not a single run.
- **Warm-up separately from the timed loop.** First load pays E5RT specialization (8 s → 254 s
  measured range); it is not part of steady-state latency and must be reported separately.
- **All three lanes, every time** (CPU / GPU / ANE). The CPU lane is a control, not a data point
  you are allowed to skip because it's slow. → `placement-and-residency.md`.
- **Report cold vs warm cache explicitly.** They measure different things.
- **Re-measure every lane after every graph rewrite.** MEASURED failure: a "4× ANE speedup" that
  was really the new graph's GPU lane.

---

## Parity harness design

Two independent quantities, both required. A parity check with only the first is not a check.

1. **Error** — max |Δ| between artifact and reference, relative to the reference's own scale.
2. **Input sensitivity** — how much the reference output *moves* between two different inputs,
   on the same scale.

Then **margin = sensitivity / error**. A model that returns a constant passes any error
threshold; the sensitivity term is what stops a degenerate graph from reading as perfect parity.

MEASURED-EQUIVALENT (LibreYOLO's harness, INHERITED, and a good design):
`REL_TOL = 3e-4`, `MIN_SENSITIVITY_MARGIN = 100.0`, `MIN_REL_SENSITIVITY = 1e-6`.

**Reference selection is a per-family decision, not a default:**

- Compare against the **eager prepared graph** — the model *after* your export-time rewrites, in
  eager mode. Not the raw model, and not a decomposed `ExportedProgram` replayed as a module:
  INHERITED (LibreYOLO) that functionalization replays mutation-sensitive buffers differently
  and can be ~1.0 relative off *before* CoreAI is involved.
- **ONNX is not automatically a valid reference.** INHERITED (LibreYOLO): at a 640 canvas the
  RF-DETR ONNX artifact disagrees with the prepared graph by **9.3e-01**, because the two make
  opposite antialiasing choices. Which is "right" is a separate question — the point is that
  picking a reference is a decision you must make deliberately and record.

---

## What a receipt must contain

Every port ends with a receipt. Missing fields make the numbers unusable later.

- Hardware, macOS build, `coreai-core` / `coreai-torch` versions
- Checkpoint identity, **SHA-256 pinned**
- Canvas / static shape, dtype
- Parity: PSNR **and** rel error **and** sensitivity margin, **per variant**
- Latency per lane, median of 20 post-warmup, cold/warm stated
- Energy per inference per lane
- First-load E5RT cost
- Memory: resident **and** peak (they differ by orders of magnitude — see below)
- **Placement evidence**, not placement claims (which oracle, what it showed)
- **Alternatives measured and rejected, with their numbers**

---

## Memory is a different SHAPE, not just a different size

**MEASURED (SRVGG, 1080p→×4).** ANE activations stay on-die, so process footprint is the
host-side accumulation buffers only: **19 MB resident / 0.86 GB peak**, against the MLX
sibling's **21.24 GB** whole-frame.

Report both resident and peak. A single "memory" number hides the entire distinction, and the
distinction is one of the strongest arguments for CoreAI over MLX on constrained hosts.
→ `mlx-vs-coreai-fit.md`.
