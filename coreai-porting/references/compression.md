# Compression — `coreai-opt`

**Closes the largest "never done" item in the skill.** As of 2026-08-29 we had run **zero**
compression on any CoreAI asset. `coreai-opt` is the official library; this file is what it
offers. Everything is **INHERITED** — installed and API-probed, **not yet run on a model.**

```bash
uv pip install coreai-opt        # 0.2.1 as of 2026-08-29
```

Docs: `apple.github.io/coreai-optimization` · Repo: `apple/coreai-optimization`

---

## API surface (probed, `coreai-opt==0.2.1`)

### Quantization — `coreai_opt.quantization`
`Quantizer`, `QuantizerConfig`, `ModuleQuantizerConfig`, `QuantizationSpec`, `ExecutionMode`,
plus `range_calculator` / `qparams_calculator` / `fake_quantize`.

**Presets:** `w4`, `w4_per_block`, `w8`

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig

config = QuantizerConfig.presets.w8()          # INT8 weight-only
quantizer = Quantizer(model, config)
prepared = quantizer.prepare(example_inputs)
finalized = quantizer.finalize()               # then torch.export -> TorchConverter
```

### Palettization — `coreai_opt.palettization`
`KMeansPalettizer`, `KMeansPalettizerConfig`, `ModuleKMeansPalettizerConfig`,
`PalettizationSpec`.

**Presets:** `w4`, `w6`, `w8` — codebook/LUT compression, k-means derived.

### Pruning — `coreai_opt.pruning`
`MagnitudePruner`, `MagnitudePrunerConfig`, `ModuleMagnitudePrunerConfig`, `PruningSpec`,
`supported_ops_registry`. No presets — configure explicitly.

### Casting — `coreai_opt.casting` ⚠️ read this before hand-rolling fp16
`cast_fp32_to_fp16`, `cast_int32_to_int16`, `cast_to_16_bit_precision`.

**This is a direct A/B against our own MEASURED work.** We derived a per-channel fp64-composite
BatchNorm rewrite to fix subnormal `running_var` in fp16 (41.4 → 68.3 dB, →`precision.md`).
`cast_to_16_bit_precision` is the official path. Three possibilities, all worth knowing:

1. It handles the subnormal-BN case → our rewrite is redundant, and that is a finding.
2. It does **not** → our rewrite is a genuine contribution, and a candidate upstream issue.
3. It does something different → measure both.

**Do not assume; A/B it on the Moebius UNet, where we have a known-good 68.3 dB target.**

### Inspection — `coreai_opt.inspection`
`ModelInspector`, `ModelSummary`, `ModuleInfo`, `OpInfo`, `SourceFrame`, `BoundaryEdge`,
`InputEdge`, `ModuleContext`.

**Another debugging facility we did not know existed** — `SourceFrame` in particular suggests
op→source-line mapping, which is what made `ane_validation_message` useful.
→ `debugging-methodology.md`, which already records that we built a bisection ladder around
tooling that turned out to ship.

---

## Compatibility warning — MEASURED at install

`coreai-opt` pulls in **`coremltools`** and **`torchao`**, and coremltools emits:

> Torch version 2.11.0 has not been tested with coremltools. You may run into unexpected errors.
> Torch 2.7.0 is the most recent version that has been tested.

Our lanes run torch **2.11.0** (coreai-torch 0.4.1) and **2.13.0** (0.4.2) — both well past
tested. **Treat compression results as suspect until parity-checked**, and record the torch
version in every compression receipt. This is exactly the kind of untested-combination seam
where silent wrongness lives.

---

## Parity targets

From Apple's vendored guidance (INHERITED), for orientation only:

| Scenario | Expected PSNR | Investigate below |
|---|---|---|
| fp16 on-device | > 50 dB | 40 dB |
| 4-bit palettized | ~40 dB | 30 dB |

We have **no measured CoreAI compression numbers of our own.** The 4-bit row is Apple's, not
ours. Every sweep must produce our own.

---

## How this enters the program

**G6 in the gate ladder**, and the first real sweep belongs on a small classifier where the
accuracy signal is cleanest and the export cycle is fast.

Sweep design, per the A/B rule (→ `measurement-protocol.md`):

- Axes: `{fp32, fp16} × {none, w8, w6, w4, w4_per_block} × {quantized, palettized}`
- **Every cell gets all three lanes**, not just the ANE — compression can change *which* unit is
  eligible, and that is arguably the more interesting result than the size saving.
- Record size, parity (PSNR + rel err + sensitivity margin), latency, energy.
- **Watch for placement changes.** A 4-bit asset that silently falls back to the GPU is a worse
  outcome than an fp16 asset that stays on the ANE, and nothing in the size number would tell
  you. → `placement-and-residency.md`.

---

## OPEN

- Does compression interact with ANE eligibility? Entirely unknown.
- Does `optimize()` compose safely with a compressed program? Given `coreai-torch#49`
  (silent miscompile), assume nothing → `known-upstream-defects.md`.
- Is there a macOS-vs-iOS compression preset split, as Apple's vendored skill hints? Unexercised.
- `ExecutionMode` in the quantization module — purpose unprobed.
