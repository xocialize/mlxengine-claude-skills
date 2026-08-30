# Case study — SwinIR: a `blocked` row retired, and a harder lesson

Ported 2026-08-30. LibreYOLO model source (`models/swinir/nn.py`; their export path stayed
sealed), official **SwinIR-S lightweight ×2** checkpoint (0.91M params, window 8, embed 60),
fp16 static 64² → 128², macOS 27.0 (26A5421a), M5 Max.

## 1. The `blocked` row does not reproduce

LibreYOLO record SwinIR as blocked: *"The export process DIES rather than hangs … the converter's
peak memory is the prime suspect on a 16 GB machine."*

**MEASURED: it exports cleanly. Peak RSS 0.70 GB across the whole run.**

That is now the second architecture in this class to export without memory trouble (SCUNet:
1.05 GB). Their row should be re-tested rather than inherited — and their own note already warns
that an earlier single-run conclusion there was contradicted by a second run.

**Their row can be retired on export grounds.** Whether SwinIR is *worth* a CoreAI port is a
different question, answered below.

## 2. The rank-5 fix generalises

SwinIR uses the classic spelling — `x.view(B, H//ws, ws, W//ws, ws, C)` then permute — rather
than SCUNet's `einops.rearrange`, but it is the same rank-6 shape. The same fold applies
(`recipes/swinir/rank5_swin.py`), bit-exact at **0.0e+00** in fp64 for partition, reverse and
round-trip.

| | baseline | rank-5 |
|---|---|---|
| graph nodes | 1495 | 1663 (**+11%** — the fold trades a few reshapes for legal ranks) |
| max tensor rank | **6** | **5** |
| ANE validation hits | **288** (`view_*`) | **0** |
| ANE latency | 97.5 ms | 88.7 ms (**only 1.1×**) |
| ANE parity | 65.86 dB | 65.86 dB |
| GPU | 4.39 ms / 71.41 dB | 4.50 ms / 71.41 dB — unchanged |

Note the fold **increases** node count here. It removes rank, not work.

## 3. The hard lesson — 0 hits, still not resident

SCUNet's rank-5 rewrite bought an **8.7×** ANE speedup and proven residency. SwinIR's bought
**1.1×**. The reason, from the oracle:

| lane | gpu_freq | power | rate | mJ/inference |
|---|---|---|---|---|
| GPU | 1620 MHz | 78.2 W | 230.7/s | **291** |
| "ANE" | **1524 MHz — GPU BUSY** | 31.5 W | 11.3/s | **1814** |

**Zero validation hits and it is still not resident.** The "ANE" lane is 20× slower than the
plain GPU lane and 6.2× worse on energy. → the correction now recorded in
`placement-and-residency.md`: **hits are a reliable negative, never a positive.**

## Verdict

**SwinIR belongs on the GPU — CoreAI-GPU or MLX.** It converts, it is accurate (71.41 dB GPU,
and even the partitioned ANE lane holds 65.86 dB), and the GPU lane is fast at 4.4 ms. There is
no ANE story here: the runtime declines the placement even with every stated objection removed.

## What this pair establishes about window attention

Two architectures, same fix, opposite outcomes on placement — and **neither ends up wanting the
ANE**:

| | SCUNet rank-5 | SwinIR rank-5 |
|---|---|---|
| rejections | 0 | 0 |
| **resident** | **yes** (338 MHz) | **no** (1524 MHz) |
| vs GPU lane | 3.6× slower | 20× slower |
| fp16 parity | 39.12 dB (fails gate) | 65.86 dB (passes) |

**Window attention is not an ANE workload.** The rank-5 fold is still worth having — it is the
only way to clear the eligibility barrier, and it is exact and reusable — but clearing that
barrier does not make the ANE the right lane. Route window-attention models to the GPU.
