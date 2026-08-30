# Wave 2 — ANE behaviour across convnet families

Breadth sweep, 2026-08-30. macOS 27.0 (26A5421a), M5 Max, `coreai-torch 0.4.1`, fp16 static
224², timm **pretrained** weights, parity vs fp32 torch.

**Headline: "convnets are good ANE citizens" is half true, and the half that fails is not the
half anyone would guess.** Every model was fully ANE-*eligible* (0 rejections) and 1.3–5.7×
faster than the GPU lane. The differentiator is **fp16 accuracy**, and it splits the families
cleanly.

## Results

| model | params | GPU dB | ANE dB | Δ | ANE ms | GPU ms | speedup |
|---|---|---|---|---|---|---|---|
| `resnet50` | 25.6M | 61.35 | **63.56** | **+2.2** | 1.001 | 1.915 | 1.9× |
| `mobilenetv4_conv_small` | 3.8M | 58.09 | **63.15** | **+5.1** | 0.366 | 2.103 | 5.7× |
| `resnet18` | 11.7M | 64.62 | **61.58** | −3.0 | 0.538 | 2.140 | 4.0× |
| `mobilenetv3_large_100` | 5.5M | 54.87 | 44.61 | −10.3 | 0.435 | 1.795 | 4.1× |
| `convnext_tiny` | 28.6M | 69.62 | 47.01 | −22.6 | 1.613 | 2.132 | 1.3× |
| `efficientnetv2_rw_s` | 23.9M | 70.28 | 46.10 | −24.2 | 0.927 | 3.171 | 3.4× |
| `efficientnet_b0` | 5.3M | 61.57 | 31.84 | **−29.7** | 0.524 | 2.391 | 4.6× |

**ANE eligibility was never the problem** — 0 rejections everywhere, max tensor rank 4. The
`≥50 dB` gate is what separates them, and four of seven fail it on the ANE while passing on the
GPU.

## `convnext_tiny` needed the `as_strided` fix — INHERITED note now MEASURED

Export failed outright:

```
ValueError: The exported program contains unsupported ATen ops: aten.as_strided.default.
Use register_torch_lowering() to ...
```

Replacing `AdaptiveAvgPool2d(1)` with an exact spatial mean (`x.mean(dim=(-2,-1), keepdim=True)`)
fixes it — **only** valid for output size 1, where the two are identical. This confirms
LibreYOLO's documented rewrite by direct experience; it moves from INHERITED to MEASURED.

## A plausible hypothesis, TESTED AND REJECTED

The losers (`efficientnet*`, `mobilenetv3`, `convnext`) all use hard-swish / SiLU / GELU, while
the winners (`resnet*`, `mobilenetv4_conv_small`) use ReLU. Upstream
[`coreai-torch#51`](https://github.com/apple/coreai-torch/issues/51) even names *"MobileNetV3
(2D MatMul + Hardswish)"* — and our `mobilenetv3_large_100` measurement independently reproduces
a deficit there. So: **is the ANE fp16 deficit an activation property?**

Controlled test — same `resnet18`, same pretrained weights, **only the activation swapped**
(17 sites), each measured against an fp32 reference of the *same* swapped model:

| activation | GPU dB | ANE dB | Δ |
|---|---|---|---|
| ReLU | 66.43 | 65.13 | −1.3 |
| **Hardswish** | 59.01 | **69.79** | **+10.8** |
| SiLU | 57.98 | 46.63 | −11.4 |
| GELU | 61.87 | 58.15 | −3.7 |

**Hypothesis rejected.** Hardswish is *better* on the ANE in this architecture, which directly
contradicts the reading of #51 as "hardswish is the problem". SiLU shows a real but modest
−11 dB; nothing here explains −22 to −30 dB. **The activation alone is not the cause.**

Worth stating plainly: had this been reasoned from the correlation instead of tested, it would
have gone into the skill as fact and been wrong.

## Second hypothesis — Squeeze-Excitation. ALSO TESTED, ALSO REJECTED.

The split maps onto SE blocks better than onto activations:

| has SE? | models | ANE verdict |
|---|---|---|
| **no** | `resnet18`, `resnet50`, `mobilenetv4_conv_small` | all **pass** (−3 to +5 dB) |
| **yes** | `efficientnet_b0`, `efficientnetv2_rw_s`, `mobilenetv3_large_100` | all **fail** (−10 to −30 dB) |
| no SE, but LayerNorm | `convnext_tiny` | fails (−22.6) — possibly a *different* cause |

An SE block is a global-pool → FC → sigmoid → **broadcast multiply** over the whole feature map —
exactly where a small fp16 error becomes a large one. Plausible. **Wrong.**

**ADD direction (the one that worked), pretrained weights:**

| model | SE? | GPU dB | ANE dB | Δ |
|---|---|---|---|---|
| `resnet50` | no | 56.96 | **59.06** | **+2.1** |
| `seresnext50_32x4d` | **yes** | 54.05 | **55.62** | **+1.6** |

**Adding SE costs essentially nothing on the ANE.** Hypothesis rejected.

**REMOVE direction — not viable, and worth recording as a method note.** Replacing
`efficientnet_b0`'s 16 `SqueezeExcite` blocks with `Identity` produced **NaN in the fp32
reference itself**, before CoreAI was involved. SE gating is what keeps activations bounded
through a trained network; remove it and the signal blows up over 16 blocks. **You cannot
ablate SE from a trained net and still have a numerically working net** — pick the ADD
direction, or ablate on a model retrained without it.

Note also `seresnet18` / `seresnet50` have **no pretrained weights in timm**, and after the
random-weight finding above those are unusable for a parity measurement. `seresnext50_32x4d`
was the SE-bearing model that both has weights and downloads.

## What remains — and why it is parked

Two plausible causes tested and killed by controlled experiment. What the failures still share,
and the passing models do not:

| model | depthwise | activation | SE | ANE verdict |
|---|---|---|---|---|
| `resnet18` / `resnet50` | no | ReLU | no | **pass** |
| `seresnext50_32x4d` | no (grouped) | ReLU | yes | **pass** |
| `mobilenetv4_conv_small` | **yes** | ReLU | no | **pass** |
| `mobilenetv3_large_100` | **yes** | hardswish | yes | fail −10 |
| `efficientnet_b0` | **yes** | SiLU | yes | fail −30 |
| `efficientnetv2_rw_s` | **yes** | SiLU | yes | fail −24 |
| `convnext_tiny` | **yes** (7×7) | GELU | no | fail −23 |

**Remaining candidate: depthwise convolution combined with a smooth activation.** Every failure
has both; every pass lacks at least one. Depthwise-with-ReLU passes
(`mobilenetv4_conv_small`, +5.1) and smooth-activation-without-depthwise is mild (the resnet18
SiLU swap, −11.4), so neither alone accounts for −23 to −30 dB — an interaction would.

**This is PARKED, not concluded.** Settling it needs to localize *where* the error accumulates
in the graph, and that requires reading intermediates out of the same compiled asset — which is
blocked on [`coreai-torch#76`](https://github.com/apple/coreai-torch/issues/76), since
re-exporting with probes changes ANE eligibility (→ `case-eomt-capture.md`). Three hypotheses
guessed from correlation, two already wrong: **stop guessing and get the instrument working.**

## Practical guidance while the cause is unknown

## What Wave 2 establishes for the fit journal

- **ANE eligibility is easy for plain convnets** — 0 rejections across all seven, rank ≤ 4.
- **ANE speed is real and large** — 1.3–5.7× faster than CoreAI's own GPU lane, biggest on the
  small models (`mobilenetv4_conv_small` 5.7×).
- **fp16 accuracy is the actual gate**, and it is *architecture-family* dependent in a way that
  is not yet explained. Plain BatchNorm+ReLU nets are safe; SE-bearing nets are not.
- Therefore: **run the fp16 parity check per model. Do not generalise from the family name** —
  two structural predictors have now been tested and both were wrong.

## Practical guidance while the cause is unknown

Empirical, from seven models plus two controlled A/Bs. Use it to *order work*, not to skip the
measurement:

- **Plain BatchNorm + ReLU convnets** (`resnet*`) — safe on the ANE, and 2–4× faster.
- **Depthwise + ReLU** (`mobilenetv4_conv_small`) — safe, and the biggest speedup measured (5.7×).
- **SE is not a risk factor** on its own (`seresnext50` passes).
- **Depthwise + smooth activation** (`efficientnet*`, `mobilenetv3`, `convnext`) — expect the
  ANE to lose 10–30 dB. Still ANE-*eligible* and fast; it is accuracy that fails. If the
  workload tolerates it, the speed is real; if not, the GPU lane on the same asset is fine.
