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

## Leading hypothesis — Squeeze-Excitation. UNTESTED.

The split maps onto SE blocks better than onto activations:

| has SE? | models | ANE verdict |
|---|---|---|
| **no** | `resnet18`, `resnet50`, `mobilenetv4_conv_small` | all **pass** (−3 to +5 dB) |
| **yes** | `efficientnet_b0`, `efficientnetv2_rw_s`, `mobilenetv3_large_100` | all **fail** (−10 to −30 dB) |
| no SE, but LayerNorm | `convnext_tiny` | fails (−22.6) — possibly a *different* cause |

An SE block is a global-pool → FC → sigmoid → **broadcast multiply** over the whole feature map.
A per-channel scalar multiplying every activation is exactly where a small fp16 error becomes a
large one.

**This is a hypothesis, not a finding.** Test it the same way the activation one was tested:
take a model with SE, ablate the SE blocks to identity, and re-measure. Do not write it down as
a cause until that runs.

## What Wave 2 establishes for the fit journal

- **ANE eligibility is easy for plain convnets** — 0 rejections across all seven, rank ≤ 4.
- **ANE speed is real and large** — 1.3–5.7× faster than CoreAI's own GPU lane, biggest on the
  small models (`mobilenetv4_conv_small` 5.7×).
- **fp16 accuracy is the actual gate**, and it is *architecture-family* dependent in a way that
  is not yet explained. Plain BatchNorm+ReLU nets are safe; SE-bearing nets are not.
- Therefore: **run the fp16 parity check per model. Do not generalise from the family name.**
