# Precision — fp32 proves the graph, fp16 ships it

---

## Export fp32 first. Always. It is cheap insurance.

**MEASURED.** An fp32 export isolates *"graph wrong"* from *"quantization loss"* in a single
number. On Moebius: fp32 = **138 dB** proved the graph was right; fp16 = **41 dB** localized the
problem to precision alone. Without the fp32 lane we would have gone looking for a graph bug
that did not exist.

Two exports, two numbers, one unambiguous verdict. Do not skip it to save time.

---

## Mixed precision does not lower

**MEASURED.** Pinning BatchNorm running stats to fp32 inside an fp16 graph — standard
MLX-converter hygiene, and the obvious first instinct — **fails legalization outright**:

```
unresolved materialization from tensor<*xf32> to tensor<...xf16>
```

**Uniform dtype per export.** Do not carry an MLX habit across; measure whether fp16 stats are
actually safe (count `running_var` values that round to zero) instead of assuming the fp32 pin
transfers.

---

## The fp16 BatchNorm fix that closed a 27 dB gap at zero cost

**MEASURED — Moebius: 41.4 dB → 68.3 dB**, landing at the same fp16 floor as the MLX sibling
(rel 9.3e-04).

**The failure:** don't evaluate `(x − mean) · rsqrt(var + ε)` in fp16 when `running_var` has
subnormal channels. The intermediate underflows.

**The fix:** replace each BN with a per-channel scale/shift computed at **fp64, then cast**:

```python
scale = gamma / sqrt(var + eps)      # computed in fp64
shift = beta - mean * scale          # computed in fp64
# then cast scale/shift to fp16 — the COMPOSITES are fp16-representable
```

The composites are representable even where `var` is not. That is the whole trick.

**This is invisible in fp32.** Any port that never exports fp16 will never find it — which is
exactly why an fp32-only support matrix can call a model "validated" and still be a long way
from shippable.

---

## The module-swap trap that silently deletes activations

**MEASURED, and it voided a result before it was caught.**
`isinstance(module, nn.BatchNorm2d)` **also matches timm's `BatchNormAct2d`**, whose `forward`
appends drop + activation. A swap that keeps only the affine transform **silently deletes the
ReLU** — rel error ~1.0.

**Rule: run a per-module differential harness over every instance you replace.** For each
swapped module, feed the same input to old and new and compare, before assembling the graph.
The export script's fp32 pre/post gate is what catches this at the only moment it is catchable.

This generalizes: *any* `isinstance`-driven graph rewrite can match a subclass that does more
than the base class. Check what you actually matched.

---

## Parity thresholds

Reference tiers from Apple's vendored skill, cross-checked against our own ports:

| Scenario | Expected PSNR | Investigate below |
|---|---|---|
| float32 end-to-end | > 70 dB | 60 dB |
| fp16 on-device | > 50 dB | 40 dB |
| 4-bit palettized | ~40 dB | 30 dB |

MEASURED against these: SRVGG fp16-on-ANE **58.15–69.36 dB** across three variants; Moebius fp32
**138 dB**, fp16 after the BN fix **68.3 dB**.

**Measure every variant separately.** MEASURED: sibling ports have had dtype verdicts *invert*
between variants of the same architecture. A verdict on `general` does not transfer to
`anime`.

---

## Dtype hygiene at the asset boundary

**MEASURED.** An fp32 asset **rejects fp16 inputs** — which is good, no silent cast. But it
means the caller must derive input dtype from the asset's name/metadata, **not from habit**.
Encode the dtype in the asset filename.

---

## OPEN

- **Compression below fp16 is entirely unexplored by us.** No palettization or quantization
  sweep has ever been run on a CoreAI asset on this volume. The ~40 dB 4-bit tier above is
  Apple's number, INHERITED, not ours.
- Whether the fp64-composite BN trick has an analogue for LayerNorm / GroupNorm at fp16.
  Untested.
