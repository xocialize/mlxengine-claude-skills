---
name: forge-net-clean-capabilities
description: The Forge §N net-clean capability track — classical-DSP and Apple-framework image/video features built with NO model, NO weights, NO MLX and no licence exposure, as specified in mlxengine-todo/GAP-PROGRAM.md §N (N1 sharpening, N2 film grain, N3 hot/dead pixel, N4 white balance, N5 VTTemporalNoiseFilter, N6 SDR→HDR, N7 Vision text masks, N8 fidelity measurement, N9 deinterlace, N10 relight-lite). Use when building or reviewing any of those rows, deciding which package a net-clean capability belongs in (media-bridge / MaskKit / ForgeRestoreKit / ForgeLightKit), working on FidelityGuard, TextMaskDetector or FilmGrainSynthesizer, or wiring these into the Forge UI. Trigger phrasings — "net-clean capability", "GAP-PROGRAM §N", "Tier 0/Tier 1 Forge", "RestoreTier0", "downsample consistency", "per-octave SSIM", "AFGS1 grain", "Vision text mask", "MPSImageGuidedFilter", "SDR to HDR anchor", "which Kit does this go in". ENCODES A MEASURED FAILURE PATTERN — every §N row specified from a documentation read has had a load-bearing claim fail on first measurement. Read before treating any §N recipe as a spec.
---

# Forge net-clean capabilities (§N)

Ten capability rows that need **no model, no weights, no engine, no licence review** — classical DSP
or Apple frameworks. They close six competitor capability groups and several are *better* than the
competitor's version, not parity. Spec: `mlxengine-todo/GAP-PROGRAM.md` §N. Product-side obligations:
`mlxengine-todo/FORGE-UI-REQUIREMENTS.md`.

## 🚨 The one thing to read first

**Every §N row specified from a documentation read has had a load-bearing claim fail on first
measurement. Four for four, and N1's was total — the first build was a no-op.** Treat the recipes in GAP-PROGRAM §N as *hypotheses with citations*,
not as specs. Budget a measurement before building on any stated mechanism.

| Row | The plan said | Measured |
|---|---|---|
| **N7** | Vision's `minimumTextHeight` defaults to 1/32, silently ignoring small text; force it to ~0 | **Not a monotone filter at all.** h=1024/14 pt found at *every* value incl. 0.5; h=2048/30 pt found at 0/0.01, **lost at 0.03125/0.06, found again at 0.1+**. It steers an internal scale search and moves under trivial fixture changes. **Tiling is the real lever** — holding the parameter at 0, whole-image found 0/3 planted strings at h=3000/24 pt where tiled@1024 found 3/3. |
| **N8** | Per-octave SSIM separates legitimate sharpening from hallucination | True, but **not on any absolute threshold.** cs-SSIM on a Laplacian band is compressed against 1.0: sharpening `[0.996, 0.998, 0.999, 1.000]` vs 1-px geometry drift `[0.986, 0.981, 0.981, 0.987]` — they overlap on every cutoff. **Ratio of deficits** (`1 − band`) separates them ~5× (0.25 vs 1.36). |
| **N1** | *"The noise-awareness collapses to one line: ε = (k·σ)²… variation below the noise floor lands in base, never entering detail"* | **Wrong in direction AND magnitude.** `a = var/(var+ε)` *preserves* high variance and *smooths* low, and `detail = I − base`, so sub-ε variation is exactly what lands in detail. And on a clean source σ→0 ⇒ ε→0 ⇒ `base ≈ I` ⇒ `detail ≈ 0` — a **no-op on the clean images the feature is for** (measured: edge slope 0.04954 → 0.049555). ε is a *detail-scale* control (halo prevention); noise-awareness is the coring + the Polesel gate, keyed to `(k·σ)²` **not** ε. Two knobs, not one. |
| **N2** | *Size* maps to "`ar_coeff_lag` + coefficient magnitude" | **Magnitude dominates; lag alone makes grain FINER.** A fixed gain budget spread over more taps dropped the nearest-neighbour weight 24/128 → 4/128 and correlation fell 0.33 → 0.17 → 0.096. For an AR (IIR) process, **correlation length is governed by total feedback gain, not kernel width** — at fixed gain, distance-3 correlation was *lower* at lag 3 (0.23) than lag 1 (0.28), despite lag 1 having no tap reaching 3 px. |

The method that caught all four is the program's own (`GAP-PROGRAM.md` §V method note): **run a probe
rather than read a header, and include a control that proves the code path is live.**

## Where a row lives — decide before writing code

Category **A** throughout: no MLX, no engine, no vendored binaries, **macOS-14 floor, no Metal
requirement**. Apple OS frameworks (Vision, CoreImage, Accelerate, MPS) are Tier 0/1 and are *not*
dependencies. Rule of thumb — what is the heaviest thing it must import?

🚨 **"No Metal *requirement*" is NOT "no Metal", and reading it that way costs real performance.** The
rule is that the package must **build and run without a GPU** — so a host wanting the planner is never
forced into a GPU toolchain or a 26.x floor. Shipping a Metal backend is fine and expected. The
in-tree precedents are `media-bridge`'s `SSIMULACRA2Metal` and `RestoreTier0`'s `GuidedFilterMetal`,
both category A. Three properties keep the guarantee intact:

1. **`init?()` returns nil with no device** — every caller has a CPU path.
2. **Compile the shader at runtime** from a source string via `makeLibrary(source:)`. No `.metal` file,
   no metallib to package, no Metal build-system dependency — and it sidesteps the metallib trap this
   codebase has already paid for once.
3. **The CPU implementation stays the oracle**, and parity is tested, not assumed.

Add a `diagnostics()` that distinguishes *no device* from *shader compile failed* from *missing
kernel* — three problems with three different fixes.

🔑 **The app-facing currency is `CVPixelBuffer` backed by `IOSurface`, never `[Float]` or `CGImage`**
(`CLAUDE.md`: *"keep pixels GPU-resident… not CGImage, which forces a CPU copy"*). The apps drive
`MTKView` from `MTLTexture`s. `[Float]` is the reference/oracle layer, **not** an app surface.

🚨 **Porting a chain to the GPU is all-or-nothing.** One accelerated stage among CPU stages gives
`texture → plane → GPU → plane → texture`, which is **slower than never touching the GPU** — two
transfers per stage, and the transfer dominates a per-pixel kernel. Partial adoption is a
pessimization, not a partial win. Reference: `RestoreTier0`'s `MetalSharpenPipeline` (v0.4.0) moves
ingest, the noise reduction, three filter bands, coring, gate, Sobel, clamp and write-back, touching
the buffer exactly twice. Its `PixelBufferBridge` carries two silent traps:

- **Never map a buffer as an `_srgb` format.** Metal linearizes on read; σ-derived constants are
  calibrated for **gamma-encoded** luma. No crash — just shadow detail eaten and highlight noise kept.
- **The `CVMetalTexture` wrapper must outlive the `MTLTexture` it vends.** Dropping it early is a
  use-after-free that usually appears to work.

🚨 **Do NOT duplicate a shared type across two Kits — the collision is worse than it looks.**
`PixelBufferBridge` was copied from `RestoreTier0` into `LightTier0` on the reasoning that "two copies
is tolerable, three is a package." **That reasoning was wrong.** The real trigger is *any consumer
importing both Kits*: an app that does gets `ambiguous use of 'PixelBufferBridge'` and must
module-qualify every reference. Since the whole point of these Kits is to be composed in one app, the
first consumer hits it. **Extract a shared Tier-0 package on the second copy, not the third.**

Also: **biplanar 4:2:0 is the easier input**, not the harder one — luma is its own plane, so "chroma
untouched" is structural. And a reduction (e.g. a noise estimate) forces a sync point: let callers
supply the value when they already know it.

| Row | Package | Why |
|---|---|---|
| N8 fidelity measurement | **`media-bridge` / `MediaMeasure`** ✅ v0.10.0 | Architecture §2.1 says explicitly the N8 guards belong inside `MediaMeasure`, next to SSIMULACRA2 — *not* a new "measure kit" |
| N7 text masks | **`MaskKit`** ✅ v0.2.0 | Every way Forge produces a selection, in one place, so Scale/Restore/Erase never depend on each other |
| N1 sharpen · N2 grain · N3 hot pixel | **`ForgeRestoreKit` / `RestoreTier0`** ◐ v0.3.0 (N1 + N2) | §4.5 reserves the target for exactly these three |
| N4 WB · N6 SDR→HDR · N10 relight-lite | **`ForgeLightKit`** ○ | §4.6. Deliberately not created yet — D1 says don't create repos speculatively |
| N5 video denoise · N9 deinterlace | `ForgeMotionKit` / `RestoreVideo` ○ | §4.7 |

⚠️ **`ForgeRestoreKit` ships `RestoreTier0` ONLY, on purpose.** The umbrella, `RestoreTier2` and the
`FrameRestoreProvider` seam are BRIDGE-070, gated on **D1** (repo granularity) and **D2** (the job
vocabulary `denoise`/`motionDeblur`/`defocusDeblur`, which becomes public API on first tag). **Do not
add anything that names a job** — that is what keeps both decisions open.

## Recurring traps

- 🔑 **Anything seeded must be addressable by ABSOLUTE position, not by tile.** Grain, dither, noise:
  a viewport tile must be bit-identical to the same region of a full-frame render, or the effect
  crawls under the picture as the user pans. This is the concrete reason **`CIRandomGenerator` is
  unusable anywhere in Forge** — no seed, no properties, so the property is not merely untested, it is
  unachievable. Test with a **non-block-aligned** origin; alignment hides off-by-ones.
- **Overlap/seam blending: the partner comes from the neighbour's own continuation, never from the
  output buffer.** Adjacent N-px blocks do not overlap in output space, so blending against the buffer
  blends against *zero* — which attenuates a stripe rather than joining two fields, i.e. a seam that
  is quieter instead of absent. (This is what AFGS1's "read 34 samples for a 32-px block" is *for*.)
  Reading from the source also makes the result independent of write order, which is what makes tiled
  and full-frame renders match.
- **Never process full-res for display.** ~30–80 ms at 50 MP on M3/M4 Max but **150–250 ms on a base
  M1/M2**, and memory binds before ALU (one 50 MP RGBA16F buffer ≈ 400 MB). Proxy for fit-to-window,
  viewport tile at 1:1, full-res only on tiled export.
- **Work on gamma-encoded luma, not linear.** Sensor noise is signal-dependent (σ² = a·Y + b), so in
  linear light one threshold is far too aggressive in shadows and too timid in highlights; gamma is
  approximately variance-stabilizing, so one global threshold works.
- **Data you cannot derive must be injected, never transcribed.** AV1's 2048-entry `gaussian_sequence`
  is the live example: a table written from memory would *look* correct and be silently wrong. Make it
  a parameter, ship a documented substitute, isolate every reconstructed constant in one `SPEC`-marked
  file so "implemented structure" vs "normative numbers" is a file boundary.
- **`MPSImageGuidedFilter` is NOT deprecated** (V7, measured — zero deprecation attributes, MPSImage
  headers byte-identical between the 26.5 and 27.0 SDKs). The real risk is **Metal 4 interop**: only
  `MPSNDArray` gained `encodeWithMTL4CommandEncoder`, so a Metal 4 pipeline needs a separate classic
  queue + `MTLEvent`. Isolate behind an internal `EdgeAwareFilter` protocol — free today, one-file swap
  later. Fallback is a hand-rolled Metal kernel (~200 lines), **not** MPSGraph, which has no
  guided-filter op.
- ⚠️ **Do not use `MPSImageIntegral` for local variance at 50 MP** — the accumulator reaches ~5×10⁷
  against fp32's 24-bit mantissa (~1.7×10⁷) and catastrophic cancellation produces visible banding.
  Two `MPSImageBox` passes instead.
- **MPS kernels are not `CIFilter`s** — wrap in `CIImageProcessorKernel` and implement
  `roi(forInput:arguments:outputRect:)` declaring the halo, or you ship tile seams.

## Test discipline that actually caught the bugs

1. **Assert structure, not golden values**, wherever a constant is reconstructed or a substitute is
   injected — a golden test would pin the substitute rather than the spec, and would have to be
   rewritten the day the real data lands.
2. **Derive the invariant independently of the code that produces it.** AR tap counts are asserted
   against the closed form `2·lag·(lag+1)`; the template's sampled window is re-derived from the offset
   arithmetic. Agreement is then a real cross-check, not a tautology.
3. **Measure the thing the feature is *for*.** Correlation for grain size, seam-vs-interior step ratio
   for overlap, tile-vs-full-frame equality for pan stability. Without these the stage can be wired to
   nothing, or backwards, and every other test still passes.
4. **Report environment-dependent numbers; assert only what is stable.** Vision's small-text behaviour
   moves under trivial fixture changes — pinning it would encode an accident as a contract. Assert
   "tiling never finds less" and "our configuration finds everything"; put the decisive numbers in a
   comment with the fixture.
5. **Pin the honest limits as tests too.** Downsample consistency is blind to invention balanced at the
   reduction scale (a Nyquist-aligned checkerboard reads bit-exact) — that has a test, so nobody later
   assumes the guard is sufficient rather than necessary.
6. **When a test fails, ask whether the test is wrong before relaxing it.** N2's first correlation test
   asserted the wrong physics; the failure was the finding.

## Build / test

```bash
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift build
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift test
```

Category-A packages test fine headless (no metallib boundary). **Benchmark in Release** — pure-Swift
numeric code is 50–150× faster with `-O`/WMO, and `swift build`/`swift test` default to Debug.

## Status (2026-07-29)

✅ **N8** zero-weight half (`media-bridge` v0.10.0) · ✅ **N7** (`MaskKit` v0.2.0) ·
🟡 **N2** engine (`ForgeRestoreKit` v0.1.0) — *residual-fit estimator, the differentiated half, still
open* · ✅ **N1** (`ForgeRestoreKit` v0.3.0, CPU + Metal guided filter) · ○ **N3, N4, N5, N6, N9, N10**.

⚠️ **N1's row was wrong in a way worth remembering**: it claimed the noise-awareness *"collapses to one
line, ε = (k·σ)²"*. It fails in both direction and magnitude, and the first build was a **complete
no-op**. ε is a *detail-scale* control (halo prevention); noise-awareness lives in the coring and the
Polesel gate, which is keyed to `(k·σ)²` and **not** to ε.

**Feed corrections back into `GAP-PROGRAM.md` §N in place** (struck through, with the measurement) —
that document is the spec the next row is built from, and leaving a disproved mechanism in it is how
the next row inherits the same wrong prescription.
