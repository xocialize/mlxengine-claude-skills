# Parity Testing — PyTorch vs MLX

Parity tests are the single most important tool when a port produces wrong output. They turn "the whole pipeline is broken" into "layer 17 diverges at max_abs = 0.3, focus there".

**Past ports have skipped this step and paid for it.** Every `-mlx` fork should ship with a parity test harness, even if it lives behind an optional dependency.

Note on code style: `.eval( )` (torch's eval-mode method) is written with a space between parentheses to sidestep a security-hook false-positive on the Python builtin. In real code write it as `pt_model.eval()`.

## Test structure

Recommend: `tests/parity/` directory with one file per major module (`test_attention_parity.py`, `test_unet_block_parity.py`, `test_vae_parity.py`, `test_full_pipeline_parity.py`).

## Canonical template

```python
# tests/parity/test_attention_parity.py
import math
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from my_model_mlx.model.attention import Attention as MxAttention
from my_model_pytorch.model.attention import Attention as PtAttention


def _load_pt_to_mx(pt_model, mx_model):
    """Copy PyTorch state dict into an MLX model with key/shape transforms."""
    import mlx_forge.transpose as tp
    pt_state = pt_model.state_dict()
    mx_weights = {}
    for key, val in pt_state.items():
        new_key = key  # rename if needed
        w = mx.array(val.detach().cpu().numpy())
        if "conv" in key and ".weight" in key:
            w = tp.transpose_conv(key, w, kind="conv2d")
        mx_weights[new_key] = w
    mx_model.update(tree_unflatten(list(mx_weights.items())))
    return mx_model


@pytest.fixture
def seeded_input():
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, 128, 1024)).astype(np.float32)


def test_attention_parity_fp32(seeded_input):
    B, N, C = seeded_input.shape
    config = dict(hidden_dim=C, num_heads=8, head_dim=128)

    pt_model = PtAttention(**config).eval( )
    mx_model = MxAttention(**config)
    _load_pt_to_mx(pt_model, mx_model)

    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(seeded_input)).numpy()
    mx_out = np.array(mx_model(mx.array(seeded_input)))

    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    mean_abs = float(np.mean(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-4, f"attention diverges: max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e}"
```

Use fp32 for parity tests — it isolates numerical drift from framework precision differences. Save fp16 / bf16 tests for regression against a known-good fp32 reference.

## Threshold table

For fp32 parity testing:

| Scope | `max_abs` target | Acceptable |
|---|---|---|
| Single linear / conv / norm layer | `< 1e-6` | `< 1e-5` |
| Self-attention / cross-attention block | `< 1e-5` | `< 1e-4` |
| Transformer block (attn + FFN + norms + residual) | `< 1e-4` | `< 1e-3` |
| Full DiT / UNet pass | `< 1e-3` | `< 1e-2` |
| Full VAE encode or decode | `< 1e-3` | `< 1e-2` |
| End-to-end pipeline (denoising + VAE) | — | qualitative (PSNR > 35 dB on fixed input) |

For fp16 inference:

| Scope | `max_abs` target |
|---|---|
| Single layer | `< 1e-3` |
| Transformer block | `< 5e-3` |
| Full UNet / DiT pass | `< 1e-2` |
| End-to-end | qualitative (PSNR > 30 dB) |

**Anything at `max_abs > 1e-1` is a port bug, not numerical drift** — *on CPU / matched precision* (see the CPU-stream note below; on the GPU stream accumulated tf32 can exceed this without a bug). Stop and isolate.

**Run correctness parity on the CPU stream — MLX GPU fp32 is tf32-like.** Apple-GPU fp32 *matmul* uses reduced-precision accumulation (~3e-3 abs error per matmul; not bf16, not true fp32). It compounds with depth and reads exactly like a port bug: it produced rel ~6e-2 over 26 layers (Zonos backbone) and ~0.31 over 32 layers (the MLX-Swift Whisper encoder) — both **correct** ports, confirmed because **MLX-CPU matmul is bitwise-identical to PyTorch fp32** (rel 0.0). So gate "is the math right?" on CPU — Python `mx.set_default_device(mx.cpu)`, Swift `Device.withDefaultDevice(Device(.cpu)) { ... }` — and treat the GPU/bf16 path as a separate, looser end-to-end concern. (Bonus for MLX-Swift: the CPU stream needs no `default.metallib`, sidestepping the SPM-CLI metallib error.)

**Diagnostic — "every sub-component is bit-exact in isolation, but the composition diverges and the error grows downstream" ⇒ it's reduction-order numerics, not a logic bug. Move to CPU.** This is the tell that saves hours. While debugging the streaming VAE decode, a chunked path diverged from whole-sequence by ~4e-3 on GPU; bisecting showed the error *seeded at one layer and amplified through the rest*, yet each piece (`CausalConv3d` via the cache, the norm, the shortcut) was bit-exact chunked-vs-whole **in isolation**. That signature — components exact alone, composition off, error compounding — is precisely how tf32-like GPU accumulation reads when one path does *more, smaller* ops (the chunked decode) than the other (whole-seq) and so accumulates in a different order. On `mx.cpu` the same test was `max|Δ| == 0.0` (bit-identical), confirming the port was correct. Don't chase it as a wiring bug on GPU; reproduce on CPU first.

**Use a *relative* gate when activations are large.** The absolute `max_abs` targets above assume O(1) activations. Some layers don't cooperate: GPT-OSS hidden states (used as DiT conditioning in the Lens port) reach absmax ~1e4 in the deepest captured layers (they grow with depth: 234 → 920 → 2496 → 10688 across the selected layers). An absolute `1e-3` gate is impossible there. For those, gate on **cosine similarity** or **relative error** `max|Δ| / max|ref|` instead, and capture the intermediate magnitudes during the golden dump so you pick the right regime. (Cosine has a failure mode of its own: a few giant outlier dims can hold cosine high while smaller components are wrong — pair it with relative-error, and isolate per-layer if a deep-layer cosine looks suspiciously good. See common-pitfalls #10.)

## When the CPU stream is NOT available — a custom Metal kernel forces a hybrid, phase-split gate

The CPU-stream doctrine above has a hard exception: **a port containing a custom Metal kernel cannot
run its full forward on CPU at all.** `MLXFast.metalKernel` dispatches are GPU-only and fail loudly —
`[metal_kernel] Only supports the GPU` (`mlx-c/mlx/c/fast.cpp`). Ports with hand-written kernels for
ops MLX lacks hit this: deformable convolution (DCNv2 in BiRefNet's `ASPPDeformable`), custom
rasterizers, bespoke gathers.

Do **not** respond by giving up and gating everything on GPU with a loose threshold. Split the gate by
phase and pin each phase to the strictest stream it can actually run on:

- **the kernel-free majority** (transformer/Swin backbone, resizes, merges) → CPU stream, tight
  *relative* bound. In `mlx-birefnet-swift` this covered 357 of 687 tensors at relMax ≤ 8.3e-5,
  cosine 1.0000000 — a genuinely strong statement.
- **the phases that touch the custom kernel** (the decoder here) → GPU stream, judged on **cosine**.

**On the GPU lane, judge cosine — never absolute maxAbs.** Measured on identical tensors in the same
port, the same encoder output was `maxAbs 3.96e-3` on CPU and **`7.25` on GPU** — a ~1800×
amplification of single-element outliers — while cosine held at 0.9999 on both. An absolute bound
there produces either false failures or, if loosened to fit, a gate that can't fail. Report both
metrics; gate on the honest one per lane.

Two practical consequences:

- **Label the stream in the gate output.** A parity line without its stream is unreadable six months
  later — the same number means "bug" on CPU and "fine" on GPU.
- **Don't pin the weight LOAD to CPU and the forward to GPU by accident.** Loading on CPU is fine
  (`mlx-porting` even recommends it for quantized setups) but leaving a half-pinned graph makes the
  first kernel dispatch fail confusingly. Let each phase scope its own pin.

**Normalize tolerances to the reference's own dynamic range**, not to a single absolute number, whenever
one gate spans tensors of different scale. Activations spanning ±32 and a matte spanning [0,1] cannot
share an absolute bound — `maxAbs / (refMax − refMin)` makes the rows comparable. (Same miscalibration
the Mage-Flow port recorded as "absolute 0.9999 was miscalibrated for this arch".)

## Isolation strategy when a test fails

The goal is to find the innermost layer where divergence first appears.

1. **Start at the block level.** If `test_transformer_block_parity` fails at `max_abs = 0.3`, go inward.
2. **Bisect within the block.** Test norm alone. Test attention alone. Test FFN alone. Test residual alone. One of them will fail.
3. **Within the failing sub-module, instrument.** Print intermediate tensor stats (`min, max, mean, std, sum_abs`) on both sides at each op. The first op where stats diverge is the bug.
4. **Common root causes** (in order of frequency from past ports):
   - Wrong QKV reshape (interleaved vs per-tensor).
   - Wrong `num_heads` vs `head_dim` interpretation.
   - Wrong RoPE layout (`traditional` flag).
   - Missing `qk_norm`.
   - Wrong weight transpose (Conv layout).
   - Default constructor arg diverging from config.
   - Missing materialization at conversion time (tensor is literally zero).
   - Wrong activation (silu vs gelu, GEGLU variant).

## Cheap, high-leverage anchors (do these before expensive parity)

Two techniques from `bernini-r-mlx` that catch whole classes of bugs for almost no cost:

**Offline key-map proof — validate the conversion recipe with zero weight download.** A wrong
weight-key mapping is the most common silent conversion bug, and you can prove the map is complete
*before* downloading 100 GB. Load the candidate's safetensors **index** (key → shape, from the HTTP
header range-read), build `{key: mx.zeros(shape)}` (lazy — never materialized, so it's free), run it
through your key premap + sanitizer, and assert the resulting key set **exactly equals** the MLX
model's parameter tree:
```python
got = set(sanitize(premap({k: mx.zeros(tuple(s)) for k, s in index_shapes.items()})))
expected = {k for k, _ in tree_flatten(model.parameters())}
assert not (got - expected)                 # no stray keys
assert (expected - got) <= COMPUTED_BUFFERS  # only internally-computed buffers unprovided
```
If this passes, converting the real weights is a pure dtype-cast + transpose op — no key surprises.

**Reduction-to-known-case anchor.** When you add a *generalized* forward (multi-segment, multi-ref,
batched, extra conditioning), the first test is that it **reduces bit-exact to the already-validated
simple forward** on a trivial input. Bernini's multi-segment forward, with a single segment at the
default `source_id`, must equal the stock t2v forward — it did, `max_abs == 0.0`. That one assertion
locks the whole sequence-assembly / rope / block-loop / slice plumbing before any guidance is layered
on, so later bugs are isolated to the new logic, not the plumbing. (Pairs with pitfall #13: a
parameter-free flag whose default is identity gives you this anchor for free.)

## Causal-VAE roundtrip: measure with full temporal context, not a single frame

A 3D causal VAE (Wan, LTX, Cosmos) has a temporal warm-up: early output frames carry the
`CausalConv3d` zero-padding artifact and reconstruct poorly **in isolation**. A single-frame
encode→decode roundtrip on a real photo measured MAD ~16/255 (looks broken) when the same VAE with a
full temporal receptive field reconstructs at ~2/255. Feed a short clip of repeated frames and score
the **last** frame (the one with full context), matching how the generation pipeline uses the VAE.
Don't gate the VAE on a single still; verify `VAE_MEAN`/`VAE_STD` constants == the checkpoint's
`latents_mean`/`latents_std` first to rule out a real normalization bug.

## Use realistic input for deep / chaotic nets — random misleads (the `nafnet-mlx` lesson)

Uniform-random parity input is fine for a few layers, but a deep restoration / generative
net is near-chaotic on out-of-distribution noise: tiny per-op conv differences amplify
through the stack. NAFNet-GoPro (a 28-block stage) diverged **>1.0 even on mx.cpu fp32**
for `np.random.rand` input, while the **same weights on a real 256-crop matched at ~9e-7**
(GPU and CPU alike). The shallower SIDD variant passed on random (~5e-4), which masks the
issue — don't generalize from it. Rule: feed an in-distribution sample (a real image / a
plausible latent) for full-model parity; reserve random inputs for shallow single-layer tests.
A failing full-model parity on random input is often the test's fault, not the port's —
re-run with a real sample before debugging the model.

## Golden-input fixtures

For reproducibility, check a small set of seeded inputs into the repo:

```
tests/parity/fixtures/
├── attention_input_2x128x1024_seed42.npy
├── image_3x512x512_cat.png
├── prompt_golden.txt
└── latents_seed42.npy
```

Same seed on both sides, same input bytes. Loaded identically via numpy. This avoids the `mx.random` vs `torch.manual_seed` incompatibility.

## Content-independent constants — regenerate + validate, don't bake as goldens (the `edgetam` video lesson)

When the reference reads what looks like opaque captured state — sinusoidal position encodings, RoPE cos/sin tables, temporal-slot embeddings, any tensor that is a pure function of shape/index — it is **content-independent**: generate it from the spec in the port, don't dump it as a golden and feed it back in. Goldening a content-independent constant adds a fixture surface AND hides whether you can actually reproduce it (the Swift/MLX port can't ship a `.npy` of it).

But **validate the regeneration against ONE captured copy** — the normalization is the parity trap. EdgeTAM's whole P2 video forward *looked* blocked on three "captured" pos tensors (current-frame pos, perceiver-input pos, memory pos); they were `PositionEmbeddingSine` over a fixed grid + a learned temporal embedding indexed `[num_maskmem − t_pos − 1]`. Regenerated in-port and validated to ~5e-7 vs the captures — which is exactly where the `normalize`/`temperature`/`eps`/even-odd-pair quirks and the `6−t_pos` index get pinned. Generate-without-validate = a silent off-by-normalization; golden-without-generate = you never learn you can't reproduce it.

This is what lets a hard **stateful** port transcribe cleanly: granular per-op goldens (each op bit-exact in isolation) + regenerated-and-validated constants + the state-machine GLUE checked by reconstructing one intermediate (EdgeTAM: the assembled memory-bank `memory_pos` vs the captured tensor, so the slot-ordering / temporal-index / token-split logic is verified without a Python twin) + a tolerant e2e. The entire video memory stack (3 novel ops + RoPE-2D + a ~900-LOC memory-bank state machine + propagate loop) hit bit-parity on every op the first build; the only e2e bug was a coordinate transform (common-pitfalls #30).

## Pipeline-level parity (hardest)

Full pipeline parity is noisy because of samplers, schedulers, and free-running RNG. To compare:

1. Run PyTorch pipeline with a fixed seed; save:
   - Initial noise tensor (this doubles as the injected start — MLX RNG is not torch-seed-compatible, so you can never regenerate it; capture it).
   - Text embeddings (per-layer, if the model uses multi-layer features).
   - DiT input/output at the first denoise step — capture **from inside the loop with forward hooks** rather than refactoring the reference: `module.register_forward_pre_hook(fn, with_kwargs=True)` + `register_forward_hook(..., with_kwargs=True)`, snapshot the first call only. This gives production-magnitude IO (not random-weight), which catches bugs small-scale tests miss.
   - Final latents before VAE decode, and final pixel output.
2. Load those same tensors in the MLX test. Inject them at each stage to bypass RNG. (For a diffusers pipeline with custom component classes, the reference loads fine for golden capture once its package is on `sys.path` — no `trust_remote_code` needed locally.)
3. Compare at each stage: noise → embeddings → latents → pixels.
4. If embeddings match but latents don't, the denoising loop is wrong (scheduler, DiT, or CFG).
5. If latents match but pixels don't, the VAE is wrong.

## Gate at the largest production grid, on decoded output — small-grid latent cosines validate nothing above them (the `anima` / `qwen3vl` lesson)

Grid-size-dependent failures are a recurring class: every gate passes at the convenient small
size, and the port collapses past it, because the mechanism scales with the **position grid**
(RoPE extrapolation/scaling, pos-embed interpolation, window/grid indexing), not with values.
Two independent ports hit it in one week (2026-06/07):

- **Anima (Cosmos-Predict2 T2I):** the parity suite ran at 256²-class latents and gated
  cosine-on-latents — bit-exact. At the model's own base resolution 512² the output was garbage
  tiles, and not even a port bug: the torch oracle collapsed identically (placeholder
  deterministic-Euler sampler + RoPE past the validated grid; fix = `er_sde` + `rope_scale
  (1,4,4)`). The suite never decoded one image at 512², so a "fully passing" port shipped a
  broken product configuration.
- **qwen3vl-mlx-swift (Boogu's edit conditioner):** conditioning goldens lived at the 576-token
  vision grid (768² input) — cos 0.998 fp32. At the ~1024-token grid (1024² input → 64×64
  patches) cosine sagged to **0.84** and was shelved as "slightly oversaturated, not
  catastrophic." Downstream it WAS catastrophic — structured horizontal glitch-banding on every
  edit with a ≈1 MP input — and it surfaced months later in in-app validation, not in any gate.

Rules:

1. **The gate matrix must include the largest grid the product will run** — max resolution, max
   vision-token count, max sequence — not just the size that keeps the fixture small. Position
   machinery only exercises its failure modes (interpolate vs identity, extrapolate vs table)
   at the big grid.
2. **Every shipping resolution tier gets a decoded-output eyeball gate** (image/video/audio),
   not only latent cosines. Both failures above are invisible in "cosine looks fine-ish"
   latents and unmissable in one decoded sample.
3. **A cosine that degrades monotonically with grid size is a structural bug, not noise.**
   0.998 at grid A → 0.84 at grid B means a different code path is running at B. Treat any
   monotone-with-size sag as a blocker until root-caused — never file it as "minor residual."
4. **Conditioned generators amplify "mild" conditioning error into full-image artifacts.**
   There is no not-catastrophic conditioning divergence at the pipeline level until a decoded
   image at that grid proves it.

Related: pitfall #7 (production-scale spatial smoke), the `nafnet` realistic-input lesson above,
and common-pitfalls #33 (the Anima sampler/RoPE fix itself).

## Making PyTorch an optional dependency

Users installing the `-mlx` fork should not need torch. Add torch as a dev / parity-only extra:

```toml
# pyproject.toml
[project.optional-dependencies]
# accelerate enables low_cpu_mem_usage — matters when the PT reference is large
# (e.g. a 20B text encoder): without it the loader forces low_cpu_mem_usage=False.
parity = ["torch>=2.3", "transformers>=4.55", "diffusers>=0.30", "accelerate>=0.30"]
dev = ["pytest", "my-model-mlx[parity]"]
```

Then in tests:

```python
torch = pytest.importorskip("torch")  # skip if not installed
```

**Install caveat:** on a `uv`-created venv there is no `pip` / `python -m pip` — the skill's `pip install -e ".[parity]"` fails with "No module named pip". Use `uv pip install -e ".[parity]"`. (Detect the venv manager, or show both.)

**MXFP4 reference encoders on Apple/CPU.** If the PT reference uses a `gpt-oss` (MXFP4) text encoder, its kernels need Hopper+ GPUs and won't run on Apple/CPU. Load it dequantized to bf16 for the golden dump: `transformers.Mxfp4Config(dequantize=True)`. The MLX side (mlx-lm) runs MXFP4 natively, so the comparison is bf16(PT-dequant) vs MXFP4(MLX) — but verify forward correctness in matched precision first (convert the encoder to dense bf16 via `mlx_lm.convert(dequantize=True)` and gate bf16-vs-bf16), because matched precision is what reveals real bugs vs the quant gap. This is exactly how pitfall #10 (the YaRN rope bug) was caught — the MXFP4-vs-bf16 0.95 cosine looked like quant noise; the bf16-vs-bf16 0.94 proved it was a forward bug.

This keeps the main install lean while preserving parity tests for contributors.

## Invariant tests vs parity tests

`mlx-arsenal` tests use **invariant** patterns (e.g. `get_timestep_embedding(t, flip_sin_to_cos=True)` should equal `concat([second_half, first_half])` of `flip_sin_to_cos=False`). These catch self-consistency bugs.

Invariant tests are **complementary**, not a substitute. They cannot catch:
- Wrong default values vs reference config.
- Wrong QKV reshape pattern.
- Wrong weight transpose.

For porting, always include PyTorch-vs-MLX comparison, at least during initial development. Invariant tests can stay as the permanent regression suite once parity is locked.

## When parity is "close enough"

- fp32 parity at 1e-5 or better: ship it.
- fp16 / bf16 parity at 1e-3: ship it, document the fp32 reference is the oracle.
- 1e-2 to 1e-1: investigate. Often a small systematic bias from a missing `qk_norm` or off-by-one in RoPE offset.
- Above 1e-1: do not ship. Bug.

**Stateful multi-step chains seeded by a reference's eval-time offload: gate the e2e on a tolerance/IoU, not bit-exactness.** A reference can store intermediate state in a *lower* precision than it computes in — SAM2/EdgeTAM keeps `maskmem_features` in **bf16** as an eval-time CPU-offload memory optimization, not a correctness need. Your on-device port skips the offload and (correctly) keeps fp32 — so it is *more precise than the reference*, and a multi-frame chain that re-feeds that state diverges from goldens generated *with* the offload, growing per step. Each op is still bit-exact in isolation (gate those at 1e-3); the accumulated e2e is legitimately off — EdgeTAM's 5-frame masklet landed binary-IoU 0.92–0.98 vs the bf16-seeded goldens, not 1.0. Don't chase that gap by degrading your port to match the reference's offload precision; pick a coverage/IoU gate for the e2e and keep the per-op bit-parity as the real proof.

## Choose the publish dtype by *parity at that dtype*, never by default — fp16 is fatal for high-magnitude-activation nets

fp32 parity passing does **not** license publishing fp16. Re-run the parity gate **at each candidate publish dtype** (load the dtype-rounded weights, gate vs the fp32 golden) and ship the smallest that holds ≤1e-3 mean. The dtype is a per-model decision, not a house default:

- **fp16 collapses models with large internal activation magnitudes** — anything with an **FFT/FFC** (LaMa: rFFT bottleneck activations ~1e3), big un-normalized residual stacks, or wide dynamic range. fp16's 5-bit exponent can't represent the magnitudes; weight rounding alone drove LaMa to **mean err 0.55 (garbage)**, while **bf16** (8-bit exponent, fp32-range) held at **4e-3, visually identical to fp32**. Ship such models `-bf16` (or `-fp32`), not `-fp16`.
- **fp16 is fine — often *better* than bf16 — for well-scaled nets** (MI-GAN, well-normalized convnets): more mantissa bits, activations in range. MI-GAN fp16 mean 3e-4 vs its own bf16 1.8e-3 → ship fp16.
- Mirror of the audio note in `mlx-community-conventions.md` (narrow audio nets sometimes *need* fp16 because bf16 collapses parity): **precision is task-specific and bidirectional**. A/B fp16 vs bf16 at the weight-rounding gate before picking the suffix.
- Per-model in multi-model packages: load each at its own validated dtype (e.g. one inpaint package loads LaMa `.bfloat16`, MI-GAN `.float16`), not one package-wide quant.

## Quantized generative models — don't gate on PSNR-vs-the-fp32-golden

A quantized diffusion/generative model does **not** reproduce the fp32 reference image, and that is not a bug. int4 perturbs each denoise step slightly (per-pass weight-level cosine ~0.998), and the sampler amplifies that across steps into a *different but equally valid* image. In the Lens port, int4 e2e PSNR vs the fp32-golden image was **15.6 dB** — pure trajectory divergence, while the actual output was sharp and artifact-free.

Gate quantized models on:
1. **Per-pass weight-level cosine** vs the fp32 reference on identical injected inputs (deterministic — this is the real fidelity number; int4 ≈ 0.99+, int8 ≈ 0.9999+).
2. **Image-validity sanity** on a generated sample: finite, in range, `std > 0.1` (not gray/degenerate), no explosion.
3. **A committed visual sample** for human review.

Reserve e2e-PSNR-vs-golden for *unquantized* parity, where the trajectory should match.

Bonus: scope the quantization. Skipping the small, precision-sensitive projections (input/output embeds, time embed, final norm) lifted int4 per-pass cosine 0.9944 → 0.9976 at the *same* size — worth a `keep_hi_precision` predicate.

## Regression safety

Once parity is locked, commit the golden fixtures and pin MLX version in CI. Every MLX upgrade can shift numerics slightly — rerun parity tests on upgrade, relax thresholds only if justified by an MLX release note.

## The exact-match ceiling — greedy AR generation cannot be token-matched across backends

**Doctrine: a greedy autoregressive text capture made on different hardware is NOT a
reproducible target. Gate AR generation on (a) op-level parity at like-precision on a
deterministic stream, and (b) semantic accuracy over a statistically meaningful sample —
never on token-exact equality with a foreign-backend capture.**

This was established the expensive way in the Lance x2t investigation (2026-06,
`lance-mlx` commits `0af739e`/`500abb9`): a 6-case VQA oracle captured on an A100
(flash-attn, bf16 autocast) resisted token-exact reproduction through ~20 runs of
root-causing, even after every component was PROVEN algorithmically exact:

- Preprocessing byte-identical (max|diff| = 0.0 vs the reference's verbatim torchvision code).
- Position ids exactly equal to HF `get_rope_index`.
- Prompt token-identical (including the assistant/"\n" boundary).
- Attention semantics matched (bidirectional vision span, per-split causality).
- ViT bisected stage-by-stage vs PyTorch on the MLX **CPU** stream: cosine 1.000000 at
  every one of 32 blocks + merger.
- Cross-feeding PyTorch's own fp32 ViT features into the MLX decoder: answers unchanged.
- Whole pipeline in fp32: answers unchanged.

What remained was pure backend noise: MLX **GPU** fp32 matmul carries ~8e-4 relative
error vs CPU (M-series accumulation; see the M5 note), which compounded over 32 ViT
blocks + merger into worst-token feature cosine 0.886 — enough to flip near-tie logits.
Greedy decode then amplifies one flipped token into a fully divergent trajectory. The
A100 capture is a *third* numerics point (CUDA flash-attn accumulation order + autocast
fp32-norms structure) that even PyTorch itself cannot reproduce on Apple hardware —
flash-attn is CUDA-only. Three byte-equivalent implementations of the same model
(Python MLX, Swift MLX, an independent third-party F32 port) all converged with each
other and all diverged from the A100 capture on the same knife-edge cases.

### Recognize the regime — red flags that you're chasing the unachievable

Any TWO of these means stop chasing token-equality and switch to statistical gating:

1. Each "fix" flips a different subset of test cases instead of monotonically improving.
2. fp32 (or fp64) produces the same answers as bf16 — precision structure isn't the lever.
3. Cross-feeding reference activations into your downstream half does not change outputs.
4. Op-level parity on a deterministic stream (CPU) is 1.0 but end-to-end text still differs.
5. The divergences are single-token knife-edges (one digit, one char, an early flip that
   changes the rest of the trajectory) rather than systematic same-wrong-answer behavior.
6. The reference ran a fused/accumulation-order-different kernel you cannot run
   (flash-attn varlen, tensor-core bf16 autocast) on hardware you don't have.

### The crucial distinction — systematic vs knife-edge failures

The same Lance gate ALSO contained a genuinely real defect: two cases failed with the
SAME wrong answer on every run, every implementation, every precision ("43" instead of
"29%", 14 consecutive runs). **Stable, reproducible wrong answers are a port bug** —
that one was a preprocessing-geometry mismatch (reference used aspect-ratio-bucket
center-crop; the port used HF smart-resize), and fixing it flipped the case to the
exact oracle answer at every resolution. Rule of thumb:

- Same wrong output every run, robust to precision/backend → **systematic → chase it.**
  (Root cause lives in config/preprocessing/structure, not numerics.)
- Different outputs across configs/backends/precisions, each near a decision boundary
  → **knife-edge → statistical gate.** No amount of op-fixing will pin it.

### Oracle capture protocol (do this when CREATING references, it's 10x cheaper than archaeology)

When capturing reference outputs on rented GPU hardware for a future port:

1. **Record the full resolved config** into the capture dir (every CLI flag, resolved
   preset values, library versions, dtype/autocast mode, attention backend). The Lance
   capture recorded none of it; reverse-engineering the preset cost a session.
2. **Dump per-stage activations** (preprocessed pixels, encoder features, first-step
   logits) alongside the text — logits at step 0 are backend-comparable; full greedy
   text is not.
3. **Dump the top-5 logprobs per generated token.** This reveals which tokens were
   knife-edges (margin < noise floor) — those tokens are EXPECTED to flip cross-backend.
4. **Prefer semantic metrics on N≥50 samples** over exact-match on showcase cases.
   Vendor showcase examples are often near-memorized (answers ≈ dataset GT verbatim)
   and sit on decision boundaries — the worst possible gate set.
5. Pin and record the random seed even for "deterministic" greedy — samplers and
   batch-packing can leak nondeterminism.

### What "parity achieved" means for an AR text model port

1. Weights load strictly (no missing/unused keys); config field-for-field equal.
2. Preprocessing byte-gated against the reference implementation (run THEIR code).
3. Position ids / template tokens / attention masks exactly equal (integer artifacts —
   these CAN and MUST be exact).
4. Per-op / per-stage parity ≤1e-5 fp32 on a deterministic stream (pin MLX to CPU for
   the comparison — GPU accumulation noise will mask real op bugs at the 1e-3 level).
5. End-to-end: semantic accuracy on a meaningful sample within the reference's own
   cross-backend variance — NOT token equality with a single foreign capture.

Steps 1–4 exact + step 5 statistical = the port is done. Spending past that point on
token-matching a foreign capture is unfalsifiable work: there is no experiment you can
run locally that distinguishes "remaining port bug" from "backend noise" once 1–4 hold.

---

## Folded from the publishing / converter lineage (skills consolidation 2026-06-15)

### Task-specific end-to-end metrics

When the modality doesn't reduce cleanly to `max_abs` (the layer-level tests still do — these are the *acceptance gates*), validate by these:

| Task | Metric | Ship threshold | Calibration source |
|---|---|---|---|
| Audio source separation | SDR (vs PT reference, MUSDB18 or fixed mix) | within `±0.15 dB` | HTDemucs ±0.11 dB; Mel-RoFormer Kim Vocal 2 66.08 dB bf16 |
| Audio source separation, narrow architecture (< 100M params) | SDR | within `±0.15 dB` at fp16 — bf16 often drops 20+ dB | Mel-RoFormer ZFTurbo (33.7M params): 44.19 dB fp16 vs 21.96 dB bf16 |
| Neural audio codec (RVQ-quantized) | Codebook-index match % vs reference encoder | 100% on a held-out clip set | Mimi/SEANet encoder (anime-studio) |
| Audio classifier (multi-class, head ensemble) | Label-agreement % vs PT fp32 reference | ≥ 95% (fp16 inference acceptable) | Emotion2Vec dual-head: 98% (9-class + V/A/D) |
| TTS / audio generation | Spectral RMS-variance scan + end-to-end qualitative listen | No periodic tonal regions; speech ends in ≤ token budget | Layer parity *does not* catch tonal-tail or runaway-token artifacts — see common-pitfalls.md bf16-AR-loop entry |

For narrow audio architectures (< ~100M params, e.g. ZFTurbo): **parity-test bf16 explicitly before publishing**. If bf16 SDR drops more than a few dB vs fp16, ship the fp16 preset and document the rationale on the model card (see `mlx-community-conventions.md`).


---

## Cross-version kernel drift: the oracle's mlx version ≠ your Swift mlx version (TRELLIS.2, 2026-06-26)

When the Python-MLX oracle and the Swift port run **different mlx core versions** (oracle mlx-Python
`0.31.2` vs mlx-swift `0.31.4`), bf16/fp16 kernels round differently, and that drift **accumulates over
deep stacks** — it looks exactly like a port bug but isn't:

- TRELLIS.2 SLat flow (30 blocks, bf16): Swift vs oracle **cos 0.9986, max_abs 0.44** — alarming.
- Same model in **fp32: cos 1.000000, max_abs 2.4e-5** — the port is structurally exact.

**The decisive test = an fp32 A/B.** Env-gate the oracle's compute dtype (e.g. patch
`compute_dtype = mx.float32 if os.environ.get("TRELLIS_FP32") else mx.bfloat16`) and run the Swift port
with `computeDtype: .float32`. If fp32-vs-fp32 is cos ~1.0, the math matches and the bf16/fp16 gap is
pure cross-version kernel rounding — **ship it**. Gate on **cosine + structural validity + exact
discrete outputs** (coords, voxel counts, token IDs), **never `max_abs`**, for any cross-version or
generative comparison. This is the same doctrine as the AR exact-match ceiling above, extended to
"different mlx build" as a backend difference.

Corollary — the production port is self-consistent. The version gap only exists in the *parity harness*
(Swift 0.31.4 vs Python 0.31.2 oracle). The shipped pipeline runs one backend (mlx-swift), so its
bf16/fp16 trajectory is internally consistent; the oracle is the reference, not a runtime peer.

For discrete-output stages (sparse mesh decoders), gate on **exact topology** under identical injected
input: feed the oracle's exact intermediate (structure coords + latent) into the Swift tail and compare
voxel/vertex/face counts. TRELLIS.2 shape-decoder→mesh gave **vert ratio 1.000** (1.174M vs 1.174M) this
way — isolating the new integration glue from RNG + version drift far better than any e2e `max_abs`.

## CFG > 1: gate the PER-STEP MAP, not the accumulated trajectory (the Mage-Flow family lesson)

At cfg 5.0 an accumulated-trajectory latent comparison **fails on a correct port**: guidance multiplies each step's ~1e-2 cross-dtype noise into the next step's input, error grows ~5×/step (observed 1e-2 → 8e-2 → 4.3e-1 over three steps), and the trajectories diverge chaotically into *different-but-equally-valid* images — the CFG-time analog of the quantized-generative doctrine above. Two correct gates:

1. **Per-step reset**: feed the oracle's exact step-k input, run ONE velocity+step, compare to the oracle's step-k+1. A correct port sits flat at the cross-dtype noise floor (Mage: 1.1–2.5e-2 per step, every step); a real per-step bug grows or spikes. This same reset trick also localized the Mage timestep-embedding bug (pitfall #37) that accumulated comparison could only report as "everything diverges."
2. **Decoded-image validity** at the real defaults (steps/cfg/negative prompt from the reference workflow).

**bf16-vs-bf16 control runs: cast the INPUTS, not just the weights.** `fp32 input × bf16 weight` type-promotes back to fp32 compute, silently turning your "bf16 control" into the fp32 run you were trying to rule out — the giveaway is bit-identical numbers between the two configurations.

Also: the two-forward CFG (`batch_cfg=False`) is mathematically identical to a fused cond+uncond varlen pack **because rotary attention depends only on relative positions** — the fused pack's shifted frame indices for the uncond copy change nothing. Gate against whichever the oracle can capture; implement whichever the port's attention supports.


## Calibration anchors for cross-backend DECODED-RENDER gates (what "good" looks like)

Measured on a correct, fully parity-locked port (Mage-Flow-Edit-Turbo, 4-step, cfg 1.0,
identical injected noise, bf16-torch-CPU oracle vs bf16-MLX-GPU port):
- **512² (2 048 packed tokens): pixel-identical** decoded renders.
- **2048² (32 768 packed tokens): PSNR ≈ 34 dB, mean |Δ| ≈ 2.6/255** — visually the same
  image; divergence is per-step cross-backend noise compounded over the trajectory.

Use these as the bar: at cfg ≈ 1 / few steps, tens-of-dB PSNR is achievable and anything
dramatically below it (≲ 20 dB, or different *content*) is a bug, not "backend noise."
At high cfg the trajectory diverges chaotically (see the CFG section above) — decoded
renders then match in character, not in PSNR. Also remember character differences that
appear at a NEW size must be checked against the oracle at that size before being called
a port bug — Mage's "painterly" 2048² texture appeared identically in the reference
(model-native at 4× base), and Anima's real reference happily rendered 896×1152.
