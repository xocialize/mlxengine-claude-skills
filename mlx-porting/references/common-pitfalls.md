# Common PyTorch → MLX Port Pitfalls

Recurring traps that have broken past ports (the first six are the reading-time checklist in SKILL.md). Each comes with a concrete failure mode — read these *before* writing MLX code, not after debugging wrong output.

## 1. Constructor defaults silently diverging from model config

**What goes wrong:** A PyTorch module's `__init__` has a default that differs from the checkpoint's `config.json`. If the config doesn't store that parameter explicitly, the MLX port inherits the *default* instead of the trained value.

**Past failure (Hunyuan3D VAE, FourierEmbedder):** Default was `include_pi=True`, config specified `include_pi=False`. Not stored in config.json, so the default propagated silently into MLX. Frequencies got multiplied by π → garbage SDF values → mesh rendered as grid noise. Correlation with PyTorch output: 0.09.

**Rule:** Cross-check **every** constructor default against the source model's training config. Pay special attention to boolean flags that affect numerical computation: `include_pi`, `use_bias`, `qk_norm`, `flip_sin_to_cos`, `downscale_freq_shift`, `pre_norm`. If a parameter isn't in `config.json`, set the MLX default to match the training config value, not the framework default.

## 2. `attention_head_dim` is secretly `num_heads`

**What goes wrong:** In `diffusers.UNet2DConditionModel`, config field `attention_head_dim` is misnamed — it is actually `num_attention_heads`. The real per-head dimension is `channels // attention_head_dim`.

**Past failure (Hunyuan3D-2.1 paint UNet, SD 2.1):** Config `attention_head_dim=[5, 10, 20, 20]` with `block_out_channels=[320, 640, 1280, 1280]`. Correct interpretation:
- Block 0: **5 heads of dim 64** (not 64 heads of dim 5)
- Block 1: 10 heads of dim 64
- Block 2–3: 20 heads of dim 64

Weights (Linear 320→320) have identical shape either way, so the bug loads silently. But the reshape `(B, L, heads, dim)` puts different dims in different axes, changing the softmax pattern completely. Error per UNet pass: 0.21 × 15 denoising steps → cyan / neutral textures. Fixing to `num_heads = attention_head_dim[i]` brought parity to 1e-5.

**Rule:** For diffusers UNets: `num_heads = config.attention_head_dim[i]` directly — **don't divide**. Let `head_dim = channels // num_heads`. Verify by comparing `model.attn1.heads` to the raw config value. If building both `diffusers.UNet2DConditionModel.from_config(cfg)` and the MLX port side-by-side, the PyTorch `tb.attn1.heads` is authoritative.

## 3. Interleaved QKV vs independent reshape

**What goes wrong:** Some attention implementations concatenate Q, K, V *before* reshaping into multi-head format. The result is that heads are interleaved across the Q/K/V dimension. Standard per-tensor reshape produces totally wrong attention.

**Past failure (Hunyuan3D DiT attention):** Source did `qkv = cat([q, k, v], -1); qkv = qkv.view(B, N, heads, 3*hd); q, k, v = qkv.chunk(3, -1)`. Porting naively with separate reshapes (`q.view(B,N,heads,hd)`) produces max-diff 2.5 vs 6.7e-4 with correct interleaving. Model is trained with this interleaving so there is no choice.

**Rule:** Before writing attention, read the EXACT PyTorch reshape / view / split sequence. Any `cat(...) → view(... , 3*hd) → split` is an interleaved pattern and must be replicated as-is. Don't assume the "natural" per-tensor reshape.

See `attention-patterns.md` for the translated MLX code for both patterns.

## 4. Weight layout differences

**What goes wrong:** Conv weights have different memory layout across frameworks.

- PyTorch Conv2d: `(O, I, H, W)` — channels-first, out-first.
- MLX Conv2d: `(O, H, W, I)` — channels-last, out-first.
- PyTorch ConvTranspose2d: `(I, O, H, W)` — in-first.
- MLX ConvTranspose2d: `(O, H, W, I)` — same layout as Conv2d but with swapped channel semantics during op.
- Linear: `(O, I)` identical in both.
- Embedding: `(vocab, dim)` identical in both.

If the conversion recipe forgets to transpose, the weights load without errors but the convolution computes nonsense.

**Rule:** Use `mlx_forge.transpose.transpose_conv(key, weight, kind)` in the recipe — it handles all the conv variants generically. For hand-rolled conversion: when you see a key containing `.conv.weight` or ending in a spatial pattern, transpose.

## 5. Normalization semantic drift

**What goes wrong:** Different frameworks have different defaults and subtle behavior differences for normalization layers.

- **Default epsilon:** PyTorch `LayerNorm` is 1e-5, MLX `mlx.nn.LayerNorm` is 1e-5 — match. But diffusers `GroupNorm` often uses 1e-6, and some transformers set 1e-12. Always read the source.
- **RMSNorm weight semantics:** some implementations fuse gain into sqrt (`x / sqrt(var + eps) * (1 + gain)`), some don't (`* gain`). Check the reference.
- **AdaLN variants:** additive-only `x + shift` is NOT the same as classic `x * (1 + scale) + shift`. Hunyuan3D uses additive-only; LTX uses classic with 9 packed params; Matrix-Game has a `condition_type="token_replace"` branch.
- **GroupNorm num_groups:** if the reference uses `num_groups=32` unconditionally, don't compute it from channels — match exactly.

**Rule:** For every norm layer: read the exact PyTorch forward pass. Don't assume MLX defaults match.

## 6. Non-obvious flags and activation choices

**What goes wrong:** A small config flag flips a computation subtly and silently.

Common culprits:
- `qk_norm` — if true, Q and K are normalized before the dot product. Skipping this changes attention scale significantly.
- `use_bias` on Linear / Conv / QKV projections — differs across block types within the same model.
- `cross_attention_dim` — when cross-attn is present but dim differs from self-attn, projection shapes differ.
- Activation: GEGLU (`gate * gelu(up)`), SwiGLU (`gate * silu(up)`), GELU (no gate), ReLU². Each has a different parameter count and math.
- `sandwich_norm`, `pre_norm` vs `post_norm` — position of norm relative to attention / FFN changes residual math.
- `flip_sin_to_cos`, `downscale_freq_shift` in timestep embedding — affects which half of the sinusoidal encoding goes where.

**Rule:** Before translating a block, list every flag the PyTorch class checks and write a mini-table mapping config value → code branch. Then translate each branch.

## Other subtler pitfalls

- **RNG is not seed-compatible.** `mx.random.normal(key=…)` and `torch.randn(generator=…)` use different algorithms. For parity tests, generate once in numpy and inject into both sides.
- **In-place semantics.** `x[idx] = y` in MLX is a rebind (copy-on-write-style), not true mutation. Don't port PyTorch code that relies on in-place performance.
- **`tensor.contiguous()`** has no exact MLX analog — arrays are logically contiguous. Ignore contiguous calls unless downstream expects a specific stride (rare).
- **`torch.einsum`** exists as `mx.einsum` but subscript semantics are identical. Prefer keeping einsum expressions as-is rather than rewriting to matmul — fewer bugs.
- **`F.scaled_dot_product_attention`** with `is_causal=True` → use `mx.fast.scaled_dot_product_attention(q, k, v, mask="causal")`. For GQA (fewer KV heads), pass unequal-head Q and K — MLX handles it natively.
- **`torch.cumsum` on bf16** differs numerically from fp32 cumulative sums. If parity is just-above-threshold, try casting to fp32 for the cumsum step.
- **`F.silu` vs `F.hardswish`** — easy typo when reading quickly. Triple-check activation names.
- **`mx.pad` has no `reflect` mode** (only `constant`/`edge`). PyTorch/torchaudio STFT and mel frontends pad `mode="reflect"` — replicate it by hand: `mx.concatenate([x[..., 1:p+1][..., ::-1], x, x[..., -p-1:-1][..., ::-1]], axis=-1)` (don't repeat the edge sample). Bit both the Zonos speaker mel and any Whisper-style log-mel. Bake fixed transforms (mel filterbank, window) as loaded buffers rather than recomputing torchaudio's formula — guarantees parity (verified ~1e-6).

## 7. The checkerboard trap (recurring — watch for it actively)

**What goes wrong:** The per-layer parity tests pass at small scale with random weights, but end-to-end inference with real weights produces an image with a visible checkerboard pattern at a specific scale (8×8, 16×16, or matching the patch/upsample stride). The model is *almost* working — the DiT moves latents away from noise, the VAE decodes without crashing — but the output has periodic artifacts instead of a coherent image.

**Past failures:**
- **Hunyuan3D paint UNet:** 2×2 mosaic at each decoder stage — wrong axis order in `F.interpolate` equivalent.
- **ERNIE-Image v0:** 8-pixel checkerboard — off-by-one in text-encoder `hidden_states[-2]` that caused the DiT to receive text conditioning with 10× wrong magnitude *and* shifted scale.
- **Qwen-Image:** VAE `Upsample2D` used `mx.repeat` on the wrong axis (channels instead of spatial) → tile-like artifacts.

**Root causes to check, in order of frequency:**

1. **`mx.repeat` vs `mx.tile` confusion.** `mx.repeat(x, 2, axis=1)` on `(B, H, W, C)` duplicates each H-row consecutively (`[a, a, b, b, ...]`) — the correct nearest-neighbor upsample. `mx.tile(x, (1, 2, 1, 1))` produces `[a, b, ..., a, b, ...]` — block tiling, which always checkerboards. Never use `mx.tile` for upsampling.

2. **Pixel-shuffle / patch pack axis order.** Both PT `reshape(B, C, H/r, r, W/r, r) → permute(0,1,3,5,2,4)` and MLX channels-last `reshape(B, H, W, C, r, r) → transpose(0,1,4,2,5,3)` produce the same logical output. Verify by a round-trip identity test on a small known tensor (not random!) — feed `mx.arange(total)` reshaped to the input shape, unpatchify, re-patchify, and assert equal.

3. **Position IDs axis swap.** `rope_axes_dim=[text, y, x]` with grid constructed via `meshgrid(grid_y, grid_x, indexing="ij")` must stack `[yy, xx]` (not `[xx, yy]`). A swap produces subtle checkerboard because y-axis RoPE rotates channels the DiT expects to carry x-info and vice versa.

4. **Text-conditioning magnitude mismatch.** Check `hidden_states[-2]` semantics in HF transformers: it's the output of layer `N-1` — which is the INPUT to the last layer, not the OUTPUT of the last layer. So use `for layer in layers[:-1]` (apply N-1 layers), NOT `for layer in layers` then return pre-norm. Off-by-one here produces correct-looking conditioning at wrong magnitude — DiT denoises in the wrong latent manifold, checkerboard at the VAE decoder.

5. **Dtype leak from scheduler into DiT.** The `FlowMatchEulerDiscreteScheduler.step()` in mlx-arsenal keeps `sigmas` as `fp32`; multiplying a `bf16` latent by an `fp32` scalar promotes the latent to `fp32`. The next DiT forward then runs with `fp32` input vs `bf16` weights. Cast back: `latents = scheduler.step(...).astype(dtype)`.

**Diagnostic procedure — use this EVERY port before shipping:**

```python
# Test 1: decode pure Gaussian noise through the VAE only.
# If checkerboard appears here, the VAE is broken (upsample, conv layout).
z = mx.random.normal((1, lat_H, lat_W, lat_C)) * 2.0
img = vae.decode(z)
# → should be smooth coloured noise, no periodic pattern.

# Test 2: decode noise through the *full post-DiT chain* (BN-inverse +
# pixel-shuffle + VAE). If smooth here but checkerboard end-to-end, the DiT
# output is the source.
dit_shape = (1, dit_C, lat_H // patch, lat_W // patch)
dit_out = mx.random.normal(dit_shape)
nhwc = dit_out.transpose(0, 2, 3, 1)
nhwc = vae.bn.apply_inverse(nhwc)  # if model has latent BN
unpacked = pixel_shuffle(nhwc, upscale_factor=patch)
img = vae.decode(unpacked)
# → still smooth noise. If checkerboard, the BN or pixel_shuffle axis is off.

# Test 3: real-weight DiT single-block vs reference.
# If parity < 1e-3, and tests 1/2 pass, the bug is in attention conventions
# (positions, mask format, qk_norm, RoPE axis order) which only manifest at
# the model-trained magnitudes.
```

**Scoring the FFT — compare, don't threshold.** If you quantify "periodic pattern" via an FFT
peak/median ratio, do **not** gate on an absolute number: a *correct* VAE decode of random latents has a
high inherent peak ratio (measured ~626 for a random latent, ~2452 for a zeroed latent through the Wan2.1
VAE — structured upsampling concentrates spectral energy), so an absolute threshold false-positives on a
working decode. The real signal is a **comparison** — run the suspect chain (Test 2: through your new
BN-inverse / pixel-shuffle / patchify code) *and* a plain `vae.decode(z)` baseline on the **same** random
latent; a stride bug shows up as the new-chain ratio **exceeding** the direct-decode baseline, not as a
large absolute number. (Phantom-Wan: through-DiT 583 ≤ direct-decode 626 ⇒ wiring clean.) When the spatial
chain is **inherited** from an already-ported base (mlx-video VAE/patchify), this test is largely moot —
a stride bug would already show in the base; spend the effort on your delta.

**Preventive: add these three tests to every port's `tests/smoke/` BEFORE running full generation.** They pinpoint the layer at fault in under 30 seconds each and catch 95% of checkerboard bugs.

**Rule:** If the end-to-end output has checkerboard, do not tweak the denoising loop parameters (steps, guidance, shift) — they can never fix a wrong spatial operator. Walk up the stack with the three tests above instead.

## 8. Tekken / Pixtral tokenizer skips the BOS

**What goes wrong:** When wiring an `mlx-lm` Mistral-family text encoder (Ministral3, Mistral Small 3, Pixtral) to a diffusion DiT, you tokenize with `add_special_tokens=True` expecting `<s>` at position 0 — and it silently does NOT get added. Every content token is fine, but the FIRST token enters the attention stack at an out-of-distribution magnitude, compounds layer by layer, and the DiT receives conditioning that's correct everywhere except position 0. End-to-end generation produces a structured-but-incoherent image (barely-visible features buried in noise).

**Past failure (ERNIE-Image port, 2026-04-20):** `mistral-community/pixtral-12b` tokenizer bundled as the runtime tokenizer. Layer-by-layer diff between `mlx_lm.models.ministral3.Model` and `transformers.Ministral3Model` on identical input_ids:

| layer | token-0 max_abs | token-1..N max_abs |
|---|---|---|
| 0-1 | 0.003 | 0.005 |
| 2   | 4.18  | 0.004 |
| 3-25 | 4-7 | 0.02-0.07 |

Token-0 is 100× off from every other position and grows across the stack. Fix is prepending `tokenizer.bos_token_id` before the first content token. Once done, the content tokens 1..N keep their same (tiny) divergence and the DiT receives correct conditioning in the positions it actually uses.

**Rule:** In your pipeline's `_tokenize` / `encode_prompt` helper:

```python
ids = tokenizer(prompt, add_special_tokens=True, ...)["input_ids"]
bos = tokenizer.bos_token_id or 1
if not ids or ids[0] != bos:
    ids = [bos] + ids
```

Do this for every Tekken-family tokenizer, not just Pixtral. `add_special_tokens=True` is an identity op for these backends even though the argument name suggests otherwise.

## 9. Structural drift from the reference repo (the `ltx-2-mlx` lesson)

**Symptom:** the port "works" on simple paths (e.g. T2V) but subtle bugs on adjacent pipelines (keyframe, image-to-video, multi-stage) take hours to track because the MLX code no longer maps 1:1 to upstream. Every investigation hypothesis costs a round-trip: "is this divergence the bug? prove sematic equivalence first."

**Root causes** observed on `ltx-2-mlx` — all variants of the same anti-pattern (taking the shortcut during translation):

1. **Reordering operations because it's easier to write that way.** Stage 2 noised state was built `noise → cond` with a manual `LatentState` constructor instead of going through upstream's `noise_latent_state(state)` helper which orders `cond → noise`. Numerically equivalent here, but the divergence had to be proven, not assumed.
2. **Passing arguments upstream leaves at default.** Sigma `num_tokens = F×H×W` was threaded through because "you have to pass the token count somewhere", missing that upstream uses the default `4096`. The port reads as more deliberate than the original.
3. **Porting against a stale upstream version.** CFG was implemented via a direct `guided_denoise_loop` because the port pre-dated upstream's refactor to `FactoryGuidedDenoiser` + `euler_denoising_loop`. Once upstream evolves, the port diverges further with every sync, and structural drift compounds.
4. **In-place mutation instead of porting the abstraction.** LoRA was fused in-place between stages because it "worked", instead of porting upstream's pattern of constructing `stage_2 = DiffusionStage(loras=base+distilled)` at init time.
5. **Flattening abstractions into the pipeline.** Upstream's `DiffusionStage` / `ModalitySpec` / `EulerDiffusionStep.step` / `euler_denoising_loop` decomposition was collapsed into procedural code inline in the pipeline. Shorter to write, but every future upstream commit becomes a 3-way merge instead of a 1:1 sync, and the cohesion that made bugs *localizable in upstream* is lost in the port.

**The right structure** would have been isomorphic:
- `DiffusionStage` class with `__call__(denoiser, sigmas, noiser, video=ModalitySpec, audio=ModalitySpec)`
- `ModalitySpec` dataclass with `context, conditionings, noise_scale, initial_latent`
- `create_noised_state(tools, conditionings, noiser, ..., initial_latent)` orchestrating init → cond → noise in upstream's order
- `EulerDiffusionStep.step(sample, denoised, sigmas, step_idx)` as the stepper
- `euler_denoising_loop` looping over it
- `FactoryGuidedDenoiser` for CFG, not a direct `guided_denoise_loop`

**Diagnostic rule (raise this first, not last):** when chasing a subtle bug in a port, the **first** diagnostic tool is a side-by-side read of the upstream pipeline vs the MLX pipeline — not a numerical parity test. Structural divergences jump out instantly to the eye and eliminate whole classes of hypotheses before any tensor is materialized. Parity tests come second, to confirm that the (now visibly aligned) code actually matches numerically.

**Prevention rule:** during initial translation, if you catch yourself thinking "it's simpler to write it this way" or "I don't need this abstraction" or "I'll just inline it" — stop. That sentence is the drift starting. Port the abstraction verbatim. Note the cleanup idea for a post-parity pass that may never come, and that's fine.

## 10. Config-*class* defaults the checkpoint `config.json` never serialized (the rope-scaling / YaRN trap)

The single costliest bug in the `lens-mlx` port. The Lens text encoder is GPT-OSS-20B, which uses **YaRN rope** (`rope_type="yarn"`, factor 32, and an `attention_scaling`/mscale ≈ **1.3466** that multiplies cos/sin at *every* position). But that rope config is a **`GptOssConfig` class default in HF transformers** — so the checkpoint's `text_encoder/config.json` does **not** contain `rope_scaling` or `rope_theta`. mlx-lm reads the json, sees nothing, and falls back to plain rope (mscale = 1.0).

**Failure mode:** bf16-vs-bf16 parity stuck at cosine ~0.94 (NOT a quant gap — matched precision both sides). Activations ~half the reference magnitude, **uniform from layer 0**, cosine *rising* with depth (a few giant outlier dims dominate cosine in the middle layers and mask it). All *weights* matched exactly (cos 1.0), so it was pure forward compute.

**The general trap:** `config.json` is NOT the full config. The parent config *class* (transformers `*Config`, diffusers `register_to_config` defaults) injects fields the checkpoint omits — rope scaling, sliding-window sizes, normalization flags. mlx-lm / a hand-port only see the json.

**Rule:** never trust `config.json` for rope. Compare the **resolved** rope on both sides before trusting any forward:
```python
# PT side — what HF actually built:
pt.model.rotary_emb.attention_scaling      # 1.3466 for yarn, 1.0 for plain
pt.model.rotary_emb.inv_freq[:8]
pt.config.rope_scaling                      # the resolved dict, NOT the json
# MLX side — what mlx-lm/your port built:
mx_rope.mscale, mx_rope_freqs[:8]
```
If the HF model *object* has rope params the json lacks, inject them into the mlx config before building the model (mlx-lm: `load_model(path, model_config={"rope_scaling": {...}, "rope_theta": ...})`). The fix that took Lens encoder cosine 0.94 → 0.998 was a 7-line injection of the gpt-oss YaRN dict when the on-disk config omitted it.

## 11. Guidance / CFG is often NOT vanilla — port the exact combine step

Don't assume `noise = uncond + scale * (cond - uncond)`. Lens uses **norm-rescaled CFG**: after the standard combine, it rescales per-token by `‖cond‖/‖comb‖` (`noise_pred = comb * (cond_norm / comb_norm)`). Skip the rescale and the output magnitude is silently wrong — plausible-but-off images that pass shape checks.

Likewise, timestep scaling can be **split across modules**: Lens has `scale=1000` inside the time-projection AND `timestep/1000` in the loop; they compose to identity-ish but each must be ported. Read the *whole* `__call__` of the reference pipeline (guidance combine, rescales, clamps, timestep transforms) and port it verbatim — these live in the pipeline, not the model, so they're easy to miss when you only port the DiT.

## 12. Derivative of an already-ported base — diff the config + weight keys FIRST, port only the delta (the `bernini-r-mlx` lesson)

Many "new" models are a fine-tune or thin wrapper over an architecture you (or mlx-community) already have. **Before scoping a Tier-3 port, diff the candidate against the ported bases (Wan, LTX, Lance, Qwen, SD3/FLUX).** This routinely collapses a Tier-3 estimate to "Tier-2 + reuse".

Bernini-R was nominally a 14B-class video DiT with a novel positional scheme (SA-3D RoPE), an MLLM planner, and editing/reference conditioning — sounds like weeks. The actual port was days, because:
- The diffusers `transformer/` weight key set was **byte-for-byte the stock Wan2.2 `WanTransformer3DModel`** — *zero* extra tensors. So the entire DiT + VAE + UMT5 + scheduler was reused verbatim from `mlx-video`; only the conditioning *assembly* was new.
- The "novel" SA-3D RoPE added **no parameters** (see #13) and its 3-axis split + theta were **identical** to the base's existing rope — a 30-line per-segment phase multiply on top of the reused `rope_apply`.

**The procedure (do this in Phase 0, before any code):**
1. Pull the candidate's `config.json` + the safetensors **index** (`*.index.json`) — the index alone gives you every weight key and `total_size` without downloading a byte.
2. Collapse the per-block key pattern (regex `\.\d+\.` → `.N.`) and **diff it against the ported base's key set.** Identical block structure ⇒ reuse the base's blocks; extra keys ⇒ that's your real delta.
3. Diff the two `config.json`s. Knobs that differ but name no new tensors are *pipeline/runtime* params (boundary, shift, a rope flag), not new layers.
4. Read one shard's safetensors header for **dtype** (a few-KB HTTP range read) before sizing the conversion — diffusers re-exports are often de-EMA'd and a different dtype than the raw release (Bernini raw = 84 GB fp32 EMA; the diffusers repo = clean ~14B you actually convert).

The deliverable of this phase is a key-mapping table (base ↔ candidate) and an enumerated delta. Writing model code before that table exists is how you re-implement something already sitting in `mlx-video`.

**Reuse the sibling port's already-*converted* weights, not just its architecture.** When a component is byte-identical to one a sibling port already converted to MLX, copy the safetensors — don't re-download + re-convert the source. Phantom-Wan's umT5-XXL is the same `google/umt5-xxl` Bernini-R already shipped in mlx-video format, so it reused `bernini-r-mlx-weights/…/t5_encoder.safetensors` verbatim and **skipped an 11.36 GB download + conversion**. Same for a shared VAE. Verify byte-identity at the *config + key* level first (the umT5 is shared across the whole Wan2.1/2.2 family; the VAE is *not* — Wan2.1/2.2-T2V use the 16-ch VAE, Wan2.2-TI2V/Lance the 48-ch — so confirm channel count before assuming a VAE is reusable).

## 13. A config flag is not a weight — verify advertised capabilities against the actual checkpoint key set (the "-R"/component-release trap)

`use_src_id_rotary_emb: true` in Bernini's config sounds like a learned module. It has **zero parameters** — it's a runtime position-id scheme over the existing rotary. Conversely, the paper's headline "semantic planner" (a Qwen2.5-VL) is referenced everywhere but **has no weights in the released checkpoint** — only the *Renderer* ("-R") was open-sourced. A component/"-R" release routinely omits the part carrying the paper's marquee capability.

**Rule:** before promising a feature in a model card, grep the actual checkpoint key set for the tensors that feature requires. A flag, an `architectures` name, or a paper section is not evidence the weights exist. Two concrete checks:
- "Does this flag add parameters?" → search the index for keys unique to it; if none, it's runtime-only (port it as logic, expect it to be parameter-free, and for a single-segment/default input it often reduces to the base behavior — a free parity anchor, see parity-testing.md).
- "Can I actually run the advertised system?" → if the headline capability's weights are absent, scope the port to what the released weights *can* do and say so plainly in the card (Bernini-R is renderer-only / planner-absent → UMT5-only conditioning).

## 14. API surface ≠ capability — verify implementation, not signature (the streaming-decode trap)

**What goes wrong:** An upstream module exposes plumbing-shaped parameters that suggest a capability exists, but the implementation never honors them. Naively trusting the signature produces wrong output rather than a clean error — there's no `NotImplementedError`, no warning, just silently incorrect behavior.

**Past failure (cross-port analysis for Lance PR #7 → mlx-video stock `wan_2/vae.py`):** Reading the public API of `mlx_video.models.wan_2.vae`, every block-level `__call__` takes `feat_cache=None, feat_idx=None` parameters — the same plumbing pattern Lance uses for streaming decode. The encoder path honors them (`WanVAE.encode` allocates `feat_cache = [None] * num_slots` and threads chunked encode). It looks like a consumer could allocate the same feat_cache, call `vae.decode` with it, and get streaming behavior.

Reading the *implementations* of those same methods: `Resample.upsample3d` runs `time_conv` unconditionally regardless of `feat_cache` (no "Rep" sentinel, no first-call skip, no cross-chunk state). `WanVAE.decode()` never allocates a feat_cache list at all — it calls `self.decoder(x)` whole-sequence. `decode_tiled` exists but uses lossy trapezoidal-blend (~1-5 px/255 error vs whole-sequence), not the bit-identical streaming pattern the parameters imply. A consumer who hand-threaded `feat_cache` through the block API would silently get whole-sequence output mixed with cache-state pollution.

**Rule:** When an upstream module exposes "plumbing" parameters (state-threading args like `feat_cache`/`past_kv`, callback hooks, optional `mode=` flags, "chunked" or "streaming" boolean kwargs), the signature is documentation, not contract. Verify by reading the implementation:

- Does the parameter appear past the function's first dispatch branch, or only in the signature?
- Are there code paths that reference it in *all* the conditional branches where it would matter?
- Is the top-level entry point (`decode`, `forward`, `generate`) actually allocating and threading the state, or does it call the inner blocks with the params defaulting to `None`?

First sign that a parameter is decorative: it's `None` by default, accepted by the inner blocks, and the public entry point doesn't pass it. Second sign: a "tiled"/"streaming"/"chunked" alternative method exists but uses a lossy approximation (cross-fade, trapezoidal blend) instead of the bit-identical cache pattern the params imply — that's the upstream punting on the hard version. Third sign: the encoder path uses the param but the decoder mirror doesn't (or vice versa); the asymmetry means someone wired one side and abandoned the other.

**Diagnostic:** Before relying on an upstream's purported capability, write a 10-line bit-identity test: `assert streaming_output == whole_sequence_output`. If you can't write that test because the upstream method doesn't actually exist (only the params do), the capability isn't there.

## 15. Consumer-side extension before fork — audit the public submodule surface (the `vae_stream.py` pattern)

**What goes wrong:** When an upstream module lacks a capability, the default move is to vendor / fork / monkey-patch the upstream and add the feature in-place. This forks your dependency tree, breaks upstream version pinning, and pays maintenance forever after. Many upstreams expose enough public submodule surface that the missing capability can be added as a free function from the *consumer* level, with zero upstream changes.

**Past failure (cross-port effort estimation, same session as #14):** Initial estimate for adding lossless streaming decode to mlx-video stock `wan_2/vae.py` was 4-6 hr including a `Resample.upsample3d` modification (~125 LOC added to upstream, requiring a vendor / fork / upstream PR). The reasoning: pitfall #14's API-vs-impl gap means `Resample.__call__` doesn't honor `feat_cache`, so it has to be modified to support the "Rep" sentinel pattern.

Re-reading the upstream port we were modeling against (`lance-mlx/src/lance_mlx/model/vae_stream.py`) revealed it imports the standard upstream `Resample`, `CausalConv3d`, `ResidualBlock` etc. unchanged. It implements cached upsample as a free function:

```python
def _resample_upsample3d_cached(rs: Resample, x, fc, fi, first, ...):
    # Doesn't modify rs.__call__. Calls inner ops directly:
    cached_in = mx.concatenate([fc[idx], x], axis=...) if fc[idx] else x
    out_t = rs.time_conv(x, cache_x=fc[idx])      # public attr access
    # ... spatial part via rs.resample[1] directly:
    spatial = rs.resample[1](out_t.transpose(...))  # public attr access
```

`Resample.__call__` is bypassed entirely. The streaming logic lives in the consumer's free function; the upstream module is used as a bag of weight-bearing submodules. Stock mlx-video's `Resample` exposes the same attributes (`self.time_conv`, `self.resample[1]`), and `CausalConv3d.__call__` already takes `cache_x`, so the same consumer-side pattern works against mlx-video stock with zero upstream changes. Revised effort: 3-5 hr, no fork, no PR dependency.

**Rule:** Before vendoring, forking, or monkey-patching an upstream module to add capability, audit whether its submodules expose enough public surface to implement the capability as a free function from the outside. Concrete checks:

- Are the relevant inner ops accessible as **public attributes** (`module.sub_op`, not `module._sub_op`)? Python's underscore convention is your contract: public = you can rely on it; underscore = upstream may change it.
- Do those inner ops take the **state / cache parameters** you need (e.g. `cache_x`, `past_key_value`)? If yes, you can call them directly instead of going through the parent's `__call__`.
- Can you replicate the **orchestration** (the ordering, the per-chunk loop, the post-processing) in your own code? If the inner ops are accessible but the parent's `__call__` does ten lines of bookkeeping you'd have to re-implement, the bypass might cost more than it saves.

**Reference example:** `lance-mlx/src/lance_mlx/model/vae_stream.py` (426 LOC, weights-free bit-identity test in `tests/test_decode_stream.py`, zero upstream edits). The pattern: import the standard module classes, implement the new capability as free functions that operate on instances, call the public submodule attrs directly when needed. The result survives upstream changes that don't touch the internals you're depending on.

**When this pattern fails:** if the upstream's inner op state is package-private (`_internal_cache`, `__hidden`), or if the parent's `__call__` mutates shared state you can't safely bypass, or if the consumer-side bypass would have to replicate >50% of the parent's logic, then vendor or fork is the right answer. But check first — the default move to fork is often expensive and unnecessary.

## 16. Faithful `torch.stft` / `torch.istft` in MLX — framing + NOLA, gated on CPU (the `cocktail-fork-mlx` lesson)

MLX has only `mx.fft.rfft` / `mx.fft.irfft` — no `torch.stft`/`istft`. Audio ports (separation, vocoders, TTS frontends) need a faithful hand-roll. The exact recipe that hit first-try parity (`< 1e-4`) for MRX:

**Forward** (`normalized=True, center=True, pad_mode="reflect"`):
1. **Reflect-pad** `n_fft // 2` each side **manually** — `mx.pad` has no reflect mode, and reflect must *exclude* the boundary sample: left = `x[..., pad:0:-1]`, right = `x[..., -2:-pad-2:-1]`.
2. Frame via fancy indexing: `starts = arange(n_frames)*hop; idx = starts[:,None] + arange(n_fft)[None,:]; frames = x[:, idx]`.
3. Multiply by a **periodic** Hann window (`0.5 - 0.5*cos(2π k / n)`, matches `torch.hann_window(n, periodic=True)` = `np.hanning(n+1)[:-1]`).
4. `mx.fft.rfft(frames, n=n_fft)` then `/ sqrt(n_fft)` for `normalized=True`.

**Inverse** = windowed **overlap-add with NOLA normalization** (what torch.istft does internally): `* sqrt(n_fft)` → `irfft` → multiply by the synthesis window → scatter-add each frame into the output AND accumulate `window²` into a weight-sum buffer → divide signal by the weight-sum (guard ~0) → trim `n_fft//2` + to `length`.

**Gate on `mx.cpu`.** Apple-GPU fp32 FFT/matmul is tf32-like and won't reach `1e-4` — run STFT/iSTFT parity on CPU (same lesson as Zonos). Round-trip `istft(stft(x)) ≈ x` on the interior is the quickest sanity check.

## 17. Bidirectional / multi-layer LSTM from torch weights — MLX `nn.LSTM` won't do it (the `cocktail-fork-mlx` lesson)

MLX's `nn.LSTM` is **unidirectional and single-layer**, and its gate convention may not match torch. For any BLSTM/stacked-RNN port, implement the cell directly from torch's parameters — it makes weight conversion a **straight key-rename, zero transpose** (torch `weight_ih_l{L}[_reverse]` is `[4H, in]`, MLX consumes `x @ W.T` identically; Linear `[out,in]` is also identical):

- Gate order is torch's `[i, f, g, o]` (rows `0:H, H:2H, 2H:3H, 3H:4H`): `i,f,o = sigmoid`, `g = tanh`, `c' = f*c + i*g`, `h' = o*tanh(c')`.
- **Hoist the input projection** out of the time loop: precompute `gx = x @ W_ih.T + b_ih` once; only `h @ W_hh.T` recurs.
- Bidirectional = run forward + a second pass over the reversed sequence, then `concat` along features → feeds the next layer.
- **Bound the lazy graph:** a python time-loop over a long sequence (minutes of audio → 10k+ steps) builds one giant graph and OOMs / stalls. Call `mx.eval(h, c)` every ~512 steps. (For production speed, batch independent branches + both directions into one loop — but only after parity is locked.)
- Perf note: such RNN-bound models are often **faster on CPU than GPU** — tiny per-step ops are GPU-kernel-launch-bound.

## 18. A target fork that followed the reference may already have the capability — audit the top-level entry point first (the three-pass LongCat lesson)

**What goes wrong:** When considering porting an optimization (or any feature) from source repo A to target fork B, the natural focus is on the source's *innovation* — what makes the source's implementation worth porting from. But target fork B was originally implemented by reading some *reference* (typically a PyTorch one), and if that reference already had the capability, B inherited it for free at port time. Estimating cross-port effort from the inner-block surface or class-hierarchy comparison misses this: the estimate is shaped like "we need to add X, Y, Z to B" when X, Y, Z are already in B's top-level entry point. The wasted effort isn't just the misjudgment — it's downstream planning decisions (sequencing, who-does-what assignments, time budgets) made on top of a wrong scope.

**Past failure (this session, three-pass LongCat audit for Lance PR #7 streaming decode):** Initial cross-port estimate for porting Lance's lossless streaming VAE decode to LongCat's `AutoencoderKLWan` was **4-8 hr**. After a first audit of `Resample.upsample3d`, revised to **2-4 hr** ("Rep sentinel already wired in the inner block; just add the top-level `decode_streaming` orchestrator"). After writing the deferred-port handoff doc with a deeper audit of `AutoencoderKLWan.decode()` itself, revised to **~0 hr** — the per-frame streaming loop with `feat_cache` threading was already the default code path (autoencoder_kl_wan.py:716-735). The original LongCat VAE port author had implemented streaming-by-default when matching the diffusers `AutoencoderKLWan` reference, where streaming is also the default. Three iterations of misjudgment over the session, each one peeling off ~2 hr from the estimate, before the actual answer (nothing to port) surfaced.

**Rule:** When estimating cross-port effort for transferring a capability from source repo A to target fork B, audit **top-down, not bottom-up**:

1. **First read B's top-level entry point** for the capability area (`decode`, `generate`, `forward`, `sample`, etc.). Does the entry point already implement the streaming / chunking / caching / optimization pattern as its default code path? If yes, the rest of the analysis is "what's missing or different" not "what needs to be added."
2. **Audit downward only if (1) is negative.** Inner blocks may expose the right plumbing parameters (`feat_cache`, `past_kv`, hooks), but signature-level presence doesn't imply the orchestrator wires them — see pitfall #14. And conversely: signature-level *absence* of a named-streaming method (no `decode_streaming`, just `decode`) doesn't imply the capability is missing — the streaming might be the default with no whole-sequence alternative to distinguish from.
3. **Diff against the reference B was originally ported from.** If reference R already had the capability and B aimed for layer-by-layer parity with R, B inherits it. Saved estimation time scales with how faithfully the original port author followed R. For diffusers-aligned MLX ports targeting capabilities that exist in diffusers, default-assume the capability is there until proven absent.

**Failure mode is symmetric with pitfall #14:** there the trap is the API surface *looks like* it has capability X but the implementation ignores the params; here the trap is the API surface *doesn't advertise* capability X (no specially-named method, the entry point just looks ordinary) but the implementation has it as the default. Both traps come from reading signatures instead of bodies.

**Diagnostic shortcut:** before scoping any cross-port effort estimate, do a **5-minute top-down read** of the target fork's main entry point for the capability area. If the per-step loop, cache allocation, and chunked-state plumbing are visible at the entry point, the capability is already there. If the call is a one-liner like `out = self.decoder(x)` or `return self.forward(x, ...)`, then port work is genuinely needed. The check is fast and front-loads the most expensive misjudgments.

## 19. Gradient accumulation in MLX — `mx.eval` every micro-batch or the lazy graph explodes (the `siglip2-nriqa-mlx` lesson)

Applies whenever a port involves **training/fine-tuning** (LoRA reproductions, from-paper IQA/regression heads, distillation). MLX is lazy: a `value_and_grad` call builds a forward+backward graph but does **not** evaluate it. If you accumulate grads over N micro-batches and only `mx.eval` at the optimizer step (every N), then **N full forward+backward graphs — including all backbone activations — pile up unevaluated**, and through a large frozen backbone (autograd still stores activations to reach LoRA params in early layers) that's tens of GB.

**Past failure:** SigLIP2-SO400M (430M) LoRA fine-tune, phys-batch 2 × grad-accum 6. A 12-micro-batch test passed (peak 12.6 GB); the full run **silently died at epoch 0** — no traceback, no jetsam log, just a dead process and ~23k pageouts. The macOS swap thrash looked like an idle GPU ("GPU not active"). One-step sanity tests don't catch it because a single graph fits; the blowup only appears once ≥2 graphs accumulate without eval.

**Rule:** materialize each micro-batch before moving on:
```python
l, g = loss_and_grad(model, x, y)
g = tree_map(lambda t: t / accum, g)
gacc = g if gacc is None else tree_map(lambda a, b: a + b, gacc, g)
mx.eval(gacc, l)                       # <-- frees this micro-batch's graph immediately
if (i + 1) % accum == 0:
    opt.update(model, gacc); mx.eval(model.parameters(), opt.state); gacc = None
```
Verify with memory, not loss: peak RSS / `mx.get_peak_memory()` must be **flat across micro-batches**, not climbing. A "fast 12-step sanity" that doesn't watch memory across an accum boundary is not sufficient — instrument peak memory over ≥ 2 full accumulation cycles.

## 20. Long training runs can't live inside the agent harness — use an external terminal (the `siglip2-nriqa-mlx` lesson)

A from-paper port that needs *training* (not just inference) often means a multi-hour run. Two harness execution modes both fail it, in opposite ways:

- **Tracked background task** (`run_in_background`): reaped at turn boundaries — the process dies shortly after the startup print, before the first epoch.
- **Fully detached daemon** (`setsid`/double-fork → reparented to launchd): survives, but macOS clamps it to **background QoS** and starves its GPU access — it runs ~10× slower (state `S`, RSS pinned, "GPU barely used"), which masquerades as a hang.

Foreground runs at full speed but is capped by the Bash-tool timeout (~10 min) and blocks the turn, so it can't host a 15-hour job either.

**Rule:** for any training/fine-tune run longer than a few minutes, **hand the user a copy-paste command to run in a normal Terminal** — foreground QoS = full speed, and it's independent of the harness lifecycle. Make the run robust for unattended execution first:
- `caffeinate -dims …` (prevent idle/system sleep), ideally inside `tmux`.
- **Per-epoch checkpoint + `--resume`** (save trainable params + epoch + best each epoch) so a stop/restart loses at most one epoch.
- Save the **best** checkpoint on metric improvement separately from the latest, so a partial run is already shippable.
- Log a parseable per-epoch line (`epN … SRCC … (Ns)`) so progress can be confirmed from a pasted snippet.

The agent's role becomes: build + sanity-check the pipeline, hand off the command, then validate the saved checkpoint and publish when the user reports completion. (SigLIP2 NR-IQA: external-terminal run hit SRCC 0.9575, beating the paper, after this pivot.)

## 21. "A port exists" doesn't end CONFIRM gate #2 — check *which runtime* it serves (the Python-exists-but-no-Swift reframe)

CONFIRM gate #2 ("port-status re-verify") fails open if you stop at *found one*. An existing MLX port can leave the real gap wide open. Two recurring shapes, both seen multiple times:

- **A "port" that's a code-less weights blob.** `themindstudio/RealESRGAN-x4plus-mlx` was just a 67 MB `.npz` (one variant, no source, no Swift, stale). "Maintenance fork" was the wrong frame — it became a **net-new clean port** (`realesrgan-mlx`, 5 variants, tiling, published).
- **An experimental MLX-*Python* port when the consumer is *Swift*.** `mflux` (SeedVR2) and `gaarutyunov/vjepa2-mlx` (V-JEPA2) are real Python impls — but with **no parity claim, no published `mlx-community` weights, and no Swift**. If the strategic consumer is on-device Swift, net-new Python has low marginal value; the high-value move is **Python-first → Swift**: build a parity-verified Python port + publish the missing weights artifact (which doubles as the Swift port's oracle), then do the Swift port nobody has.

**Rule:** when gate #2 finds a port, classify it on three axes before deciding scope — *(a) is there actual source or just weights? (b) are converted weights published to `mlx-community`? (c) does it serve the target runtime (Python vs Swift)?* The gaps on (a)/(b)/(c) — not the mere existence — set the work. Don't let "found one" collapse the evaluation.

## 22. The reference's preprocessing lives in its DATASET code, not its model code (the `lance-mlx` x2t lesson)

A research repo's inference entry point (`inference_*.sh` → CLI args → dataset/transform
classes) often applies its OWN image/video preprocessing — resolution presets, aspect-ratio
bucket crops, custom normalization — that the HF `AutoProcessor` for the same base model
does NOT reproduce. Lance's x2t used `BucketResize` (AR-bucket center-crop) +
`DivisibleCrop(28)` from `data/transforms.py`, selected by a `RESOLUTION=image_768res`
shell flag; every MLX port assumed HF smart-resize and systematically misread chart values
("43" instead of "29%", stable across 14 runs and three independent implementations).

Protocol: **decode the actual release invocation FIRST** — the shell script names the
flags; the dataset class maps presets to pixel ops; an hour of reading beats days of
activation bisects. Then **vendor the reference's transform files verbatim** (they're
usually pure torch/torchvision/PIL) and byte-gate your preprocessing against them
(`max|diff| == 0.0`), exactly like the resampler byte-gate. The same script also reveals
non-preprocessing surprises: attention-mode flags, position-embedding toggles, logit
masking (`pred_logits[:, len(tokenizer):] = -inf`), connector branches keyed on `vit_type`.

## 23. Vendor showcase cases are the worst possible parity gate

Repo README/assets demo cases are typically near-memorized (model output ≈ dataset GT
verbatim) and sit on greedy decision boundaries — single-token knife-edges where any
backend's noise flips the trajectory. Six such cases gated Lance for ~20 runs of
root-causing; the systematic defects they exposed were real (worth it), but exact-match
on the residual was unachievable noise-chasing. Gate on N≥50 semantic samples (benchmark
subsets); treat showcase cases as smoke tests only. Full doctrine + red-flag checklist:
`parity-testing.md` § "The exact-match ceiling".

## 24. Judge parity against the reference you PORTED, run live — not a further-upstream capture

When a port chain is A (PyTorch) → B (Python MLX) → C (Swift MLX), C's gate is B, and
B must be RUN LIVE on the failing cases before any C-side debugging. The Lance Swift port
spent runs chasing two "failures" that B reproduced byte-identically — they were B's own
ceiling vs A, invisible because the oracle fixtures were captured from A. A reference's
own gap reads as your bug if you gate against its grandparent. (Cost: a 9-second script
run, once someone thought to do it.)

## 25. Hours-long inference runs die by silent SIGKILL — two stacked causes, fix both (the `scail-2-mlx` lesson)

Long denoise loops (40 steps × minutes/step on a 14B DiT) kept dying mid-run with no
traceback, no log error, and no MLX exception. There were TWO independent killers, and
fixing only the first produced a false all-clear:

1. **Metal buffer-cache ratchet.** `mx.eval` per step bounds the lazy graph, but freed
   step workspace stays in MLX's Metal buffer cache, so RSS climbs monotonically across
   the loop until macOS memory pressure SIGKILLs the largest process (swap near
   exhaustion is the tell). Fix: `mx.clear_cache()` after each step's `mx.eval`, plus
   `mx.set_cache_limit(8 * 1024**3)`. Instrument every step with
   `mx.get_active_memory()/get_peak_memory()` in the log — a healthy run is FLAT
   (scail-2-mlx: 34.3 GB active / 47.2 GB peak, unchanged from step 1 to 40).

2. **Harness-tracked tasks are mortal.** Runs launched as agent-harness background tasks
   (and their Monitor watchdogs) die when the task panel is stopped or cleared — stale
   "Running" rows are indistinguishable from live ones, and one panel cleanup killed a
   benchmark at step 4 with memory perfectly flat. A run that was accidentally
   double-backgrounded (`&` inside the wrapper) detached from the harness process group
   and survived 2h38m untouched. Fix: launch any >10-minute GPU job as
   `nohup bash -c '...' > out.log 2>&1 & disown`; register at most an expendable
   notify-only Monitor with the harness.

Diagnostic discipline: a kill with RISING memory + swap is cause 1; a kill with FLAT
memory and no system log entry is cause 2 (or some other external reaper). Companion to
pitfall #20 (training runs need an external terminal) — the same principle extended to
inference, with the memory mechanism made explicit.

## 25. "Never terminates" on a SAMPLED loop — sweep-quantify before structural debugging (the Qwen3-TTS E1 lesson)

A runaway/no-EOS report against a sampling (non-greedy) autoregressive loop is a single draw
from a distribution, not a reproducible state. Qwen3-TTS x-vector cloning "never emitted EOS"
(47 GB runaway, P1, suspected broken stop logic) — an N=10 sweep with the exact same artifacts
showed **9/10 runs stopping naturally**; the EOS id, sampler protection, and prompt were all
correct. The defect was the missing BOUND (no text-proportional cap) and missing RETRY, not
the stop logic.

Protocol: before diffing stop conditions or conditioning code, run the failing config N≥10
with fresh RNG and count. ~1/N failures → stochastic class: fix = text-proportional cap
(`max(floor, textTokens × k)` frames) + retry-with-fresh-RNG on cap-hit (p → p^attempts) +
per-token `Task.isCancelled` so the caller can always stop it. 0/N or N/N → structural class:
now diff the code paths. (Cousin of the exact-match ceiling: single observations of stochastic
processes mislead in both directions.)

## Reading strategy

When reading PyTorch source before porting, open three files side-by-side:
1. The module file (`model.py`).
2. The config (`config.json` in the checkpoint).
3. The base class from the parent framework (diffusers / transformers).

The module defaults often mislead; the config is the oracle — but **only the *resolved* config, not the on-disk json**: the parent class hides additional defaults one level deeper (see trap #10 — rope scaling is the classic case). Instantiate the reference, then read `model.config` / the built submodules (`rotary_emb.attention_scaling`, etc.), not just the json.

## 26. The reference's internal fallbacks ARE the reference (capture true tensors, don't reconstruct)

Qwen-Image-Edit-2511's diffusers pipeline runs its Qwen2.5-VL text encoder
without `mm_token_type_ids`, so HF's refactored forward silently falls back to
PLAIN SEQUENTIAL positions (1D RoPE on all three mRoPE axes) — no vision grid.
A Swift port that "correctly" implements mRoPE grids diverges (encoder hidden
cosine 0.85) while every component checks out individually: pixels 1-LSB exact,
ViT cosine 1.0, position ids exact vs `get_rope_index`, rope tables exact,
q/k/v projections byte-exact, MLP exact.

The defect was found only by capturing the TRUE tensors entering the framework
op (monkeypatch `F.scaled_dot_product_attention`, save q/k/v/mask/is_causal):
true q/k matched a sequential-position reconstruction at max_abs 0.0 and the
mRoPE reconstruction at 90.2.

Rules:
- When component X is reused inside a different pipeline, gate against what the
  pipeline ACTUALLY executes, not what X's API or model card implies.
- Manually reconstructed "goldens" (applying the formula you believe the
  reference uses) can validate your own bug. Capture at the lowest boundary —
  the framework op — where there is nothing left to reconstruct.
- Symptom signature worth memorizing: text-token hidden states exact, vision
  (or any special-token) spans drifting ~0.1%/layer compounding — suspect the
  POSITION pathway first; for text tokens all mRoPE axes coincide, so position
  bugs are invisible outside the special spans. Massive-activation dims can
  hide a uniform error from per-layer cosines on text tokens.

---

## Folded from the publishing / converter lineage (skills consolidation 2026-06-15)

> Sections unique to the converter branch (renumbered to avoid colliding with #1–26 above).

## 26b. VAE numerics — black images, cyan textures, gray output

**What goes wrong:** End-to-end output is structurally there but visually broken — black frames, cyan-tinted textures, washed-out gray, low-contrast mush. DiT-level parity is green; VAE parity at small scale (`hidden=64`, random weights) is green; full-pipeline output is wrong. The reading-time traps (1–6) almost never produce *color tints* — they produce structural failure or checkerboards. A color symptom is its own signal.

**Past failures:**
- **Qwen-Image VAE:** Cyan tint on every output. Cause: `groupnorm_eps` defaulted to `1e-5` in the MLX port; upstream uses `1e-6`. The eps difference compounds across decoder stages and the activation distribution shifts into the chroma channels' codomain.
- **LTX-2 video VAE:** Black frames every 4 timesteps. Cause: `fused_norm=True` flag missing from `ModelArgs`, defaulted to `False`, so the post-attention residual ran without upstream's fused-norm path.
- **Hunyuan3D-2.1 texture UNet:** Gray output. Cause: `groupnorm_num_groups` computed from channels instead of read from config (upstream uses unconditional `32`).
- **Lance image VAE:** Black tiles every 8 pixels. Cause: `mx.eval(out)` missing on the decoder return path. The lazy tensor handed to `Image.fromarray(np.array(out))` materialized partial state and rendered as zeros in the affected regions.

**Idiomatic solution:**

1. **Parity-bisect from encoder forward.** Most VAE bugs surface at the *decoder* output but originate in the encoder's first norm or first conv. Run `tests/parity/test_vae_encoder.py` first, then `test_vae_bottleneck.py`, then `test_vae_decoder.py`. Whichever fails first is your culprit.
2. **`groupnorm_eps`, `fused_norm`, and `groupnorm_num_groups` are the three usual culprits.** All three must be `ModelArgs` fields with the **trained** default. Cross-check `config.json` (see pitfall #1).
3. **Wrap VAE decode in `mx.eval(out)` before any image conversion.** Lazy tensors handed to `np.array()` or PIL convert through whatever current numerical state exists — for an unmaterialized graph, that's zeros (renders black) or partial computation (renders cyan/gray noise). This is the same lazy-tensor failure mode as weight saving, but at inference time.

```python
img = vae.decode(latents)
mx.eval( img )                      # force materialization
img_np = np.array(img)              # now safe
Image.fromarray((img_np * 255).astype("uint8")).save("out.png")
```

**Rule:** When the symptom is a color-tinted or low-contrast image but layer parity passes at small scale, suspect a norm eps / fused-norm flag mismatch before suspecting attention or RoPE. Color tints almost never come from attention bugs.

## 27. AutoTokenizer hangs on multimodal local paths

**What goes wrong:** Quantization / calibration / DWQ scripts that need a tokenizer use `AutoTokenizer.from_pretrained(local_weights_dir)`. The call hangs indefinitely (no output, 100% CPU) with no useful error. Same path works fine for text-only models.

**Cause:** When `config.json` in the local dir declares `model_type=qwen2_5_vl` (or any multimodal model_type), `AutoTokenizer` resolves the *processor pipeline* class for that model_type. For multimodal models, the processor needs image-preprocessor configs (`preprocessor_config.json`, `processor_config.json`) that aren't in our minimal converted-weights directory. The resolver retries in some stuck state instead of erroring.

**Idiomatic solution:** Use `AutoProcessor.from_pretrained(hf_repo_id).tokenizer` instead — same path the production inference pipelines use. The HF repo has the full processor config; it's cached after first download.

```python
# ❌ Hangs forever on multimodal local paths
tokenizer = AutoTokenizer.from_pretrained("/path/to/local/qwen2_5_vl_weights")

# ✅ Always works (cached after first download)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = processor.tokenizer
```

**Tokenizer length surprises (Qwen BPE):** related tokenization gotcha — Qwen's BPE is more compact than typical English-trained tokenizers. Median typical-sentence token count ≈ 25, max ≈ 35. Calibration-set filters of `len(tokens) >= 32` that work for older tokenizers will reject ~95% of lines and infinite-loop typical corpora. Use `>= 8` and let downstream batchers pad.

## 28. Background `uv run` commands need explicit `cd`

**What goes wrong:** `Bash` tool's `run_in_background` (and similar harness-managed background commands) can have cwd reset to a default project root that is NOT your MLX port's repo. Without explicit `cd`, `uv run python script.py` resolves dependencies against the *wrong* `pyproject.toml`. With no installed `mlx` in that venv, uv attempts to install/sync — and can hang for hours at 100% CPU producing zero output.

**Symptom:** Background MLX command shows 100% CPU, ~600MB RAM, no log lines for tens of minutes. Looks identical to a Python import hang but is actually a `uv sync` deadlock against the wrong project.

**Idiomatic solution:** Always prepend `cd /absolute/path/to/your/repo &&` to backgrounded `uv run` / `python -m` commands in harness contexts. Foreground commands inherit the harness cwd correctly; background commands don't.

```bash
# ❌ Hangs at 100% CPU forever in some harness contexts
uv run python scripts/my_script.py

# ✅ Always works
cd /Volumes/.../my-mlx-repo && uv run python scripts/my_script.py
```

This is harness-specific (Claude Code, codex, etc.) and only bites on `run_in_background=true`. If you see CPU spinning with no output where you'd normally see progress, suspect this before suspecting model imports.

### Additional cross-lineage bullets (not in the local Other-subtler list)

- **MLX-Swift float64 → GPU crash.** Swift `Double` literals construct float64 `MLXArray`s; MLX-Swift has no float64 GPU path and crashes hard on the first GPU op. Wrap literals as `Float32(…)` before constructing arrays, cast `MLXArray.linspace` output via `.asType(.float32)`, and use `MLX.multiply` / `MLX.divide` for `Float * MLXArray` to disambiguate it from the `Duration * Duration` overload. Cast tensors handed in across public API boundaries before the first op.
- **bf16 in an autoregressive Metal diffusion / flow-matching loop garbles output.** Distinct from pitfall #7 root-cause #5 (scheduler scalar leaking fp32 *into* a bf16 latent): here the entire AR loop runs bf16 on Metal and accumulates audible glitches / garbled tokens over hundreds of steps. Pattern from OpenBMB/VoxCPM PR #263: load weights bf16 to save memory, promote AR-loop compute (LocDiT, CFM, codec head) to float32, keep AudioVAE float32. Symptom shows up in TTS LocDiT, AudioLM diffusion, and any RVQ-conditioned generator.
- **`bs_roformer >= 0.4` breaks classic Mel-Band-RoFormer checkpoints.** lucidrains/bs_roformer 0.4 added nGPT-style norm parameters (`static_alpha`, `dynamic_alpha_fn`, `pre_branch_scale`, `residual_scale`, `static_beta`) and reordered the `layers` ModuleList — produces a ~500-key state-dict mismatch against Kim Vocal 2 / viperx / ZFTurbo v1.0.0 weights. Pin `bs_roformer==0.3.10` in parity test envs.

## 29. Lazy-mmapped weights on slow/external storage trip the Metal command-buffer watchdog (the `lens-mlx` Turbo lesson)

MLX lazily mmaps weights, so the **first GPU forward pages the weights off disk inside the command buffer**. From fast internal storage this is invisible; from a slow/external SSD (a Thunderbolt archive volume, a network mount) the kernel stalls waiting on `pread` and the command buffer exceeds the ~10 s Metal watchdog → `[METAL] Command buffer execution failed: GPU Timeout (kIOGPUCommandBufferCallbackErrorTimeout)` → MLX `gpu::check_error` throws → uncaught → **SIGABRT**. The crash thread is `com.Metal.CompletionQueueDispatch`; worker threads sit in `mlx::core::io::ParallelFileReader::read → pread`.

**It's storage-speed × forward-size dependent, not a correctness bug.** Lens ran fine originally from internal disk; re-run from an external archive it crashed at 1024² (8192-token forward) but *succeeded* at 512² — the smaller command buffer finished under the watchdog. Same weights, same code.

**Rule (Metal-watchdog family):** never let the first forward be the thing that pages weights from disk. Force-materialize every component's params into unified memory right after load:
```python
from mlx.utils import tree_flatten
for comp in (transformer, vae, encoder.model):
    mx.eval([v for _, v in tree_flatten(comp.parameters())])
```
Now the forward command buffers are pure compute. Bonus: it also makes the cost visible and faster (Lens load 71 s → 4.6 s, since the read is one bulk eval, not fused per-kernel). Alternative: stage weights to fast local storage before running. (Swift side: the same lesson lives in `mlx-swift-integration`'s Metal-watchdog family — CPU-stream weight loads, never eval giant constant fills, ARC-scope big models.)

## 30. Every op bit-exact but the e2e is wrong → it's the INPUT transform, not the model (the `edgetam` video click lesson)

When per-op parity is all green (each layer ≤1e-3 vs goldens) but the end-to-end result is garbage, **stop debugging the model — the bug is in how you map the caller's input into model space**, almost always a coordinate or preprocessing convention. The model is proven correct by the op goldens; what isn't goldened is the glue that builds its inputs.

EdgeTAM video: every op (perceiver, mem-encoder, RoPE-2D mem-attention, tracking decode, memory-bank assembly) hit oracle precision first build, yet the e2e mask came out ~8× too large. The single bug: the click arrives in **source-video pixels** and must be normalized `point / [video_W, video_H]` *then* scaled `× image_size (1024)` before the prompt encoder (`add_new_points_or_box` does both steps; the image-mode path only did `coord/orig*1024` for a square resize, so the missing `/[W,H]` for non-square video was easy to drop). One `MLXArray` of scaled coords, not a layer.

Classic members of this class: the SAM `+0.5` pixel-center offset, point/box normalization, image resize mode (`align_corners`), ImageNet mean/std, BOS-token prepend (Tekken/Pixtral). They never show up in layer parity because layer tests inject already-correct tensors. **Diagnostic shortcut:** if ops pass and e2e fails, diff your input-construction against the reference's *predictor/processor* code (the `predict()` / `add_*` / `processor` wrapper), not its `forward`. That's where the coordinate math and normalization live — and it's 20 lines, not the model.

## 31. Reduced-precision (`manual_cast` bf16) layers must RETURN the compute dtype, not fp32 (the TRELLIS.2 assembly lesson)

Many diffusion/3D models run the heavy block loop in bf16 for memory (`compute_dtype = mx.bfloat16`; "manual_cast") while computing norms in fp32 for stability. The crucial detail: the oracle's norm modules (`LayerNorm32`, per-head RMSNorm) and the AdaLN modulation add compute in fp32 **then cast back to the input dtype** — `x = x.astype(fp32); …; return x.astype(x_dtype)`. A port that computes in fp32 and *returns fp32* is **invisibly correct in an all-fp32 parity test** (single layers, fp32 e2e all pass) and **silently wrong the instant the bf16 compute lever is added**: the oracle keeps bf16 between ops, your port stays fp32, and they diverge.

- **Replicate the cast-back exactly** (`out.astype(x.dtype)` at the end of every norm/modulation). No-op for fp32 inputs (so earlier fp32 gates stay green), required for bf16.
- **Gate the ASSEMBLED model, not just the block** — the cast lever is applied at assembly time; a per-block fp32 test won't exercise it.
- **Know which ops still upcast.** Even on the "bf16 path", rope (`q_bf16 * cos_fp32`), the fused-SDPA softmax (fp32 regardless of input), and the fp32 affine all promote to fp32 — so after the first residual the block is effectively fp32 anyway, and assembled bf16-vs-oracle parity stays ~1e-7 (NOT bf16-grade ~1e-2). If you see ~1e-2 drift, suspect a real missing cast, not "bf16 noise."
- **Diagnostic:** diff your norm's return against the reference `norm` line-by-line for the `.astype(x_dtype)` tail; it's the single most common silent bf16-port bug.

## 32. The SAMPLER is a first-class port surface — a placeholder integrator passes latent parity yet emits garbage at the model's real resolution (the Anima / Cosmos-Predict2 lesson)

A generative port can pass **every** layer + e2e gate and still produce pure garbage, because the piece that isn't a layer — the **sampler** (integrator + scheduler + model_sampling↔prediction pairing) — was approximated instead of ported. This is pitfall #11 ("guidance/timestep math lives in the pipeline") sharpened into its most expensive form. The Anima port (Cosmos-Predict2-2B anime T2I) shipped a **deterministic Euler placeholder** for what the model card documents as a **stochastic `er_sde` default**, and it went unnoticed for weeks.

- **Read the model card for the sampler AND the resolution range, and treat both as hard requirements.** Anima's `ARCHITECTURE.md`: *"Resolution 512²–1536², steps 30–50, CFG 4–5. Samplers: `er_sde` (default), euler_a, dpmpp_2m_sde_gpu."* The port used plain euler — the porting notes even flagged it as a deferred "Phase-B beauty" nicety. It is not a nicety: wrong sampler → **resolution-dependent** garbage.
- **The failure is resolution-gated, so single-resolution gates miss it entirely.** The port was validated only at 32×32 latent (256²) and was coherent there; at the model's actual base resolution 512² (64×64 latent) it collapsed to tiled/green garbage — in the **Python-MLX, the Swift-MLX, AND the team's own PyTorch oracle** (they all shared the placeholder sampler). **Always validate decoded-image coherence across the model's FULL documented resolution range**, not latent-cosine at one small size.
- **Latent-cosine parity against your OWN reconstruction is false confidence.** Every gate compared the port to the team's torch reconstruction of the sampler (which was itself wrong), so "Swift ≡ Python ≡ torch, cos 0.999" just meant *all three were wrong the same way*. Parity must terminate at a **real reference-runtime render** (ComfyUI / diffusers `__call__`), not a home-rolled oracle.
- **Community-runtime models carry model-SPECIFIC sampling classes — find the right one; don't reconstruct from diffusers.** ComfyUI ships `ModelSamplingCosmosRFlow` (timestep `σ/(σ+1)`, σ∈[0.002,120] via ContinuousEDM) specifically for Cosmos-Predict2; the team reverse-engineered the generic `ModelSamplingDiscreteFlow` (σ∈[0,1], timestep=σ). The `model_sampling ↔ prediction (CONST/EPS/EDM) ↔ sampler` triple is coupled: `sigma_to_half_log_snr` branches on `isinstance(model_sampling, CONST)`, `er_sde` needs matching `alpha_t`/`er_lambda`. Swapping one piece (e.g. the diffusers `c_in/c_skip/c_out` EDM preconditioning) onto the wrong schedule regresses the working case. Read the reference runtime's actual classes; a diffusers pipeline may use a *different-but-equivalent* preconditioning that won't transplant piecemeal.
- **Stochastic / higher-order samplers AMPLIFY the MLX-GPU-fp32 vs torch-CPU-fp32 divergence that euler tolerates.** Deterministic euler matched between MLX and torch (both coherent at 256²), but the SAME model under `er_sde` was coherent in torch-CPU-fp32 and **garbage in MLX** — the per-step DiT deltas (cos 0.999, GPU-fp32 accumulation) compound through the stochastic/multi-stage updates. Corollary: **parity-lock the sampler against the reference on the SAME backend/dtype**, and don't assume a sampler that's stable in the torch reference is stable once the compute moves to Metal.
- **Stochastic samplers + MLX RNG in a Python loop:** the per-step injected noise must be genuinely iid. `mx.random.normal` inside a lazy loop can yield correlated draws (structured/tiled artifacts) — drive it from a seeded `np.random.default_rng` (or explicit key splitting), same as the cross-binding RNG rule in the main SKILL.
- **Practical order of attack when a generative port is garbage but layers pass:** (1) reproduce via the fast CLI, not the app; (2) confirm it's *not* the layers by decoding the reference's OWN known-good latent through your VAE (isolates decode vs sample); (3) sweep resolution — coherent-small/garbage-large ⇒ sampler or positional, not a layer; (4) diff your sampler against the reference **runtime** (`__call__` + its `model_sampling`/`k_diffusion` classes), not a further-upstream or reconstructed one; (5) get a real reference render as the coherence oracle before porting the fix.

### 32b. Parity-testing a STOCHASTIC sampler — inject the reference's captured noise, or RNG masquerades as a port bug

When the sampler injects noise every step (`er_sde`, `euler_a`, DPM++ SDE, any ancestral/SDE solver), a naive port-vs-reference comparison is uninterpretable: `mx.random`/`np.random` ≠ `torch.randn`, so the two runs see **different noise** and diverge even if the code is a perfect transpose. The Anima er_sde port looked broken — coherent in the torch reference, garbage in MLX across every seed — and the instinct "it's a transposition bug" was **wrong**. The method that settled it in two runs:

1. **Bisect on `s_noise=0` first.** Run both sides fully deterministic (no injection). This gates the *integrator math* (the multi-stage update, the `er_lambda`/`alpha` scalars) with zero RNG. Anima: MLX-vs-torch deterministic er_sde matched to **cos 0.994** — same level as the euler port — so the solver math was correct and the bug was isolated to the stochastic term.
2. **Then capture the reference's exact per-step noise and inject it.** Dump `torch.randn(...)` for each step to `.npy`, load it on the MLX side, and feed it in place of the port's RNG. Now both sides are bit-for-bit comparable: Anima MLX er_sde + torch's noise → **cos 0.999** and the identical image. That single run proves the port correct and reassigns the "garbage" to the RNG draw, not the code.
3. **What's left after that is a MODEL property, not a port bug:** if the port is proven correct but *different valid iid noise draws* flip coherent↔garbage, the model/sampler is **fragile to the noise realization** (Anima does this — a "good" torch seed is coherent, several numpy seeds collapse). Don't keep hunting the port; that fragility travels with the checkpoint and is the reference's behavior too. Drive the port's per-step noise from a seeded `np.random.default_rng` (not `mx.random` in a lazy loop, §32) so at least it's reproducible, and note the seed-sensitivity.

**Takeaway added to the stochastic-sampler workflow:** `s_noise=0` deterministic gate → captured-noise-injection gate → only then judge output coherence. Skipping straight to "decode and eyeball" conflates three independent failure modes (solver math, RNG source, model fragility).

## 33. RoPE extrapolation / NTK ratio is a TRAINED config value — take it from the reference's config-DETECTION code, not a diffusers default (the Anima 512² lesson)

The single bug that made the Anima port (Cosmos-Predict2-2B) produce a **regular tile grid at its native resolution** while looking fine at half-resolution was **one wrong RoPE config triple**. The port set `rope_scale = (t2.0, h1.0, w1.0)` — the diffusers *video*-Cosmos default. But ComfyUI's `model_detection.py`, which is where the real per-checkpoint config is assembled, sets **image** Anima (`in_channels == 16`) to `rope_h/w/t_extrapolation_ratio = (h4.0, w4.0, t1.0)`. Wrong on all three axes; the h/w `4.0` is the killer.

- **What the ratio does.** These DiTs scale RoPE frequency by an NTK factor: `theta = 10000 · ratio**(dim/(dim-2))`. A ratio > 1 *lowers* the frequency (spreads positions), which is exactly the mechanism that lets the model run **above its base grid** — a 4× ratio ≈ "trained to extrapolate to 4× the base resolution." With the ratio left at 1.0, the RoPE **aliases** once the position count exceeds the base: fine at 256² (16×16 patch grid), regular-tiled garbage at native 512²+ (32×32 grid). The NTK formula is usually byte-identical between comfy and diffusers, so the fix is a one-line config change — but only if you use the right numbers.
- **Why the parity gate missed it (again #10, #32).** The DiT golden was generated at ONE small resolution (32×32) with the *same wrong ratio on both sides*, so it matched to cos 1.0. A frequency/extrapolation error is **invisible to a fixed-resolution parity gate** and only shows above the gate's size — you must gate the DiT at ≥2 resolutions, including the model's documented max.
- **Where to get the true value.** Not the diffusers `config.json` default, not the reference model's `__init__` default (Cosmos `MiniTrainDIT` defaults every extrapolation ratio to 1.0) — the value is injected by the runtime's **per-checkpoint detection** (`comfy/model_detection.py`, keyed on `in_channels`/`image_model`/tensor shapes). For community-runtime models, read that detection block; it encodes the trained resolution behavior that the bare architecture defaults omit. (Same lesson as #10 "resolved config, not the json," sharpened to positional-embedding frequency.)
- **Diagnostic shortcut.** Symptom *coherent-small + regular-tiled-garbage-large* ⇒ suspect **positional-embedding frequency** (RoPE theta / NTK / extrapolation ratio, or a learned pos-embed dropped in the port), not the sampler and not a layer. Confirm by rendering at 2–3 resolutions: a sampler bug degrades *everywhere*; a RoPE-frequency bug is **resolution-gated** with a periodic signature.
