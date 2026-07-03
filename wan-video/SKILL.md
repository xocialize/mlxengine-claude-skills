---
name: wan-video
description: The Wan video-model family on the shared `wan-core` Swift-MLX substrate — Bernini-R (Wan2.2-A14B + Wan2.1-1.3B), TI2V-5B (t2v+i2v, 720p, 48-ch vae22), Helios (autoregressive long video), Phantom (subject-to-video). Use for ANY Wan-family port or its tier/memory behavior — a wan-core consumer, the config-driven `WanModel` DiT, 16-ch WanVAE vs 48-ch channels-last `vae22`, umT5-XXL, FlowUniPC/DPM++, the decode-memory lever sequence (streaming decode → `Memory.cacheLimit` cap → sequential eviction → fp8 umT5 → halo-tiling), the fp32-DiT / NFKC-neg-prompt / `withDefaultDevice(.cpu)` quirks, the tier ladder + image-grounding axis, or Wan parity conventions. Triggers — wan-core, Wan2.1/2.2, WanModel, Bernini, TI2V-5B, vae22, Helios, Phantom-Wan, umT5, 16 GB tier, decode memory, cacheLimit OOM, streaming decode, fp32 DiT, i2v mask-blend, DPM++/Lightning. DOMAIN layer: delegates layer-translation/parity to `mlx-porting`, `ModelPackage`/C0–C13 conformance to `mlx-swift-integration`. Skip non-Wan.
---

# The Wan video family on `wan-core`

## What this skill owns (and what it delegates)

This is the **domain playbook** for the Wan family — the cross-consumer architecture, the tier ladder, the decode-memory levers, and the hard-won numerical quirks that **every** Wan consumer re-hits. It does NOT re-teach generic porting:

- **PyTorch/Python → MLX layer translation + parity** → `mlx-porting`.
- **Swift `Core` → `ModelPackage` + C0–C13 engine conformance** → `mlx-swift-integration`.
- This skill = the Wan-specific glue between them, plus the memory/tier discipline.

**The exhaustive detail lives in the repo + memory, not here** — this skill is the index + the load-bearing lessons. Authoritative sources: `WAN-STACK-PLAN.md` (execution), `ENHANCEMENTS.md` §0 (roadmap spine) + §1.x (models) + §2.x (perf/memory), `EngineeringDocs/ENH-*.md`, `APP-VALIDATION.md` (live runs), and the memory cluster (`wan-tier-ladder`, `wan-family-substrate-and-speed`, `bernini-family-tiers-radar`, `ti2v-5b-tier-and-t5-floor`, `ti2v-t2v-multiframe-divergence`, `lance-vae22-spatial-halo-tiling`, `helios-b1-active-port`).

## The 3-layer architecture

```
L1  wan-core-mlx-swift   neutral substrate — WanModel DiT · 16-ch WanVAE · 48-ch vae22 · umT5-XXL ·
                         RoPE · FlowUniPC/Euler/DPM++ · StreamingDecode(+22) · TextEncode · WeightLoader
L2  per-model packages   bernini-r-mlx-swift · ti2v-5b-mlx-swift · helios-mlx-swift · phantom-wan-mlx
                         (one model = one ModelPackage; modes are intra-model; consume wan-core by LOCAL PATH dep)
L3  MLXEngine            multi-package-per-capability: textToVideo backed by {bernini=quality, ti2v-5b=i2v,
                         helios=long, phantom=subject} at once. The "video stack" is EMERGENT.
```

**`WanModel` is a single dense, fully config-driven DiT.** Dense tiers (1.3B, TI2V-5B, Phantom) are `WanModel(config)` — **no new core architecture**. A14B's "MoE" is just two `WanModel` instances + a pipeline-side timestep-boundary switch (`dual_model:true`). Adding a Wan model = a `WanConfig` + weight conversion + wrapper wiring, not a new DiT.

## The family — ONE substrate, 2.1 vs 2.2 is NOT a fork

Wan2.1 and Wan2.2-classic share the same DiT blocks, the same umT5-XXL, and the same 16-ch VAE (stride 4/8/8). Pick models by **capability + license**, not version number.

| Model | Backbone | VAE | Grounding | Tier |
|-------|----------|-----|-----------|------|
| **Bernini-R-1.3B** (Wan2.1-1.3B) | dense, 30L/dim1536 | 16-ch | t2v/edit | consumer (~3.6 GB active) |
| **Phantom-1.3B** (Wan2.1) | dense | 16-ch | **subject-ref (S2V)** | consumer image-grounded |
| **VACE-1.3B** (Wan2.1) | dense + **Context Adapter** (15 vace_layers, vace_in_dim 96) | 16-ch | **all-in-one: first-frame i2v + ref + inpaint + control + v2v** | consumer (ACTIVE) |
| **Bernini-R-A14B** (Wan2.2) | dual-expert | 16-ch | t2v/edit (r2v/v2v/rv2v) | mid/quality |
| **TI2V-5B** (Wan2.2) | dense 5B, dim3072/30L, in/out 48 | **48-ch vae22** (stride 4/16/16) | **native t2v + true i2v**, 720p | mid/i2v |
| **Helios** (Wan2.2-A14B + AR delta) | dense 14B + AR machinery | 16-ch | long-form t2v | long video |
| **SCAIL-2** (Wan2.1-I2V-14B fork) | dense 14B + **CLIP-i2v + 3-seg ref/video/pose RoPE + dual-mask embeds** delta | 16-ch | **character animation** (ref image + driving video, end-to-end, no pose-rep) | mid/heavy (ACTIVE, `scail-2-mlx-swift`) |
| LongCat | its OWN 48-block DiT | 16-ch | episodic | separate port (not a Wan variant) |
| Wan2.2-Animate | WanModel | 48-ch (shared w/ TI2V) | character animation | different lane (new capability) |

## Tier ladder × image-grounding axis (see `wan-tier-ladder`)

| Tier | Model | Image-grounding | Memory recipe |
|------|-------|-----------------|---------------|
| **Consumer (16 GB)** | T2V-1.3B / **Phantom-1.3B** | none / **subject-ref (t2i→S2V)** | **light 16-ch WanVAE** (no halo-tiling) + fp8 umT5 + eviction |
| **Mid / quality** | TI2V-5B | **true i2v** (frozen frame-0, mask-blend) | int4 + **`cacheLimit` cap** + (halo-tiling only at ≥1024²) |
| **Pro (128 GB)** | TI2V-5B 720p / A14B | — / editing | int4 + `cacheLimit` |

**Image-grounding is FOUR distinct things** — none (base 1.3B, `in_dim==out_dim`) / subject-identity (Phantom S2V, refs as trailing temporal frames) / literal frozen-frame-0 (TI2V-5B i2v) / **all-in-one control (VACE-1.3B — first-frame + ref + inpaint + depth/pose + v2v via a Context Adapter branch; ✅ ACTIVE, the lead consumer image-grounding model, `vace-1.3b-active-port`)**. Base Wan2.1 has no plain I2V-1.3B; VACE is how you get first-frame i2v at the consumer tier.

## Decode-memory — the lever sequence (the most-repeated lesson)

The 720p decode peak decomposes into pools, each needing a different lever — apply IN ORDER:

1. **Temporal extent** → streaming decode (`decodeStreaming22` for vae22 / `StreamingDecode` for 16-ch). One latent chunk at a time, threads the causal cache, **flat in length, bit-identical**. The E11 first-chunk frame-0 skip uses the 3-state `Rep` cache (empty / Rep / cached) — chunk-0 marks the slot `Rep` (frame 0 bypassed, NOT cached) so chunk-1's `time_conv` zero-pads the fresh `rest`. **Frame-count caveat:** `T_out=(T_lat-1)*4+1` holds ONLY once every temporal-upsample stage sees T>1; the degenerate **T_lat=1 → 3 frames** (the inner stage doubles the lone frame, the outer bypasses), so the t2i/single-frame case is 3, not 1, and the streaming==whole-seq gate must match whatever whole-seq emits — not the formula. **The SAME lever applies to ENCODE** (`encodeStreaming`, the analog): `WanVAE.encode` already chunks the encoder forward but accumulates every chunk into ONE lazy graph → the full-res fp32 working set of the whole sequence is live at the caller's `eval`. Image-grounding models that VAE-encode condition frames (VACE's VCU build = TWO encodes, i2v init frames, v2v source) hit this **before the denoise loop**, in a path the per-block-eval fix and the denoise cap never cover (E15: VACE 4.8→**106 GB** + >20 min in one un-streamed encode). Stream it (eval each chunk + `featCache` before the next; bit-identical).
2. **Reclaimable buffer cache** → **`Memory.cacheLimit` cap during decode** (THE big one). Uncapped, freed full-res conv intermediates accumulate into the OS high-water (720p int4 5f: phys **110 → 41 GB**). A cap forces continuous reclamation; a `clearCache` can't (the peak is transient mid-chunk). **Default `DECODE_CACHE_MB=2048`** — measured best (same 41 GB phys, fastest wall, beats both cap=0 and uncapped). **Cap = stopgap, streaming = cure (key distinction, E15):** the cap bounds the *freed-buffer cache*, NOT the *live* working set — so a tight cap (2 GB) around a big single-shot graph (one un-streamed full-res encode, live ~13–41 GB) makes it **thrash** (continuous reclaim/realloc → glacial, >20 min). Streaming (#1) bounds the *live* set to one chunk; with that, a modest cap no longer thrashes. Apply BOTH, but streaming is the one that fixes runtime — reach for it whenever the cap drops memory but the phase stays slow. **Wrap EVERY full-res VAE pass** (encode and decode) in the cap, including pre-denoise ones — the denoise-loop cap doesn't reach them.
3. **Max-phase residency** → sequential eviction: umT5 paged per-request + evicted after encode (§2.4, `withTextEncoder`); DiT paged + evicted before decode (`pageDiT`). residentBytes = `max(phase)`, not sum (§2.12).
4. **Encoder floor** → fp8 umT5 (§2.11): fp32 22 GB → fp8 6.7 GB. Quantize the encoder first/most (produces conditioning once). The **umT5-XXL floor is shared across the whole family** — so even 1.3B isn't automatically 16 GB without it.
5. **Live high-res spatial floor** → spatial halo-tiling of the vae22 suffix (§2.13, `lance-vae22-spatial-halo-tiling`). **DEMOTED** — the cap solved 720p; this only bites at 1024²+ on small RAM.

**Governor basis = OS `phys_footprint` (task_info), NOT `Memory.peakMemory`** — under the cap the latter counts cumulative allocations (reads 76 while phys is 41) and misleads. Declare `QuantFootprint.residentBytes` from the **measured phys** at the production config.

## Family numerical quirks (the bit-exactness + correctness traps)

- **fp32 the DiT at video-scale seqLen.** Metal bf16 nondeterministically NaNs + over-grows the latent at large seqLen (≥1024). `ditDType: .float32` is the default; fp32 matches the oracle exactly. int4 weights + fp32 activations = fp32 compute (correct) at ~3 GB weights. (`ti2v-t2v-multiframe-divergence`)
- **NFKC the negative prompt.** wan-core `cleanText` must apply `text.precomposedStringWithCompatibilityMapping` (NFKC, replicates ftfy's fullwidth→halfwidth). Without it the Chinese `sample_neg_prompt` mis-tokenizes → wrong uncond → CFG amplifies garbage. **Affects every wan-core consumer.** No-op on ASCII.
- **Pin the CPU stream with `Device.withDefaultDevice(_:_:)` for VAE parity — NOT `Device.setDefault(device:)`** (a NO-OP: convs silently run on Metal, lose ~3-4 fp32 bits → a fake ~6e-3 "parity floor"). The bit-exact VAE gates depend on this.
- **vae22 is channels-LAST `[B,T,H,W,C]`** (V22-prefixed primitives, upsample conv dim→dim) vs the **16-ch WanVAE channels-FIRST `[B,C,T,H,W]`** (transposes internally, upsample dim→dim/2). They co-live in wan-core; don't cross the primitives.
- **`@ScalarOrArrayDouble`** decodes `sample_guide_scale` as scalar (single-expert: TI2V/1.3B) OR `[hi,lo]` (A14B) into `[Double]`. Indexing `[1]` on a scalar config crashed Bernini-1.3B.
- **TI2V i2v = MASK-BLEND, not the `WanModel y:` channel-concat.** `latents=(1-mask)*zImg+mask*noise`, per-token timesteps (`maskTokens*t`, frame-0 tokens clean), re-blend each step. `WanModel` supports `[B,L]` timesteps already.
- **A parallel/ControlNet branch MUST replicate `runBlocks`' per-block `eval` at large seqLen** — else it's one unbounded fp32 lazy graph. At `seqLen ≥ wanLargeSeq` (1024, now public) every block runs **fp32 SDPA**; `WanModel.runBlocks` already `eval`s after each main block to bound the graph, but a hand-written branch loop (VACE's `vaceHints`, any future adapter) that omits it builds ALL its blocks + accumulated residuals at once and materializes them in one shot. Symptom = **plateaued memory + glacial runtime** (VACE E15: ~107 GB / ~44 min @ 480p/17f). Fix: `if state.x.dim(1) >= wanLargeSeq { eval(c, hint) }` per block. **Diagnostic warning:** plateaued+glacial reads like a CPU-stream fallback but usually ISN'T — MLX arrays aren't device-pinned (the ambient stream at op-creation decides execution), so check the **eval discipline at large seqLen FIRST**, before suspecting stream placement.
- **Multi-frame temporal-RoPE application diverges ~0.35% from the mlx_video reference (latent substrate item, E16).** The RoPE *precompute* (`prepareRope`) is bit-exact and the *spatial* RoPE is correct (VACE single-frame 8×8 = 3e-5), but the **temporal-axis RoPE *application* at non-zero frame positions** drifts (max-abs ~0.0148 @ 4 frames, growing with frame count). Never caught because all prior *forward* gates were single-frame (temporal identity at pos 0) and multi-frame consumers were validated end-to-end. So **a multi-frame forward gate won't be bit-exact** — gate it at `< 0.02` for the functional pass and flag the quality follow-up (fix wan-core `ropeApply` temporal axis → re-verify ALL consumers). Phantom-Wan first exposed it.
- **The 16-ch WanVAE (8× spatial) hits the large-seqLen wall at LOWER resolution than vae22 (16×).** Same H×W → ~4× the tokens (480p/17f ≈ seqLen 7800 vs TI2V 720p/5f ≈ 1760). So 1.3B/16-ch tiers cross `wanLargeSeq` and the fp32-attention cost wall earlier than the heavier 5B/vae22 tier — size the frame count / resolution accordingly, and chunked-attention is the shared long-term lever.

## Speed levers (license is the gate)

- **DPM++(2M) @ 16 steps** ≈ 2.53× vs UniPC@40, quality-equivalent. Shipped as the `.fast` mode (`.quality` = UniPC/40). Free.
- **Lightning 4-step** (`lightx2v/Wan2.2-Lightning`, **Apache**) — per-expert distill LoRA merged into the A14B experts → ~20× fewer forwards. CFG-free, euler, shift 5.0, 4 steps = 2 hi + 2 lo. A different *checkpoint* (= config), not a request-mode.
- **Turbo / CausVid are NonCommercial** (CC-BY-NC) → **fail the engine's permissive C7 gate.** Research-only, not shippable.

## Parity conventions

- Oracle = `mlx-video` (pin the commit; e.g. vae22 `87db56a`, Helios PR #21 `27902e7`). Goldens under `/Volumes/DEV_ARCHIVE/.../parity` (CPU stream).
- Gate on the **CPU stream** (`withDefaultDevice(.cpu)`), `verify: [.all]` (VAE) / `[.noUnusedKeys]` (DiT) on `update(parameters:)`, max-abs **0.0** for the bit-exact components.
- Converted checkpoints are pre-MLX-sanitized (channels-last keys for vae22, named layers) → load directly. Build **debug** on the M5 box (stale Metal cryptex post-reboot blocks release; see `metal-toolchain-stale-mount-after-reboot`); run heavy GPU gates one at a time (`dev-machine-beta-os-metal-flakiness`).
- **HF-diffusers-sourced Wan models need converter renames to wan-core canonical** (mlx-video-named checkpoints load direct; diffusers ones do NOT). The re-ported converter maps `attn1/attn2.to_{q,k,v}/to_out.0`→`self_attn/cross_attn.{q,k,v,o}`, `ffn.net.0.proj`/`net.2`→`ffn.fc1`/`fc2`, `scale_shift_table`→`modulation`, `proj_out`→`head.head`, `norm_out.scale_shift_table`→`head.modulation`. **The subtle trap:** the affine cross-attn norm is **`norm2` in diffusers but `norm3` in wan-core** — the parameterless norms carry no checkpoint keys, so only this one affine norm needs the `norm2`→`norm3` rename. Gate S0 **bijectively** (every HF key → canonicalize lands in the generated set, 0 missing / 0 unused) against the real `index.json` before trusting — Helios's `HeliosWeightKeys.{ditKeys,canonicalize}` is the worked example (1101 keys = 21 global + 40×27).
- **The shared umT5-XXL + 16-ch WanVAE are bit-IDENTICAL across the family — VERIFY and MAP this during the initial porting analysis, not later.** A new consumer's VAE/T5 ship in whatever format that checkpoint used (original-Wan `.pth` → already canonical, only conv transpose; **HF-diffusers `AutoencoderKLWan` → needs a key rename too**). To confirm "shared" cheaply BEFORE writing any converter: load the candidate's tensors + a known-canonical sibling's (e.g. Bernini's `vae.safetensors`) and match by a **permutation-invariant value fingerprint** `(numel, sum, sumsq, min, max)` — invariant to transpose/reshape/rename. All-194-match ⇒ same weights (a fine-tune would shift the norm `gamma`s, so 0 same-shape-different-value is the tell), and you can then **derive the exact key map empirically** by pairing each tensor with its twin + detecting its transpose. Helios proved this: its diffusers VAE → canonical is **bit-exact (max_abs 0.0, 194 keys) vs Bernini's**. The diffusers→canonical VAE map (`HeliosVAEConverter`): `quant_conv`→`conv1` · `post_quant_conv`→`conv2` · `{enc,dec}.conv_in`→`conv1` · `conv_out`→`head.2` · `norm_out`→`head.0` · `mid_block.resnets.{0,1}`→`middle.{0,2}.residual` · `mid_block.attentions.0`→`middle.1` · resnet internals `norm1/conv1/norm2/conv2/conv_shortcut`→`residual.{0,2,3,6}`/`shortcut` · **encoder `down_blocks.N`→`downsamples.N` (already flat)** · **decoder `up_blocks.X`→`upsamples.M` FLATTENED (resnet Y→4X+Y, upsampler→4X+3)** · `resample.1.*`/`time_conv.*` verbatim · conv transpose 5D `(0,2,3,4,1)` / 4D `(0,2,3,1)` (no-op on `[C,1,1,1]` gammas). **Ship the consumer's OWN converted weights** (don't depend on a sibling's checkpoint at runtime) even though they're bit-equal — a `ModelPackage` must be self-contained.

## MLX-Python rung — already have one for the substrate (it's `mlx-video`)

`mlx-video` (`wan_2`) IS the MLX-Python reference for the wan-core **substrate** (WanModel / 16-ch & 48-ch VAE / umT5 / RoPE / schedulers). Ports that had it (Bernini-R, TI2V-5B) went smoothly; **VACE-1.3B was the hard one only because `mlx-video` has no `vace.py`** — the net-new Context-Adapter branch had no rung, so PyTorch→Swift fused the framework-port and language-port. Lesson: the gap is **net-new per-consumer branches, not the substrate**.

- **Don't rebuild the substrate in Python** — it duplicates `mlx-video` AND the parity-locked Swift core. Granular per-sub-op goldens (the `g_hint_00..14`/`g_vace_patch_embed` discipline) already localize a Swift break without a twin.
- **For a future LARGE + NOVEL net-new branch**, a throwaway MLX-Python scratchpad can de-risk the framework-port first. Candidate on the roadmap = **Animate** (largest/novel). **VACE-Fun-A14B and VACE-14B are reuse-shaped** (existing branch ⊗ existing dual-expert / same arch as 1.3B) → skip the rung.
- **Prefer contributing the missing branch upstream to `mlx-video`** over a private twin (gauge their receptiveness to that level of addition first); a private scratchpad is the fallback, kept throwaway in the measure/oracle dir.
- Full decision rule (generic): mlx-porting → "Deciding whether to insert an MLX-Python rung."

## Repos (all `xocialize`, consumed by LOCAL PATH dep in the MLXEngine workspace)

`wan-core-mlx-swift` (L1) · `bernini-r-mlx-swift` · `ti2v-5b-mlx-swift` · `helios-mlx-swift` · `vace-mlx-swift` · `scail-2-mlx-swift` (Wan2.1-I2V char-animation; CLIP-i2v delta SCAIL-local for now, Python oracle `DEV_ARCHIVE/scail-2-mlx` parity-locked) · `phantom-wan-mlx` (Python scaffold). A wan-core source edit is recompiled into every local-path consumer on their next build — no pin bump (no git-pinned dep). Net-new components land in wan-core so all consumers inherit.

## When to stop and ask

- A new candidate's `transformer*/config.json` — confirm single-vs-dual expert (`dual_model`/`skip_transformer_2`/`switch_dit_boundary`) before assuming the tier.
- A license that isn't clearly permissive (NonCommercial Turbo/CausVid) — flag the C7 wall before porting.
- Before declaring a tier admits — re-ground `residentBytes` on a **measured** phys at the production frame count, not a projection (frame-count scales the denoise activations = the chunked-attention wall).
