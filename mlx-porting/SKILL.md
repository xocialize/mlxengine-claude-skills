---
name: mlx-porting
description: Port PyTorch / CUDA models — LLMs, VLMs, TTS/STT, audio, diffusion, 3D — to Apple MLX or MLX-Swift for Apple-Silicon inference. Routes weight conversion through the official `mlx_lm.convert` / `mlx_vlm.convert` / `mlx_audio.convert` CLIs with mlx-community HF conventions (Tier 1), a manual `model_type` file when unsupported (Tier 2), or the `mlx-forge` recipe + `-mlx` fork only for multi-component diffusion/video/3D pipelines (Tier 3, may delegate to the `mlx-recipe` skill). Invoke whenever working on an MLX port — scaffolding a `-mlx` fork, translating attention / RoPE / VAE / norm layers, setting up PyTorch-vs-MLX parity tests, diagnosing wrong numerics (black images, cyan textures, gray output, garbage tokens, metallib-not-found, shape-safe silent failures), picking `mlx.fast` primitives, choosing quant scope, or publishing to mlx-community. Trigger phrasings — "port to MLX", "MLX port", "MLX-Swift port", "MLX-ify", "metallib not found", "publish to mlx-community", "mlx_lm.convert", "mlx-forge recipe", "-mlx fork", "MLX parity", "diffusers to MLX", and any mention of `mlx-arsenal`, `mlx.fast.*`, or converting diffusers/transformers code to MLX. Invoke eagerly — MLX ports fail silently (head-dim misnomer, QKV interleaving, lazy tensors saved as zeros, defaults diverging from config). Skip for general MLX API questions unrelated to porting, installing MLX, pure CUDA work, or abstract attention math.
---

# Porting PyTorch / CUDA to Apple MLX

## Mission

This skill captures the workflow and pitfalls accumulated across ~10 production MLX ports (LTX-2, CogVideoX-Fun, Matrix-Game, VOID, Fish S2 Pro, Mistral Small, Qwen Image, Hunyuan3D-2.1, Mel-RoFormer, Lance, Ming-omni, …). It is for **inference-only ports** on Apple Silicon — publishing to **mlx-community** as the default, falling back to the **mlx-forge recipe + `-mlx` fork** convention only for multi-component pipelines.

**Scope boundary — delegate:** for writing or updating an *mlx-forge conversion recipe specifically* (Tier 3), invoke the `mlx-recipe` skill — it owns recipe completeness (text encoders, tokenizers, schedulers, VAE). This skill owns everything else: routing, layer translation, attention patterns, parity testing, repo layout, debugging wrong numerics, publishing.

**Mostly inference, but some ports need training.** A from-paper reproduction (no released weights — an NR-IQA head, a LoRA fine-tune) means you port the architecture, then *train* it. The reading/translation/parity workflow still applies to the backbone (parity-lock it first), but two MLX-specific training gotchas bite: **grad-accumulation must `mx.eval` every micro-batch** or lazy graphs OOM (pitfall #19), and **long training runs can't live in the agent harness** — hand the user a copy-paste external-terminal command with `--resume`/per-epoch checkpoints (pitfall #20).

## Routing — three tiers (decide this BEFORE writing code)

Most ports land at Tier 1. The point of the routing rule is to stop you rebuilding what already exists.

**Tier 1 — official converters (default for every single-stack model):**

```bash
mlx_lm.convert    --hf-path <upstream> -q --upload-repo mlx-community/<name>-4bit   # LLM
mlx_vlm.convert   --hf-path <upstream> -q --upload-repo mlx-community/<name>-4bit   # VLM
python -m mlx_audio.convert --hf-path <upstream> --dtype bfloat16 \                 # Audio (bf16 default)
    --upload-repo mlx-community/<name>-bf16
```

Runtime: `from mlx_lm import load, generate` | `from mlx_vlm import load, generate` | `from mlx_audio.tts.utils import load_model`. Browser fallback: https://huggingface.co/spaces/mlx-community/mlx-my-repo.

**Tier 2 — manual architecture port (when `model_type` is not yet supported):** fork `ml-explore/mlx-lm`, `Blaizzy/mlx-vlm`, or `Blaizzy/mlx-audio`. Add one file (mlx-lm) or one sub-package (mlx-vlm/mlx-audio) named after upstream `config.json` `model_type`. Implement `@dataclass ModelArgs(BaseModelArgs)` + `class Model(nn.Module)` with `.model_type`, `.layers`, `.sanitize(weights)`. Re-run the official converter against your fork, parity-test, then upstream. Canonical reference port: Mel-RoFormer at `mlx-audio#654`. File shape → `references/manual-port-templates.md`; publishing contract → `references/mlx-community-conventions.md`. CONTRIBUTING: [mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/CONTRIBUTING.md) · [mlx-vlm](https://github.com/Blaizzy/mlx-vlm/blob/main/CONTRIBUTING.md).

**Tier 3 — fallback for multi-component pipelines:** when the model is **not** a single-stack LLM/VLM/audio architecture — T2V, T2I, 3D mesh, multi-component diffusion/audio-gen — use `mlx-forge` for per-component conversion with separate dtype/quant routing (LTX-2, Qwen-Image, Hunyuan3D-2.1, CogVideoX-Fun, Matrix-Game). **Delegate the recipe YAML itself to the `mlx-recipe` skill.** Diff against already-ported bases first (Step 1) — a fine-tune of a ported base often collapses to "reuse base + port the delta."

- `mlx-forge` (per-recipe conversion): https://github.com/dgrauet/mlx-forge
- `mlx-arsenal` (shared diffusion ops): https://github.com/dgrauet/mlx-arsenal

Full routing detail: `references/weight-conversion.md`.

## Helper modules to reach for first (Tier 1 / Tier 2)

For LLM/VLM/audio ports reach for `mlx_lm.models.base` and `mlx_lm.models.rope_utils` **before** importing `mlx.fast.*` directly or hand-rolling attention/mask/RoPE — they handle GQA, mask polarity, KV-cache offset bookkeeping, and the full `rope_scaling` dispatch correctly. `mlx-arsenal` covers only what the canonical helpers don't (flow-matching, diffusion tiling) and belongs at Tier 3. Per-helper breakdown is in Step 3; attention/RoPE patterns in `references/attention-patterns.md`.

## Core mental model

A port is a **transpose operation**, not a redesign. The reference implementation's config values, algorithmic choices, and numerical behavior are the oracle. Any deviation — "better" defaults, "cleaner" modes, "optimized" schedules — almost always hides a port bug. Match first, optimize later with justification tied to a framework constraint (e.g. Metal command-buffer timeout), not taste.

**Preserve isomorphic structure with the reference repo.** Hard rule, not aesthetics — `ltx-2-mlx` suffered repeated drift bugs that were costly to track because the MLX code no longer mapped 1:1 to the official source. Concretely:

- **Same file paths and module names** as upstream (`models/transformer.py` → `models/transformer.py`, not `model/dit.py`).
- **Same class names** (`LTXVideoTransformer3DModel`, not `DiT`).
- **Same method names and decomposition** — keep `_split_qkv` + `_apply_rotary` separate if upstream does. Don't inline, merge, or "pythonize".
- **Same forward-pass call order** + intermediate variable names. A reader should diff `model.py` vs `model_mlx.py` and see only PyTorch↔MLX op substitutions.
- **Same config / kwargs surface.** No silent renaming (`num_attention_heads`, not `n_heads`), no dropping "unused" fields, no convenience flags.
- **Resist refactor temptation.** Even upstream dead code / odd naming — keep it. Refactoring during a port is paid twice: now (breaks parity diffing) and forever after (every upstream update becomes a 3-way merge).

Refactor only AFTER fp16 parity is locked, end-to-end output matches, and there's a concrete reason (perf, framework constraint). If output looks almost-right but subtly off (color tint, slight blur, minor drift), assume port bug, not artifact — layer-level numerical parity is the only reliable signal.

## Workflow

```
1. Read reference        → exact shapes, configs, math; diff against ported bases first
2. Route conversion      → Tier 1 (mlx_*.convert) | Tier 2 (manual model file) | Tier 3 (mlx-forge recipe)
3. Scaffold -mlx fork    → only if Tier 2 or Tier 3; Tier 1 needs no fork
4. Translate modules     → layer by layer; reuse mlx_lm.models.base helpers first
5. Parity test           → PT ref vs MLX, layer by layer, max_abs < 1e-3 fp16, CPU stream
6. End-to-end            → full pipeline, golden image/text
7. Quantize              → int8/int4 only after fp16 parity locked
8. Publish               → weights → mlx-community; code → xocialize; group quants in a Collection
```

Never skip step 5. Wrong output at step 6 with no layer-level parity data is unrecoverable.

## Step 1 — Read the reference carefully

Before writing any MLX code, read the PyTorch source skeptically for these traps. Full detail + past-failure examples in `references/common-pitfalls.md`.

- [ ] **Diff against already-ported bases FIRST** — before scoping a Tier-3 port, diff the candidate's weight-key set (from the safetensors `*.index.json` — no download) and `config.json` against bases already in MLX (Wan, LTX, Lance, Qwen, SD3/FLUX). A fine-tune/wrapper of a ported base collapses to "reuse the base + port only the delta." Config knobs that name no new tensors are runtime params, not layers. This turned `bernini-r-mlx` from weeks to days. **Every format's keys are readable without downloading:** sharded → `*.index.json`; single-file safetensors → first 8 bytes are the header length, then range-request the header; **torch `.pth` → it is a ZIP, so read the (ZIP64) central directory, pull `data.pkl` alone, and walk it with `pickletools.genops` — 130 KB of a 33 GB file, nothing unpickled.** 🚨 **But diff against an upstream ORIGINAL and carry a control arm on a checkpoint whose keys you already know: a QUANTIZED file's shapes are PACKED** (MLX 8-bit = logical ÷ 4), and the packed number is always plausible, so it yields a confident wrong architecture — that is how a nonexistent config field got written for SeedVR2's 7B. (#12/#13, #48)
- [ ] 🚨 **Check whether your REFERENCE implementation has the same scope as your TARGET** — a port can take the *model* from one upstream and the *implementation* from another, and the second one's shortcuts travel silently. `mlx-seedvr2-swift` ports ByteDance's **video** SR model but traced its MLX code against **mflux, which is image-only**; mflux gates the causal `remove_head` on `T == 1` (its only case), so the decoder emitted 4×latT frames at T>1 — correct at T=1, silently wrong beyond it, and invisible for a year because the driver only ever asked for T=1 either. Record both provenances, grep the reference for guards on the dimension you intend to exercise (`if T == 1`, `B == 1`, `frames=1`), and assert a round trip on that dimension even before your driver uses it. When generalising an inherited shortcut, keep the old expression verbatim on the old branch so the shipping path is bit-identical by construction. (#49)
- [ ] **Constructor defaults** — every `__init__` default verified against `config.json` (`include_pi=True` when config says `false`, `groupnorm_eps=1e-5` when config says `1e-6`) silently ruins outputs.
- [ ] **`attention_head_dim` misnomer** — in diffusers UNets `attention_head_dim=[5,10,20,20]` means `num_heads`, NOT per-head dim. Real head_dim = `channels // attention_head_dim`.
- [ ] **QKV reshape pattern** — `qkv = cat([q,k,v]) → view(B,N,heads,3*hd) → split` means heads are **interleaved**, not stacked. Replicate EXACTLY.
- [ ] **Weight layout** — PyTorch Conv `(O, I, *K)`, MLX `(O, *K, I)`. Linear/Embedding identical. Conv transpose has its own rule.
- [ ] **Normalization semantics** — RMSNorm/GroupNorm/LayerNorm differ in default epsilon across frameworks; AdaLN variants differ (additive vs `x*(1+scale)+shift`). For VAE ports `groupnorm_eps` is the #1 silent killer.
- [ ] **Non-obvious flags** — `qk_norm`, `pre_norm`, `use_bias`, `cross_attention_dim`, activation (GEGLU/SwiGLU/GELU). Cross-check config, not defaults.
- [ ] **Resolved config, not the json** — `config.json` omits fields the parent config *class* injects (rope scaling/YaRN, sliding windows, norm flags). GPT-OSS YaRN rope (`attention_scaling` ≈ 1.347) isn't serialized → plain rope → uniform divergence (cosine ~0.94, looks like a quant gap). Compare the **resolved** rope on both sides. (#10)
- [ ] **Guidance/timestep math lives in the pipeline** — CFG is often not vanilla (Lens norm-rescales by `‖cond‖/‖comb‖`); timestep scaling can be split across modules. Port the reference `__call__` verbatim. (#11)
- [ ] **The SAMPLER is a first-class port surface, not a "beauty" afterthought** — read the model card for the *default sampler* AND the *resolution range*; a placeholder integrator (deterministic euler for a documented stochastic `er_sde`) passes latent-cosine parity yet emits **resolution-dependent** garbage (coherent at the one small size you gated, garbage at the model's real base resolution). Port the reference runtime's actual `model_sampling ↔ prediction ↔ sampler` triple (ComfyUI has model-specific classes, e.g. `ModelSamplingCosmosRFlow`); don't reconstruct it from diffusers piecemeal. Validate decoded-image coherence across the FULL documented resolution range against a **real reference render** — an eyeball gate without a reference beside it passes "prompt-correlated" garbage (the Anima port shipped exactly that as "validated"). Stochastic samplers are gated by `s_noise=0` + captured-noise injection (#32b), and are NOT inherently fragile on MLX — do not shelve the reference sampler on that theory. Also port the reference's **prompt defaults**: an empty negative prompt collapses CFG-trained models at large sizes where the workflow's quality-tag negative is load-bearing. Read the reference workflow JSON (latent size, sampler, scheduler, cfg, shift) before theorizing. (#32, #36)
- [ ] **Tekken / Pixtral tokenizer skips BOS** — `add_special_tokens=True` is a no-op; prepend BOS manually (Mistral Small 3 / Ministral3 / Pixtral / ERNIE-Image).
- [ ] **Checkerboard trap** — periodic noise at stride 2/4/8/16 ← (in order) `mx.tile` where `mx.repeat` was needed, pixel-shuffle axis order, text-encoder `hidden_states[-2]` applying N not N-1 layers, or scheduler dtype leaking fp32 into a bf16 DiT. Run the pitfall #7 3-test diagnostic BEFORE shipping.
- [ ] **VAE numerics (color tints, black/gray)** — layer parity passes but e2e is cyan/gray/washed-out → suspect `groupnorm_eps`/`fused_norm`/`groupnorm_num_groups` before attention. Wrap `vae.decode(...)` in `mx.eval`. (#26b)
- [ ] **Structural drift** — port the upstream pipeline's *abstractions* (Stage classes, ModalitySpec, denoising loops, guided-denoiser factories), not just tensor ops. (#9, 5 concrete patterns)
- [ ] **API surface ≠ capability** — state-threading params (`feat_cache`, `past_kv`, hooks) are documentation, not contract. Check whether the top-level entry actually allocates+threads the state or inner blocks accept it but the orchestrator never wires it. (#14)
- [ ] **Extend before fork** — to add capability to an upstream module, check if its submodules expose enough public surface for a free-function bypass at the consumer level. Lance's `vae_stream.py` = 426 LOC of streaming VAE decode, zero upstream edits, no fork. (#15)
- [ ] **Audit the target's top-level entry point before estimating cross-port effort** — read B's actual `decode`/`forward`/`generate`, not just inner blocks. If B was ported from a reference that already had the capability, B inherited it for free. (#18)
- [ ] **An unshippable HOST dependency (ffmpeg, a CUDA ext, a native codec) is usually a narrow SEAM, not an architecture** — never scope from the README's dependency list. Find the interface it's consumed through and count what crosses it; Mage-VL's entire "codec-native" surface was **4 arrays per frame** feeding ~110 lines of numpy. Then ask what those signals are *proxies* for (bit-allocation/saliency/energy ≈ optical flow + residual), whether the reference already exposes an alternative-input path, whether the signal is consumed **upstream of the weights** (preprocessing seams are replaceable without touching a weight), and whether your own fleet already ships a provider of that signal — #12/#13's reflex applied to *auxiliary* components, not just backbones. **Then #43b: a substituted signal the model was TRAINED on is a distribution shift, not an upgrade — gate on agreement (selection IoU) with the original signal, and run the unshippable dep ONCE offline to bake those fixtures. An oracle is not a dependency.** And **#43c: never choose a backend from a regime that cannot discriminate between backends** — a one-input A/B gave a clean, confident, *inverted* answer; enumerate the regimes the component's quality varies over, cover them, report per-item not just the mean, hold everything but the regime fixed, and carry a trivial baseline (a tie is evidence your sample can't discriminate, not that the options are equivalent). (#43, #43b, #43c)

If any apply, open `references/common-pitfalls.md` + `references/attention-patterns.md` before writing MLX code.

## Step 2 — Scaffold (Tier 2 or Tier 3 only)

**Tier 1 needs no fork** — run `mlx_*.convert` and publish.

**Tier 2 — manual architecture port:** fork `mlx-lm`/`mlx-vlm`/`mlx-audio`, add one file/sub-package named after upstream `model_type`. Canonical `Model`/`ModelArgs`/`sanitize` template → `references/manual-port-templates.md`.

**Tier 3 — multi-component `-mlx` fork** (full layout in `references/repo-layout.md`):

```
<model>-mlx/
├── README.md                  # Quick Start + HF repo link
├── pyproject.toml             # depends on mlx, mlx-arsenal, optionally mlx-forge
├── <pkg>/{model/, pipeline_mlx.py (from_pretrained), utils/weights.py}
└── tests/{parity/ (PT optional dep), smoke/}
```

The Swift consumer side under `xocialize/<package>-mlx` mirrors the Python repo with an Xcode workspace + `Package.swift` (Xcode is the build tool; SwiftPM CLI is not used). **HF auto-download:** single `ModelClass.from_pretrained(repo_id)` downloading split safetensors lazily; support `<MODEL>_MLX_WEIGHTS_DIR` env override.

## Step 3 — Translate modules

Order of preference when reaching for a helper:

**1. `mlx_lm.models.base` + `mlx_lm.models.rope_utils`** (Tier-2 canonical helpers — check first): `scaled_dot_product_attention` (GQA + mask polarity + `mx.fast` dispatch), `create_attention_mask` (KV-cache offset), `initialize_rope` (default/traditional/linear/llama3/yarn/longrope via `rope_scaling` — where almost every long-context scaling bug is fixed).

**2. `mlx.fast` primitives** (when base doesn't cover it — faster + more stable than hand-rolled): `mx.fast.scaled_dot_product_attention` (GQA native), `mx.fast.rope`, `mx.fast.rms_norm`, `mx.fast.layer_norm`.

**3. `mlx-arsenal`** (ops outside the LLM/VLM/audio canon — Tier 3): `mlx_arsenal.diffusion` (`get_timestep_embedding`, `euler_step`, `classifier_free_guidance`, `FlowMatchEulerDiscreteScheduler`, …), `.spatial` (`interpolate_nearest`, `pixel_shuffle`, `patchify`/`unpatchify`, `PatchEmbed2d/3d`), `.attention`, `.norm`, `.encoding`, `.moe`, `.tiling`, `.layout`, `.rasterize`. **Warning:** root `__init__.py` only re-exports a subset — import directly from submodules.

**Not in arsenal / `mx.fast` — hand-roll from `references/spatial-and-rope-ops.md`:** bilinear `grid_sample` (warp), bilinear `interpolate`, **3D-RoPE** (video ViTs), 3D-tubelet Conv3d patch embed. Each has a parity-verified NHWC recipe (`mlx_arsenal.spatial` covers only nearest/pixel-shuffle/PatchEmbed; `mx.fast.rope` is 1-D only).

**Don't extract prematurely.** Conceptual similarity ("all 3 have AdaLN") rarely equals literal compatibility — verify byte-identical via diff + seeded parity before extracting to `mlx-arsenal`. AdaLN "duplicated" across LTX/Hunyuan/Matrix was three incompatible variants.

## Step 4 — Weight conversion

Conversion follows the tier from Step 2. **Most ports never write a recipe:** if mlx-community has the repo, `load()` it; if the `model_type` is supported, run `mlx_*.convert`. For a Tier-2 manual port the converter calls `Model.sanitize(weights)` automatically — that is where **all** key remapping + conv-layout transposes belong (never the constructor, never first forward). **For Tier 3, delegate the recipe to the `mlx-recipe` skill.** Canonical invocations + quant-suffix grammar → `references/weight-conversion.md`; `sanitize` shape → `references/manual-port-templates.md`.

The **one thing you must remember** when conversion goes through anything *other* than `mlx_lm.convert` (Tier 3 or a manual script):

> **Materialize every tensor before saving.** MLX is lazy — unevaluated tensors serialize to safetensors as zeros with no error. Call `mx.eval(weight)` (or the `_materialize` helper in `mlx-forge/.../quantize.py`) right before `mx.save_safetensors`. The silent killer of conversion recipes. **`mlx_*.convert` does this internally** — Tier-1 ports never hit it.

Other conventions: split safetensors per component (transformer/vae/text_encoder) for independent load/quant; quantize only transformer-block `Linear.weight` by default, keep VAE/vocoder/connectors/ViT at fp16/bf16 (`mlx_vlm.convert` enforces ViT exclusion); PyTorch Conv `(O,I,*K)` → MLX `(O,*K,I)` inside `sanitize` (Tier 2) or `transform` (Tier 3, `mlx_forge.transpose.transpose_conv`). Publishing conventions → `references/mlx-community-conventions.md`.

## Step 5 — Parity testing (the step everyone skips)

**Do not skip.** When e2e output is wrong, only layer-level parity tells you which module broke. Template in `references/parity-testing.md`:

```python
def test_attention_parity():
    x_np = np.random.randn(2, 128, 1024).astype("float32")
    pt_out = pt_attention(torch.from_numpy(x_np)).detach().numpy()
    mx_out = np.array(mx_attention(mx.array(x_np)))
    assert np.max(np.abs(pt_out - mx_out)) < 1e-3
```

**Thresholds (fp16):** single layer `<1e-4` ideal / `<1e-3` ok · block `<5e-3` · full UNet/DiT `<1e-2` · anything `>1e-1` is a bug.

**Pin MLX to the CPU stream** (`mx.set_default_device(mx.cpu)`): Apple-GPU fp32 matmul accumulates ~8e-4 relative error per op, compounding over deep stacks (32 ViT blocks → worst-token cosine 0.886 vs 1.000000 on CPU). GPU noise at that level both masks real op bugs and gets mistaken for them.

**If parity fails:** isolate — run smaller sub-modules until the divergent layer appears, then diff the forward pass line-by-line. Treat PyTorch as an **optional dev dep** (`pip install -e ".[parity]"`).

**If parity PASSES but output is wrong: suspect the oracle.** A self-written torch oracle can share your misreading (same author, same source), and port-vs-oracle parity then certifies the wrong computation to 1e-4. For community-runtime models, budget one **reference-runtime render + tensor dump** (e.g. headless ComfyUI driven by workflow JSON) as the independent ground truth — one run both proves whose bug it is and localizes the divergent stream. (#39, the Anima TE-tap lesson.)

**Small-scale parity is necessary but not sufficient.** `hidden=256, num_layers=2` + random weights miss bugs that only trigger at production scale (bf16 over 36 layers, real RoPE positions, trained conditioning magnitudes). Add an **e2e noise-path smoke test** — decode random Gaussian through the full post-DiT chain (BN inverse → unpatch → VAE.decode); any periodic pattern means a spatial op is broken at the production config regardless of layer parity. See pitfall #7.

## Step 6 — End-to-end validation

Only after layer parity is green.

- Full pipeline on a golden input (same seed/prompt/image as the PT ref); compare PSNR/BLEU/mesh-delta.
- Log peak memory (`mx.metal.get_peak_memory()`) + per-step wallclock.
- Wrong output but every layer passed → suspect sampler/scheduler, RNG semantics (`mx.random.normal` is NOT seed-compatible with `torch.randn`), or preprocessing.
- **Gate at the largest PRODUCTION grid, on decoded output.** Small-grid latent cosines validate nothing above them: position machinery (RoPE extrapolation, pos-embed interpolation, grid indexing) only fails at the big grid. Anima passed every 256² gate and collapsed at its own base 512²; qwen3vl was cos 0.998 at the 576-token vision grid and 0.84 at 1024 tokens — "not catastrophic" until in-app edits came out glitch-banded. A cosine that sags monotonically with grid size is a structural bug, not noise; every shipping resolution tier needs one decoded-output eyeball. → `references/parity-testing.md` (largest production grid).
- **If the model's thesis is a SELECTION/IMPORTANCE mechanism, ablate it against an ARBITRARY control** — dense vs mechanism vs random, and print how much of the budget the control actually varies (an always-kept component can make the comparison vacuous). Mage-VL's "codec-derived importance" turned out to be carried by the full-frame ANCHOR, while motion-ranking scored *worse than random* — motion priors starve static regions, and scoreboards/captions/UI are static. Often the port gets simpler. (#46)
- **VLM prompts: gate the TEXT stream and the FINAL position, never prompt-wide argmax agreement.** An image-heavy prompt is mostly image tokens (Mage-VL: 2048 of 2075) whose next-token logits are off-manifold and unstable — prompt-wide argmax read **84%** on a port that produced **48/48 identical tokens**. And before chasing any bad cross-framework gap, run the **bf16-vs-fp32 self-control**: if your own dtype change disagrees with itself more than you disagree with the reference at matched dtype, it's precision and the port is clean. (#44)
- **Greedy AR text: do NOT gate on token-exact equality with a capture from different hardware.** Once weights/config/preprocessing/positions are exact and per-op parity ≤1e-5 on the CPU stream, residual answer flips on knife-edge inputs are backend accumulation noise amplified by autoregression. Distinguish **systematic** failures (same wrong answer every run/backend → real bug, usually config/preprocessing) from **knife-edge** flips (vary across configs/precisions → statistical gate). Doctrine + oracle-capture protocol → `references/parity-testing.md` (the exact-match ceiling).

## Step 7 — Quantize

Lock fp16 parity first.

> **⚠ Run the quantized FORWARD on the GPU stream — never the CPU stream.** Quantized matmul
> kernels are Metal-only; a quantized forward under a CPU pin (`mx.set_default_device(mx.cpu)` /
> Swift `Device.setDefault(.cpu)`) has no efficient path and **silently grinds — process state `R`,
> ~100% CPU, zero output, for HOURS** (a Z-Image quant gate spun this way for 10 h before it was
> killed; Helios hit the same at 100 min). It does NOT error or trip the watchdog — it just looks
> like a hang. **Weight LOAD + `nn.quantize`/`applyQuantization` can be CPU-stream; the forward must
> run on GPU.** The parity-gate cosine still holds there — GPU fp32 noise ~1e-3 is negligible against
> int4 error at a ≥0.99 gate. Full failure signature + triage: `references/parity-testing.md`
> (Quantized generative models) and, for the Swift lane, `mlx-swift-integration`
> `swift-port-parity.md` (Metal-watchdog family, item 2).

- `nn.quantize(model, group_size=64, bits=4)` standard 4-bit Linears; `group_size=128, bits=8` higher-quality fallback. MLX 0.30+: NVFP4 / MXFP8 / 3,5,6-bit QMV (`references/mlx-docs.md`).
- **Verify at the weight level** (per-pass cosine on identical injected inputs: int4 ≈ 0.99+, int8 ≈ 0.9999+) — NOT PSNR-vs-fp32-golden-image. Quantization perturbs the denoise trajectory into a *different but equally valid* image (Lens int4 e2e PSNR 15.6 dB despite sharp output). Gate on per-pass cosine + image-validity + a visual sample. → `references/parity-testing.md` (Quantized generative models).
- Scope the quant: skipping small precision-sensitive projections (in/out embeds, time embed, final norm) via a `keep_hi_precision` predicate lifted Lens int4 cosine 0.9944 → 0.9976 at the same size.

## Step 8 — Publish (two targets, never one)

A finished port has **two** homes — sending code to the weights host or weights to the code host is the recurring mistake:

- **Weights → Hugging Face `mlx-community`.** Single-stack LLM/VLM/audio (Tier 1/2) → `mlx-community/<UpstreamRepoName>-<quant>` using the quant-suffix grammar (`-4bit`/`-bf16`/`-mxfp4`/… — full grammar in `references/mlx-community-conventions.md`), preserving upstream casing. `--upload-repo` does it; hand-uploads (bf16 audio, learned quants) match the same naming. **HF weight-repo naming never carries an `-mlx` suffix** (that's for GitHub code repos); applies to WIP personal-namespace repos too, so graduation to mlx-community is a move, not a rename. Tier-3 pipelines with no clean single-`model_type` slot host split weights in the `-mlx` repo (documented exception).
- **Code → GitHub `xocialize`.** Python `-mlx` port + its Swift consumer both under `github.com/xocialize/<package>-mlx`. The Swift package references the `mlx-community` weights via `from_pretrained`; it doesn't re-host weights unless Tier-3.

**Group quant variants in a Collection.** When a model ships >1 quant, gather them into an `mlx-community` HF Collection so they're one discoverable family. Mechanics in `references/mlx-community-conventions.md` ("Quant collections").

**Pick the publish dtype by parity at that dtype, then validate the PUBLISHED artifact end-to-end.** fp32 parity passing does not license `-fp16`: high-magnitude-activation nets (FFT/FFC) collapse under fp16 and need `-bf16`, while well-scaled nets prefer fp16 — re-run the gate at each candidate dtype (`references/parity-testing.md` "Choose the publish dtype"). And make the last publish step a fresh `hf_hub_download` → run on a real input → eyeball the output: a broken dtype-rounded upload builds and parity-passes locally yet emits garbage (LaMa `-fp16` shipped broken; caught only by downloading + running). Compositing masks it — judge the generated region, not the pasted-back whole.

**Upstreaming a Tier-2 port: the parity harness is NOT the bar.** Run the framework's own `load` → `apply_chat_template` → `generate` on the real checkpoint before opening the PR — Mage-VL was 48/48 token-exact and still broken through `mlx_vlm.load` three separate ways (torch-native remote processor, missing `prompt_utils` entry so the prompt had zero image slots, bare-array return where dispatch wants `InputEmbeddingsFeatures`). Execute the published model card's snippet against the published bytes, stdin closed. (#47)

## MLX-Swift consumer angle (most ports end here)

Most published mlx-community ports are eventually loaded by Swift. The Swift side has its own contract — `Module` property names can't contain dots, `Float64` crashes the GPU, `swift test` can't compile Metal shaders — and getting it wrong wastes a day per port even when weights are correct.

- **Canonical 3-step load + safetensors dotted-key remap + `@unchecked Sendable` GPU-state classes** → `references/repo-layout.md` ("MLX-Swift consumer idioms").
- **`Failed to load the default metallib` on `swift test`** → build via `xcodebuild` against the workspace by default; the bundle-copy + `default.metallib`→`mlx.metallib` rename is the escape hatch for SPM-CLI lanes that need it. → `references/repo-layout.md`.
- **`Float64` → GPU crash, `Float * MLXArray` ambiguity** → `references/common-pitfalls.md`.
- **`.newAxis` subscripts silently no-op unless EVERY axis is named** — Swift-MLX does not imply
  trailing axes, so a transcribed `[:, None]` can leave the shape untouched with no error (or,
  worse, still broadcast and produce wrong numbers). Port `[:, None]` as
  `expandedDimensions(axis:)`. → `mlx-swift-integration` `swift-port-parity.md`.

If "weights load, `model.update` succeeds with `.noUnusedKeys`, but inference is garbage *only on Swift*", it's almost always one of these three before a layer-translation issue.

### Deciding whether to insert an MLX-Python rung (PyTorch → MLX-Python → Swift)

The "standard → MLX → MLX-Swift" path looks like it should always go through the middle. It shouldn't. Inserting your own MLX-Python implementation is a **third parity surface to keep in sync** (PyTorch oracle ↔ MLX-Python ↔ Swift) — worth it only when it actually shrinks the work.

- **Default: port PyTorch → MLX-Swift directly.** Generate **granular per-sub-op goldens** from the PyTorch oracle (one `.npy` per intermediate — patch-embed, each block/hint, the output). They localize a Swift parity break to the exact op *without* a Python twin, which is most of the "isolate the failure" value an intermediate would buy.
- **Insert an MLX-Python rung ONLY when the net-new piece is BOTH large AND novel** (no existing MLX-Python donor for it). Then the rung decouples the two jumps that the direct path fuses: the **framework-port** (PyTorch→MLX semantics — conv/attention/RoPE/lazy-eval, debugged fast in a REPL) and the **language-port** (MLX-Python→Swift — near-mechanical, same framework). Each diff halves. For anything *reuse-shaped* or already mirrored by an upstream MLX-Python ref, the rung pays maintenance for little gain — and note the heaviest Swift-port frictions (the substrate injection seam, `open`/`public`, `Module` reflection) are Swift-specific and a Python layer does nothing for them.
- **If an upstream MLX-Python ref already exists** (e.g. a `mlx-*` repo covering the substrate), that IS your middle rung — prefer **contributing the missing net-new branch upstream** over maintaining a private twin. Gauge their receptiveness to substrate-level additions before committing.
- **Keep any private rung throwaway** — a scratchpad in the oracle/measure dir, never a 4th maintained package.
- **Secondary tip-the-scales rationale:** if the same scratchpad doubles as a proving ground for *non-porting* experiments (perf / optimization strategies, where REPL iteration compounds), that can justify building it even when the porting case alone wouldn't.
- **The oracle is also a standing differential probe, not just a parity fixture.** When a Swift port misbehaves at a *regime the Python reference handles fine* (long seq, large batch, a dtype), **bisect dtype and kernel-path against the oracle BEFORE touching the math.** Diff the actual op sequence — e.g. the oracle's `attention.py` vs your Swift attention: does the reference upcast to fp32 at large seqLen, or is that a Swift-only remedy? Does it run the DiT in bf16 or fp32? A 5-minute source diff against the reference caught a Swift-only fp32 divergence (the thing forcing a materialized-attention OOM) with no new code — far cheaper than re-deriving from the math. This is the highest-leverage use of having the reference on disk.

## Framework constraints to know

- **Metal command-buffer timeout** (≈10s): long graphs without `mx.eval` between steps hit it. Insert `mx.eval(x)` at natural boundaries.
- **Lazy evaluation**: computed only on `mx.eval`/`.item()`/`np.array()`/`mx.save_*`/`print`. Time with `mx.eval` + `mx.synchronize`; save after `mx.eval`.
- **Unified memory**: one pool, no `.to(device)`; `mx.metal.set_memory_limit` if OOM risk. **Video VAE decode is the usual OOM source** — whole-sequence decode peak grows ~linearly in frames, OOMs past ~49f (the DiT often isn't the bottleneck; lossy spatial `decode_tiled` doesn't help — the blow-up is temporal). Fix = temporal-chunked **streaming decode**, flat memory + bit-identical → `references/streaming-decode.md`.
- **No in-place mutation**: `x[idx] = y` is emulated via copy.
- **bf16**: full on M-series GPU; some reductions still fp32 internally.
- **`mx.fast.scaled_dot_product_attention` — fused (flash, non-materializing) vs. materialized fallback, and the fp32-softmax guarantee.** A fused fp32 kernel DOES exist (MLX #1610, fires at q-seqLen ≥ 32) — so "fp32 → materialized N×N → OOM" is **not** categorical. The fused path is **conditionally gated** (head_dim set, and mask handling: an additive mask must promote to q/k/v's dtype); a long-seq fp32-with-additive-mask call on an older core can drop to the materialized `softmax(QKᵀ)@V` path that DOES form `[B,H,Tq,Tk]` → tens of GB at long seq. **Key fact: the fused kernel runs softmax in fp32 regardless of input dtype** — so **bf16 SDPA gets flash memory AND fp32-grade accuracy**, which is why a careful reference (e.g. mlx-video) runs the DiT in bf16 and stays both light and stable. Practical decision tree when a Swift port is heavy/slow at long seq: **(1) try bf16 weights + bf16 SDPA** (disable any fp32-upcast remedy first — it may be redundant if the real NaN fix was per-block `eval`/graph-chaining against the long-seq fused **dispatch race**); **(2) if fp32 is genuinely required, chunk** (manual query-block tiling — the SDPA analog of streaming VAE decode; or split-K two-pass à la mlx-qsdpa / Open-TQ-Metal); **(3) NOT quantization** — int4 shrinks weights, not the fp32 *activation/attention* working set. bf16 stability on M-series is well-evidenced on current MLX (vllm-metal #281, M5 Max @ 0.31.1, cos 0.99999898); a historical "bf16 NaN at long seq" is often **environmental** (beta OS/toolchain) or the materialized fallback, not the fused bf16 kernel. Confirm by A/B at identical shapes/mask: bf16-light + fp32-heavy ⇒ dtype-gated fallback (bf16 is the fix); both-heavy ⇒ non-dtype fallback (mask/head_dim) ⇒ chunk.
- **Known-kernel-bug registry — check BEFORE debugging your own math.** mlx-swift ≤ 0.31.6 (`MLX_METAL_JIT=ON`) mis-instantiates the NAX split-K GEMM with the output dtype (mlx#3797, fixed by #3810; Python wheels are AOT and unaffected): any **half-precision** matmul with **M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N)** returns garbage/NaN on M5-class GPUs — in a transformer that is exactly the **FFN down-projection** (K = 4×hidden) past a token threshold, which is why the symptom reads "bf16 breaks above size X, edit path first" (packed target+ref doubles the sequence). Fix = row-chunk that ONE GEMM at ≤896 rows (exact), NOT full fp32. This single unrecorded bug stranded three ports (LTX → Boogu → Mage) as "mystery bf16 instability" — full registry entry, probes, and the same-symptom checklist: `references/common-pitfalls.md` #35.
- **RNG**: `mx.random` ≠ PyTorch algorithm; seeds not cross-compatible — generate on one side (numpy) and inject on both for parity.
- **Long denoise loops need cache discipline AND process detachment.** Two silent-SIGKILL modes hit hours-long runs (scail-2-mlx, 2026-06): (1) the Metal buffer cache retains freed per-step workspace, ratcheting RSS until macOS memory-pressure kills it — call `mx.clear_cache()` after each step's `mx.eval`, consider `mx.set_cache_limit(8GB)`, log `get_active_memory()/get_peak_memory()` per step. (2) Agent-harness-tracked background tasks die when the panel is stopped — launch any >10-min GPU job as `nohup bash -c '...' > out.log 2>&1 & disown` (the harness task must never BE the run), with at most an expendable notify-only watcher.

## Reference files

Open these only when you need the detail — `SKILL.md` gives the workflow, references give depth.

- `references/mlx-docs.md` — Curated official URLs: core, nn, fast, custom kernels, quantization, mlx-lm/vlm/audio/community, recent additions.
- `references/mlx-community-conventions.md` — Publishing contract: repo naming, quant-suffix grammar, quant-collection grouping, README frontmatter from `create_model_card()`, body templates (LLM/VLM/audio), mirror policy for `xocialize` Swift packages.
- `references/manual-port-templates.md` — Tier-2 file shape: `ModelArgs`+`Model`+`sanitize` template from `mlx_lm/models/qwen2.py`, mlx-vlm/mlx-audio sub-package layouts, CONTRIBUTING checklist.
- `references/weight-conversion.md` — Three-tier routing, canonical `mlx_*.convert` invocations, output layout, lazy-tensor materialization (Tier 3), conv-transpose table, mlx-forge recipe skeleton, DWQ calibration for custom architectures.
- `references/common-pitfalls.md` — The reading-time + harness traps (defaults, head-dim, QKV interleaving, weight layout, norms, flags, checkerboard, resolved-config/YaRN, structural drift, VAE numerics, API-surface≠capability, extend-before-fork, AutoTokenizer-hangs, background-uv-cd, grad-accum, long-run detachment) with past-failure examples and idiomatic solutions.
- `references/attention-patterns.md` — Attention translation + `mlx_lm.models.base`/`rope_utils` pointers: MHA, GQA, rotary, QKV interleaving detection, SDPA fast path, mask conventions.
- `references/parity-testing.md` — Templates, threshold table, isolation strategy, torch-as-optional-dep, CPU-stream pinning, largest-production-grid + decoded-output gating, the exact-match ceiling for cross-backend AR (+ oracle-capture protocol), quantized-generative gating.
- `references/repo-layout.md` — `-mlx` fork layout (Tier 3), monorepo vs single-package, HF auto-download wiring, Swift consumer side under `xocialize` (MLX-Swift consumer idioms, SPM-CLI metallib), README conventions, weight-repo naming.
- `references/spatial-and-rope-ops.md` — Hand-rolled NHWC ops not in `mlx_arsenal`/`mx.fast`, parity-verified: bilinear `grid_sample`, bilinear `interpolate`, **3D-RoPE** (per-axis head_dim split + concat-tiled/interleaved quirk), 3D-tubelet Conv3d. From `rife-mlx` + `vjepa2-mlx`.
- `references/streaming-decode.md` — Memory-bound video VAE decode: temporal-chunked streaming (flat memory, bit-identical), `CausalConv3d` feat_cache, the `upsample3d` "Rep"-sentinel cross-impl trap, CPU bit-identity gating, cross-port reuse.

## Bundled scripts

- `scripts/parity_helpers.py` — Reusable PT-vs-MLX parity helpers: `make_seeded_input`, `pt_to_mx`, `mx_to_np`, `transpose_pt_conv`, `load_pt_state_into_mx`, `assert_parity`, `tensor_stats`. Copy into your fork at `tests/parity/_helpers.py`.

## When to stop and ask the user

- Reference has multiple branches/modes and it's unclear which production uses — ask which checkpoint/config to match.
- A PyTorch op has no direct MLX equivalent and a custom Metal kernel seems needed — confirm first; the arsenal or `mx.fast` may have added coverage.
- About to deviate from a reference config "to fix" an artifact — stop and ask. Deviations almost always hide bugs.
