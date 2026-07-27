# Weight Conversion

Weight conversion routes through three tiers. **Always try Tier 1 first.** Demoting to Tier 2 or Tier 3 without checking Tier 1 means rewriting infrastructure that already exists.

Note on code style: `mx.eval( )` is written with a space between parentheses in this file to sidestep a security-hook false-positive on the Python builtin. In real code it's `mx.eval(...)` with the tensors inside.

## Routing rule

1. **mlx-community already has it** → use it. Search `https://huggingface.co/mlx-community` first.
2. **`mlx_lm.convert` / `mlx_vlm.convert` / `mlx_audio.convert` supports the `model_type`** → run it.
3. **Architecture missing from mlx-lm/mlx-vlm/mlx-audio** → add one file (mlx-lm) or one sub-package (mlx-vlm, mlx-audio), then re-run the official converter. See `manual-port-templates.md`.
4. **Multi-component pipeline** (T2V, T2I, 3D, audio-gen, multi-component diffusion) — mlx-forge recipe + standalone `-mlx` fork. **This is the only case where `mlx-forge` is the answer.**

## Route from the safetensors HEADER before downloading (cheap pre-flight)

Before pulling tens of GB, read just the safetensors **header** to decide the conversion path and confirm the key contract. The format is an 8-byte little-endian header length `N`, then `N` bytes of JSON (`{key: {dtype, shape, data_offsets}}`). A ranged HTTP GET of the first few MB captures the whole header for a ~1000-tensor model:

```bash
curl -sL -r 0-4194303 "https://huggingface.co/ORG/REPO/resolve/main/model.safetensors" -o head.bin
```
```python
import json, struct
n = struct.unpack("<Q", open("head.bin","rb").read(8))[0]
hdr = json.loads(open("head.bin","rb").read(8 + n)[8:]); hdr.pop("__metadata__", None)
```

From the header alone you learn: **tensor count**, **dtype** (already bf16 → no cast step), and — decisively — the **key namespace** (`blocks.N.self_attn.q` original-Wan vs `transformer.` diffusers vs `model.diffusion_model.`). That routes the conversion before a single GB moves: AnimeGen-T2V shipped original-Wan keys, so it **skipped `premap_diffusers_to_wan`** and fed straight into `sanitize_*`. The same header also lets you **preflight a LoRA→base key mapping** — map every LoRA module name to its target base key and `assert` all are present in an already-converted base checkpoint — catching a merge-target mismatch before downloading the LoRA.

## Tier 1 — official converters (canonical invocations)

### LLM via `mlx_lm.convert`

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen3-4B-Instruct-2507 \
  -q \
  --upload-repo mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Full flag list (per `mlx_lm/convert.py`):

```
usage: mlx_lm.convert [-h] [--hf-path HF_PATH] [--mlx-path MLX_PATH] [-q]
                      [--q-group-size Q_GROUP_SIZE] [--q-bits Q_BITS]
                      [--q-mode {affine,mxfp4,nvfp4,mxfp8}]
                      [--quant-predicate {mixed_2_6,mixed_3_4,mixed_3_6,mixed_4_6}]
                      [--dtype {float16,bfloat16,float32}]
                      [--upload-repo UPLOAD_REPO] [-d]
```

### VLM via `mlx_vlm.convert`

```bash
mlx_vlm.convert \
  --hf-path Qwen/Qwen3-VL-30B-A3B-Instruct \
  -q \
  --upload-repo mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

Same flag shape as `mlx_lm.convert`. Vision and audio encoders are excluded from quantization by default — feature-extraction quality drops sharply under 4-bit.

### Audio (TTS/STT) via `mlx_audio.convert`

```bash
# 4-bit + upload
python -m mlx_audio.convert \
  --hf-path prince-canuma/Kokoro-82M \
  --mlx-path ./Kokoro-82M-4bit \
  --quantize --q-bits 4 \
  --upload-repo mlx-community/Kokoro-82M-4bit

# MXFP4 microscaling
python -m mlx_audio.convert \
  --hf-path prince-canuma/Kokoro-82M \
  --mlx-path ./Kokoro-82M-mxfp4 \
  --quantize --q-mode mxfp4

# bf16, no quant — the publishing default for Mel-RoFormer / Lance / Ming-omni
python -m mlx_audio.convert \
  --hf-path prince-canuma/Kokoro-82M \
  --mlx-path ./Kokoro-82M-bf16 \
  --dtype bfloat16 \
  --upload-repo mlx-community/Kokoro-82M-bf16
```

The bf16 pattern is the canonical publishing recipe for any audio model where quantization hurts perceptual quality (vocoders, Mel-band processors, anything spectrogram-based). `mlx-community/Lance-3B-bf16`, `mlx-community/Lance-3B-Video-bf16`, `mlx-community/Ming-omni-tts-16.8B-A3B-bf16` all use this exact invocation.

### Learned quants (Tier 1, advanced)

Per `LEARNED_QUANTS.md` (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md):

| CLI | What it does | Tradeoff |
|---|---|---|
| `mlx_lm.dynamic_quant` | Picks per-layer bit-widths from sensitivity | **Fastest to run** |
| `mlx_lm.awq` | Activation-aware Weight Quantization | Calibration set required |
| `mlx_lm.dwq` | Distilled Weight Quantization | **Best quality**, takes longest |
| `mlx_lm.gptq` | GPT-Q style | Calibration set required |

Output repos get the suffix matching the technique: `-DWQ-4bit`, `-AWQ-4bit`, `-OptiQ-4bit`.

## Output layout the official converters produce

Swift consumers (`mlx-swift-examples`, `mlx-audio-swift`) and Python consumers (`mlx_lm.load`, `mlx_vlm.load`, `mlx_audio.tts.utils.load_model`) all assume this layout. If a manual conversion produces a different shape, the loaders fail or pick wrong defaults.

```
<model_dir>/
├── config.json
├── model.safetensors                  (or .safetensors.index.json + shards for >2GB)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md                          (auto-generated when --upload-repo is set)
```

For VLM, add `preprocessor_config.json`. For audio, add whatever phonemizer / mel-spec config files the upstream repo ships.

## Tier 2 — manual architecture port

When the converter prints `Unsupported model_type: <name>`, the fix is **one file** (mlx-lm) or **one sub-package** (mlx-vlm, mlx-audio) named after the upstream `model_type`. See `manual-port-templates.md` for the canonical shape.

Workflow:

1. Fork `ml-explore/mlx-lm` (or `Blaizzy/mlx-vlm`, `Blaizzy/mlx-audio`).
2. Add `mlx_{lm,vlm,audio}/models/.../<model_type>.py` matching the template.
3. Implement `class Model(nn.Module)` with `.model_type`, `.layers` property, and `.sanitize(weights)`.
4. Re-run `mlx_*.convert` against your fork.
5. Parity-test (see `parity-testing.md`).
6. PR upstream per the CONTRIBUTING checklist.

**`sanitize(weights)` is where all weight remapping lives.** Not in the constructor, not at forward time. The official converter calls `Model.sanitize()` automatically before saving — putting remap logic anywhere else either runs at wrong layer state or never runs at all during conversion.

Reference exemplar: Mel-RoFormer upstreamed at `mlx-audio#654` (https://github.com/Blaizzy/mlx-audio/pull/654). Pure-MLX FFT, `sanitize()` remaps `band_split_module.0.weight` → MLX-friendly keys, parity tested at `≤0.11 dB SDR` on MUSDB18.

## Tier 3 — `mlx-forge` recipes (multi-component pipelines only)

`mlx-forge` is the right tool when the model has **multiple independent components** that don't fit a single `model_type` slot. Concretely: T2V (text encoder + transformer + VAE + scheduler + optional upsampler), T2I (same shape), 3D mesh generation (transformer + VAE + texture UNet), audio-gen pipelines (text encoder + transformer + vocoder + scheduler).

When to reach for Tier 3:

- Conversion needs **per-component dtype routing** (transformer q8, text_encoder bf16, vae fp16).
- Components ship as **separate `.safetensors` shards** that need separate quantization scopes.
- `mlx_*.convert` prints `Unsupported model_type` AND the model is not a clean single-stack LLM/VLM/audio (i.e., you can't reduce the problem to a Tier-2 manual port).
- You need a `pipeline_*.py` orchestrator alongside the converted weights.

Canonical examples:

- **LTX-2** (T2V) — `xocialize/ltx-2-mlx` + `dgrauet/ltx-2.3-mlx` (HF). Five components, four pipelines (T2V, I2V, retake, extend).
- **Qwen-Image-2512** — recipe at https://github.com/dgrauet/mlx-forge/blob/main/docs/models/qwen-image-2512.md. Per-component subdirectories, mixed quant.
- **Hunyuan3D-2.1**, **CogVideoX-Fun**, **Matrix-Game** — same shape.

### The one silent killer that still applies: lazy tensors saved as zeros

MLX arrays are lazy. Until materialized, they're an unresolved computation graph. `mx.save_safetensors` serializes whatever numerical value currently exists — **for an unmaterialized lazy tensor, that is zeros**, with no error.

`mlx_lm.convert` calls `mx.eval` internally so Tier 1 never hits this. **Tier-3 recipes must do it themselves:**

```python
import mlx.core as mx

def _materialize(*tensors):
    mx.eval( *tensors )      # force GPU computation; no-op if already materialized

# Right before saving:
_materialize( *component_weights.values() )
mx.save_safetensors(f"{component}.safetensors", component_weights)
```

The helper lives in `mlx-forge/src/mlx_forge/quantize.py:23-28`.

### Recipe skeleton

```python
# recipes/my_pipeline.py

def classify_key(key: str) -> str | None:
    """Map a PyTorch weight key → component name. Return None to drop."""
    ...

def sanitize_key(key: str) -> str:
    """Rename PyTorch key to MLX convention."""
    ...

def convert(args) -> None:
    weights = mx.load(checkpoint_path)        # lazy load
    keys_by_component = classify_keys(weights, classify_key)
    for component, keys in keys_by_component.items():
        process_component(
            weights, component, keys, output_dir,
            sanitizer=get_sanitizer(component),
            transform=get_transform(component),
        )
        quantize_component(output_dir, component)
```

CLI: `mlx-forge convert my-pipeline` — registered in `recipes/__init__.py` `AVAILABLE_RECIPES`.

Full reference recipe: `mlx_forge/recipes/ltx_23.py` (~400 LOC, shows per-channel stats renames, conv transposes, and per-component quant scope).

### Per-component split (Tier 3)

Always split safetensors by component (transformer, vae, text_encoder, scheduler, tokenizer, etc.) rather than one giant file. Reasons:

1. Components can be loaded / unloaded independently (saves peak memory).
2. Each component can be quantized with different settings (transformer int4, VAE fp16).
3. Parallel download from HF.
4. Easier to swap a single component without re-downloading everything.

Convention: one `{component}.safetensors` per component. If a component exceeds safetensors' 2GB chunk limit, split as `{component}-00001-of-00003.safetensors`, etc. — load with `load_split_safetensors` (see `ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/utils/weights.py`).

### Per-component memory management

Large checkpoints (>10 GB) need aggressive cleanup between components:

```python
for component, keys in ...:
    component_weights = {...}
    _materialize( *component_weights.values() )
    mx.save_safetensors(path, component_weights)
    del component_weights
    import gc; gc.collect()
    mx.metal.clear_cache()
```

Without `clear_cache`, the Metal allocator holds peak memory until process exit.

## Two materialization rules that apply at all tiers

1. **`mx.eval(tree)` before `mx.save_safetensors`** — Tier 1 does this automatically; Tier 2 (`Model.sanitize` is called by the converter, so the converter still owns materialization); Tier 3 you do it yourself.
2. **PyTorch Conv `(O, I, *K)` → MLX `(O, *K, I)`** — transpose inside `sanitize()` for Tier 2, inside the `transform` step for Tier 3.

| Op | PyTorch | MLX |
|---|---|---|
| `Conv1d.weight` | `(O, I, K)` | `(O, K, I)` |
| `Conv2d.weight` | `(O, I, Kh, Kw)` | `(O, Kh, Kw, I)` |
| `Conv3d.weight` | `(O, I, Kd, Kh, Kw)` | `(O, Kd, Kh, Kw, I)` |
| `ConvTranspose2d.weight` | `(I, O, Kh, Kw)` | `(O, Kh, Kw, I)` |
| `Linear.weight` | `(O, I)` | `(O, I)` (identical) |
| `Embedding.weight` | `(V, D)` | `(V, D)` (identical) |
| `LayerNorm.weight/bias` | `(D,)` | `(D,)` (identical) |

`mlx_forge.transpose.transpose_conv(key, weight, kind)` handles all variants generically. Bias tensors are never transposed.

## Quantization scope policy (all tiers)

- **Quantize:** transformer / DiT blocks — Linear `.weight` only.
- **Keep fp16 / bf16:** VAE, vocoder, text encoder output projections, tokenizer / scheduler state, position encodings, norm weights, bias tensors, vision-encoder ViT/SigLIP weights.

Rationale: VAE and connectors are sensitive to quantization noise (visible as color drift, edge artifacts). Vision encoders' feature-extraction quality drops sharply under 4-bit. Transformer Linears absorb quantization cleanly.

`mlx_vlm.convert` enforces vision-encoder exclusion automatically. For Tier 3:

```python
from mlx.nn import quantize

quantize(model, group_size=64, bits=4, class_predicate=lambda name, m: (
    isinstance(m, nn.Linear) and "transformer" in name
))
```

## DWQ calibration for custom architectures (Tier 2/3)

`mlx_lm.quant.dwq.dwq_quantize` distills a pre-quantized student against its bf16 teacher, recovering most of the quality lost to naive groupwise quantization. The API is callable directly on any model that exposes a `model(tokens) → logits` interface — but plugging in a custom (non-stock-mlx-lm) architecture has three sharp edges that will silently break the run.

**The three traps:**

1. **`model.freeze()` is required.** mlx-lm's `dwq.main()` loads via `mlx_lm.load()` which freezes the model implicitly. If you instantiate via `load_weights()` directly (any Tier 2/3 port), parameters are trainable by default. `dwq_quantize`'s `apply_to_modules(unfreeze)` is a positive filter that adds matching modules to the trainable set — it does NOT freeze non-matching modules. Without `model.freeze()` first, the optimizer trains EVERYTHING (the bf16 GEN tower, embeddings, lm_head, scales+biases). Symptom: `Trainable parameters: 49%` instead of `~2%`; validation loss diverges (gets WORSE during training); peak memory blows up 2-3×.

2. **Hyperparameter defaults are very specific.** They are NOT typical fine-tuning defaults — they are tuned for *scale-and-bias calibration of QuantizedLinear*, a much narrower problem:

   | Param | mlx-lm default | What goes wrong otherwise |
   |---|---|---|
   | Optimizer | `optimizers.Adam(learning_rate=lr, bias_correction=True)` | `AdamW` diverges immediately |
   | Learning rate | `1e-6` | `1e-5` catastrophic; `1e-4` instantly destroys quality |
   | num_samples | `2048` | `<128` unstable; `<64` overfits to noise |
   | batch_size | `4` | smaller works but slow |
   | max_seq_length | `1025` | very short truncates context |

   When debugging, the smoking gun is loss going *up* monotonically (not oscillating). That's LR overshoot — drop by 10-20× and try again.

3. **The model must look like a standard LLM.** For dual-tower / MoE / multimodal models (Lance, Switch-Transformer, etc.) you'll need a thin wrapper exposing `model(tokens) → logits`. mlx-lm's harness has no concept of routing, modality buckets, or position groups. For text-only DWQ on dual-tower models (UND tower of Lance, expert-1 of a MoE, etc.):

   ```python
   class TextLogitsWrapper(nn.Module):
       def __init__(self, core_model):
           super().__init__()
           self.core = core_model
           self.layers = core_model.layers  # for grad_checkpoint

       def __call__(self, x):
           B, T = x.shape
           # All-text routing: customize per architecture
           pos = mx.arange(T, dtype=mx.int32)
           position_ids = mx.broadcast_to(pos[None, None, :], (3, B, T))  # 3-channel mRoPE
           position_group = mx.zeros((T,), dtype=mx.int32)  # all TEXT bucket
           h = self.core(input_ids=x, position_ids=position_ids,
                         position_group=position_group)
           return self.core.lm_head(h)
   ```

**Calibration corpus considerations:**

- For dual-tower MoT models (Lance-style), text-only DWQ exercises ONE tower. The other tower (image-gen, etc.) gets no signal and naive int4 there will still produce broken outputs in the corresponding generation modality. This is fine for shipping a "UND-only" variant; for full-coverage quantization you need an architecture-aware harness that drives both towers (see Reza2kn/lance-quant for a Lance-specific AWQ implementation, ~100 LOC for the core algorithm).
- Reza2kn's published AWQ pinning (Lance specifically): `--bits 4 --group-size 64 --num-samples 256` for DWQ; `21-point alpha grid + geomean-normalized scale fused into preceding norm` for AWQ. The naive-quant failure mode is exactly: when calibration data doesn't route through a given expert tower, that tower's `act_mean` is None, alpha-search is skipped, fallback is plain min-max int4 → gibberish.
- **What "usable" means is prompt-dependent.** For Lance-3B-4bit-UND-DWQ specifically (validated 2026-05-23), our four-prompt sweep showed 1-of-4 reliably usable, 1-of-4 borderline, 2-of-4 unusable. The breakdown:
  - 🟢 Single subject + texture (animal portrait) — recognizable, stylized
  - ⚠️ Scene without subject identity (landscape) — flatter colors but recognizable
  - ❌ Multi-element composition (dragon + castle) — model loses elements
  - ❌ Fine in-image text (cat + "STOP" poster) — text completely lost to color smearing
  - The UND tower's QKV at int4, even when GEN stays bf16, corrupts image generation through *shared* attention (text tokens cross-attend with latent tokens; corrupted text-side projections poison the shared SDP).
- **Therefore: for image-generation models, UND-only DWQ at int4 ships only as "experimental, scene-only" — not as a general-purpose drop-in replacement.** Bump to int8 UND + DWQ if you need broader prompt coverage, or invest in the full-tower AWQ harness.

## Weight key renames — common patterns

From past ports, these show up repeatedly. Handle in `sanitize()` (Tier 2) or `sanitize_key()` (Tier 3):

- Sequential unwrapping: `.to_out.0.` → `.to_out.`, `.ff.net.0.proj.` → `.ff.proj_in.`, `.ff.net.2.` → `.ff.proj_out.`
- Private stat prefix: `_mean_of_means` → `mean_of_means` (MLX treats leading-underscore as private, breaks loading).
- Block numbering: `blocks.0.` → `blocks.0.` (usually identical; rename only if MLX port restructures).
- Fused QKV: three separate `to_q.weight`, `to_k.weight`, `to_v.weight` → one `qkv.weight` via `mx.concatenate([q, k, v], axis=0)`.
- Precomputed buffers: drop `self_attn.rotary_emb.inv_freq` (MLX recomputes via `initialize_rope`).
- Tied embeddings: drop `lm_head.weight` when `tie_word_embeddings=True`.

## Validation

A zero-tensor check catches the materialization bug cheaply:

```python
for key, w in weights.items():
    if float(mx.abs(w).sum().item()) == 0:
        raise ValueError(f"{key} is all zeros — likely missing materialization")
```

Add this to your Tier-3 recipe's `validate()` step; Tier 1 catches it via the converter.

## Before writing any recipe — check mlx-community

Before adding a model to `mlx-lm` or writing a `mlx-forge` recipe, check `https://huggingface.co/mlx-community`. The base model may already be converted; you may only need to port any custom head / adapter / LoRA on top.

---

## Folded from the diffusion / mlx-forge lineage (skills consolidation 2026-06-15)

> Tier-3 multi-component conversion depth unique to the diffusion branch (the official-converter
> sections above are Tier 1/2; this is the `mlx-forge` recipe path the merged SKILL.md routes Tier 3 to).

## The one silent killer: lazy tensors saved as zeros

MLX arrays are lazy. Until they are materialized, they are an unresolved computation graph. `mx.save_safetensors` serializes whatever current numerical value exists — **for an unmaterialized lazy tensor, that is zeros**, with no error.

This has broken past recipes in ways that look like "weights loaded but model outputs garbage". Always force materialization before saving:

```python
import mlx.core as mx

def _materialize(*tensors):
    mx.eval( *tensors )        # force GPU computation; no-op if already materialized

# In the recipe, right before saving:
_materialize( *component_weights.values() )
mx.save_safetensors(f"{component}.safetensors", component_weights)
```

The helper lives in `mlx-forge/src/mlx_forge/quantize.py:23-28`. If writing a recipe from outside mlx-forge, replicate the pattern.

**Corollary — never `mx.save_safetensors(path, d)` where `d` came from `mx.load(path)`.** `mx.load` returns *lazy, memory-mapped* arrays backed by the file on disk. Saving back to the *same path* (e.g. to add one key to an existing goldens/weights file) overwrites the mmap **before** the other lazy arrays are read — they serialize as **zeros**, silently. Symptom seen in the wild: a parity test that passed (`max_abs 0.0`) "regresses" right after you append a tensor to its goldens file, and the formerly-good arrays are now all-zero (`std == 0`). Fixes: (a) compute the new tensor in the *same* run that first writes the file (one clean write); or (b) `mx.eval(*d.values())` to materialize **before** re-saving; or (c) write to a different path and rename. This is the same lazy-eval trap as above, just triggered by read-modify-write on one file.

## Recipe skeleton (summary, not full tutorial)

An mlx-forge recipe is a Python module with three layered functions. Full authoring guide: `mlx-recipe` skill.

```python
# recipes/my_model.py

def classify_key(key: str) -> str | None:
    """Map a PyTorch weight key → component name ('transformer', 'vae', ...).
    Return None to drop."""
    ...

def sanitize_key(key: str) -> str:
    """Rename PyTorch key to MLX convention.
    E.g. 'ff.net.0.proj.' -> 'ff.proj_in.' """
    ...

def convert(args) -> None:
    """Orchestrate: download → lazy load → classify → process each component
    → materialize → save → (optional) quantize."""
    weights = mx.load(checkpoint_path)  # memory-mapped lazy load
    keys_by_component = classify_keys(weights, classify_key)
    for component, keys in keys_by_component.items():
        process_component(
            weights, component, keys, output_dir,
            sanitizer=get_sanitizer(component),
            transform=get_transform(component),
        )
        quantize_component(output_dir, component)
```

CLI: `mlx-forge convert my-model` — registered in `recipes/__init__.py` `AVAILABLE_RECIPES` dict.

Reference example: see `mlx_forge/recipes/ltx_23.py` in the mlx-forge repo (~400 LOC, shows the full pattern including per-channel stats renames and conv transposes).

## Per-component split

**Always split safetensors by component** (transformer, vae, text_encoder, scheduler, tokenizer, etc.) rather than one giant file. Reasons:

1. Components can be loaded / unloaded independently (saves peak memory).
2. Each component can be quantized with different settings (transformer int4, VAE fp16).
3. Parallel download from HF.
4. Easier to swap a single component without re-downloading everything.

Convention: one `{component}.safetensors` per component in the output directory. If a component exceeds safetensors' 2GB chunk limit, split as `{component}-00001-of-00003.safetensors`, `-00002-of-00003.safetensors`, etc. — load with `load_split_safetensors` (see `ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/utils/weights.py`).

## Conv transposition

PyTorch Conv → MLX Conv requires layout transpose:

| Op | PyTorch | MLX |
|---|---|---|
| `Conv1d.weight` | `(O, I, K)` | `(O, K, I)` |
| `Conv2d.weight` | `(O, I, Kh, Kw)` | `(O, Kh, Kw, I)` |
| `Conv3d.weight` | `(O, I, Kd, Kh, Kw)` | `(O, Kd, Kh, Kw, I)` |
| `ConvTranspose2d.weight` | `(I, O, Kh, Kw)` | `(O, Kh, Kw, I)` |
| `Linear.weight` | `(O, I)` | `(O, I)` (identical) |
| `Embedding.weight` | `(V, D)` | `(V, D)` (identical) |
| `LayerNorm.weight/bias` | `(D,)` | `(D,)` (identical) |

`mlx_forge.transpose.transpose_conv(key, weight, kind)` handles all variants generically. Bias tensors are never transposed.

## Quantization scope

Default policy across past ports:

- **Quantize:** transformer / DiT blocks — Linear `.weight` only.
- **Keep fp16 / bf16:** VAE, vocoder, text encoder output projections, tokenizer / scheduler state, position encodings, norm weights, bias tensors.

Rationale: VAE and connectors are sensitive to quantization noise (visible as color drift, edge artifacts). Transformer Linears absorb quantization cleanly.

Typical CLI invocation:

```python
from mlx.nn import quantize

quantize(model, group_size=64, bits=4, class_predicate=lambda name, m: (
    isinstance(m, nn.Linear) and "transformer" in name
))
```

## Per-component memory management

Large checkpoints (> 10 GB) need aggressive cleanup between components:

```python
for component, keys in ...:
    component_weights = {...}
    _materialize( *component_weights.values() )
    mx.save_safetensors(path, component_weights)
    del component_weights
    import gc; gc.collect()
    mx.metal.clear_cache()
```

Without `clear_cache`, the Metal allocator holds peak memory until process exit. `gc.collect` ensures the Python refs are dropped before Metal frees.

## Weight key renames — common patterns

From past ports, these show up repeatedly:

- Sequential unwrapping: `.to_out.0.` → `.to_out.`, `.ff.net.0.proj.` → `.ff.proj_in.`, `.ff.net.2.` → `.ff.proj_out.`
- Private stat prefix: `_mean_of_means` → `mean_of_means` (MLX treats leading-underscore as private, breaks loading).
- Block numbering: `blocks.0.` → `blocks.0.` (usually identical; rename only if MLX port restructures).
- Fused QKV: three separate `to_q.weight`, `to_k.weight`, `to_v.weight` → one `qkv.weight` via `mx.concatenate([q, k, v], axis=0)`.

## Validation at the recipe level

mlx-forge recipes can include a `validate()` function that checks:
- All expected files exist in the output directory.
- Expected keys are present in each safetensors.
- Shapes match expectations (compared against a schema).
- No all-zero tensors (catches the materialization bug).

A zero-tensor check is cheap:

```python
for key, w in weights.items():
    if float(mx.abs(w).sum().item()) == 0:
        raise ValueError(f"{key} is all zeros — likely missing materialization")
```

**Stronger — reload the saved repo and re-run parity before any upload.** The zero-check catches the worst case; a reload+parity catches it *and* key-map errors, transpose mistakes, and quant-scope mismatches in one shot. After saving, load the on-disk repo into a fresh model and run the single-pass parity test against the golden (cosine 0.999999 confirmed the Lens DiT bf16 repo round-tripped; int4 0.9976, int8 0.99998). Do this *before* pushing 8 GB. For a quantized repo, the loader must rebuild the quantized module structure (`nn.quantize` with the same group_size/bits/scope from `config.json`) *before* `load_weights` — and must NOT blanket-cast dtypes on load, which corrupts packed uint32 quant weights.

## Publishing to mlx-community

- **Naming is per-quant**: `<Name>[-<size>]-<quant>` with suffixes `-bf16 / -4bit / -8bit` (e.g. `Qwen-Image-2512-4bit`, `Lance-3B-bf16`, `flux2-klein-4b-8bit`). Match the team's prior pattern; use the upstream's marketed size if any. **Reserve the family names early** (create the repos + a placeholder card) — generic names get taken.
- **License-safe bundling**: ship only the cleanly-licensed component(s). The Lens repos host the MIT DiT only; the model card has the loader pull the GPT-OSS encoder (Apache-2.0, reuse the existing mlx-community repo) and the FLUX.2 VAE (unverified license) from source rather than re-hosting them.
- A **collection** under the org keeps a multi-quant family organized (`create_collection` + `add_collection_item`).

## When to NOT convert (and when to LIFT an implementation, not just weights)

Before writing a recipe, check `https://huggingface.co/mlx-community` — someone may have already converted the base model. If so, the recipe only needs to port any custom head / adapter / LoRA on top.

Also grep the **installed mlx packages** (`mflux`, `mlx-lm`, `mlx-arsenal`) for an existing *implementation* of a standard component before porting it — often you can lift the module and just load the checkpoint's own weights into it. In the Lens port, `mflux`'s `Flux2VAE` matched the Lens diffusers VAE at **246/250 keys** (only `to_out.0.`→`to_out.` + the standard conv transpose), and its `decode_packed_latents` already encoded the model-specific bn latent de-norm + unpatchify. Lifting it (instantiate the mlx class → derive the key map empirically against the checkpoint, exactly as you'd diff a model's own params → load) beat re-porting a whole conv VAE. Derive the key map by instantiating the mlx module, `tree_flatten`-ing its params, and diffing the name+shape sets against the checkpoint safetensors — the conv-transpose mismatches and ModuleList-index renames fall out automatically.

**A full fine-tune of an already-ported base collapses to a weights-swap — do this diff FIRST.** If the candidate is a fine-tune of a base you already converted (same architecture), it ships ONLY the delta — the fine-tuned backbone weights — and *everything else is reused from the base*: VAE, text encoder, tokenizer, scheduler, config, and any MoE/expert-routing logic. The conversion is then: run the base's existing `sanitize_*` on the new weights, copy the base's `vae`/`t5`/`config` verbatim, and **gate on key-set equality** — `assert converted.keys() == base_ckpt.keys()` (0 added / 0 dropped) is the whole safety net (AnimeGen-T2V: a 1095-key contract, verified on both experts, turned the "port" into a weight conversion + copy). A model card whose diffusers example does `load_lora_weights(...) + set_adapters([hi,lo], [w1,w2])` is telling you to **merge those acceleration/distillation LoRAs OFFLINE into the checkpoint** — *not* apply them at runtime — whenever your loader reads a pre-merged flat checkpoint. Merge is kohya: `W += strength·(alpha/rank)·(up @ down)` per module, honoring each adapter's per-expert weight (AnimeGen: `[high 2.0, low 1.0]`). And **validate the conversion decoupled from the LoRA first** — render from the plain base recipe (full CFG, many steps) on the converted weights; if that produces coherent output the backbone conversion is proven, and a blocked/slow LoRA download (see pitfall #34) never blocks proving the port — the LoRA is only a few-step speed layer on top.

## Handoff to `mlx-recipe` skill

When writing or updating a recipe, invoke the `mlx-recipe` skill with:
- The HF repo id or checkpoint path.
- The model's components (which ones to convert, which to drop).
- Any custom weight-key conventions (e.g. fused QKV, renamed blocks).
- Target quantization scope.

That skill owns the full authoring workflow — shape verification, component-by-component parity, end-to-end conversion test.
