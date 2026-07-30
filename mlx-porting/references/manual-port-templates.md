# Manual Architecture Port Templates

When `mlx_lm.convert` / `mlx_vlm.convert` / `mlx_audio.convert` returns `Unsupported model_type: <name>`, the fix is **one file** (mlx-lm) or **one sub-package** (mlx-vlm, mlx-audio) named after the upstream `config.json` `model_type`. Add it to your fork, re-run the official converter, parity-test, then upstream.

This file is the canonical shape. Copy-paste, edit, parity-test, PR.

## mlx-lm — `mlx_lm/models/<model_type>.py`

Filename **must equal** the `model_type` in upstream `config.json`. `mlx_lm.utils._get_classes()` imports `mlx_lm.models.{model_type}` dynamically by string — a mismatch is a `ModuleNotFoundError` at load time, not a friendly error.

Canonical shape (from `mlx_lm/models/qwen2.py`):

```python
from dataclasses import dataclass
from typing import Optional, Dict, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # q_proj / k_proj / v_proj / o_proj
        # rope = initialize_rope(args.head_dim, args.rope_theta, args.rope_traditional, args.rope_scaling, args.max_position_embeddings)
        ...

    def __call__(self, x, mask=None, cache=None):
        # Use scaled_dot_product_attention from .base — handles GQA natively
        ...


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type           # REQUIRED — mlx-lm checks this
        self.model = Qwen2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Standard remap hook. Drop tied weights, drop precomputed inv_freq,
        rename keys that don't match the MLX layer tree."""
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {k: v for k, v in weights.items()
                if "self_attn.rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers
```

Three structural rules to memorize:

1. **Filename equals `model_type`** from upstream `config.json`. (mlx-lm imports `mlx_lm.models.{model_type}` by string.)
2. **The outer class is always `Model`**, sets `self.model_type`, exposes a `layers` property, implements `sanitize(weights)`.
3. **`ModelArgs` is `@dataclass(BaseModelArgs)`** with the upstream HF config fields as kwargs. Defaults must match the **trained** config, never the framework defaults (see common-pitfalls.md #1).

## mlx-vlm — sub-package `mlx_vlm/models/<model_type>/`

A VLM is a sub-package, not a single file. Layout is consistent across `qwen2_vl/`, `qwen2_5_vl/`, `qwen3_vl/`, `qwen3_vl_moe/`, `llava/`, `pixtral/`, `paligemma/`, `phi3_v/`, `mllama/`, `gemma3/`, `gemma4/`:

```
mlx_vlm/models/<model_type>/
├── __init__.py
├── <model_type>.py         # top-level Model glue: wires language + vision, defines sanitize
├── language.py             # text decoder (often imports mlx_lm/models/<base> if reusable)
├── vision.py               # image encoder (ViT, SigLIP, custom)
├── config.py               # @dataclass ModelArgs (one TextConfig + one VisionConfig)
└── processor.py            # optional — image preprocessing if not handled by HF AutoProcessor
```

Same three rules as mlx-lm, plus: image/vision encoders stay **unquantized by default** (`mlx_vlm.convert` excludes them automatically — feature-extraction quality drops sharply under 4-bit). If you have a config flag that gates QK-norm in the text decoder, mirror it from the upstream config.

## mlx-audio — sub-package `mlx_audio/{tts,stt}/models/<name>/`

Same sub-package pattern. Examples: `mlx_audio/tts/models/kokoro/`, `qwen3_tts/`, `vibevoice/`, `ming_omni/`, and on the STT side `mlx_audio/stt/models/whisper/`, etc.

```
mlx_audio/tts/models/<name>/
├── __init__.py
├── <name>.py               # outer Model with .sanitize(weights)
├── <component>.py          # encoder / decoder / vocoder per architecture
├── config.py
└── (optional) tokenizer.py / phonemizer.py / etc.
```

Mel-RoFormer (`mlx-audio#654`, https://github.com/Blaizzy/mlx-audio/pull/654) is the canonical reference for what "right" looks like here: pure-MLX FFT, weight `sanitize()` remaps the upstream `band_split_module.0.weight` → MLX-friendly keys, parity tested against the PyTorch reference at `≤0.11 dB SDR` on MUSDB18.

**Three mlx-audio-specific contract points a parity-passing model still gets wrong** (arktts /
Audio8-TTS, 2026-07-30 — each one caught only by running the framework's own loader, per the
SKILL's "the parity harness is NOT the bar" rule):

- **`post_load_hook` is how you get the model directory.** `base_load_model` constructs
  `Model(ModelConfig)` and calls `sanitize`, but the config carries no path, so anything needing
  the checkout — a tokenizer, a voice bank, a sidecar codec — has nothing to open. The house
  mechanism is an optional classmethod the loader calls last:
  ```python
  @classmethod
  def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
      model.model_path = Path(model_path)      # tokenizer loads lazily off this
      return model
  ```
  Without it the model works in your harness (where you set the path by hand) and throws through
  `mlx_audio.tts.utils.load`. Precedent: `spark.py` builds its tokenizer + audio tokenizer there.
- **`sanitize()` runs on ALREADY-CONVERTED weights too, so make it idempotent.** The loader calls
  it unconditionally, including on the repo you published *after* sanitizing. Key remaps and conv
  transposes then apply twice — a double transpose is shape-plausible and silently wrong. Cheapest
  guard is a prefix/shape check that returns the dict untouched:
  ```python
  if all(k.startswith(("model.", "codec.")) for k in weights):
      return weights          # already converted
  ```
- **Publish PRE-sanitized weights when a Swift consumer is coming.** Folding weight norm and
  fixing conv layout at conversion time means the Swift loader needs no folding, no transposes,
  and no key remapping — just a dotted-path → module-tree walk. It moves the fiddliest, most
  error-prone half of the port out of the second implementation entirely.

## The mlx-lm CONTRIBUTING checklist (verbatim)

Source: https://github.com/ml-explore/mlx-lm/blob/main/CONTRIBUTING.md.

1. **Fork and clone** `ml-explore/mlx-lm`, then `pip install -e ".[dev]"` from the fork root.
2. **Pick a similar existing model** as your starting point. For a Llama-family architecture, start from `mlx_lm/models/llama.py`; for a Qwen-family, `qwen2.py`; for an MoE, `mixtral.py`.
3. **Add the model file** at `mlx_lm/models/<model_type>.py`. The filename must equal the `model_type` in the upstream `config.json`.
4. **Derive layer names** by one of:
   - Reading the Transformers implementation if you know the codebase.
   - Loading the weights and listing keys: `weights = mx.load("model.safetensors")`; print `list(weights.keys())`.
   - Reading `model.safetensors.index.json` in the HF repo (it lists every key without downloading the shards).
5. **Add a test** for the new `model_type` at `mlx_lm/tests/test_models.py`. The pattern is a small shape-and-config smoke test; full numerical parity is your local responsibility before PR.
6. **Install pre-commit** (`pre-commit install`) — it runs black on Python and clang-format on any C++. Don't disable hooks.
7. **Run the converter against your fork**: `mlx_lm.convert --hf-path <upstream> --mlx-path /tmp/test-port`. Verify `load()` round-trips, then run `generate()`.
8. **PR upstream.** Title: `[<model_type>] Add support`. Description includes: link to upstream HF repo, your test output, peak memory on the model size you tested, and any architectural notes (MoE? GQA? non-standard RoPE?).

The mlx-vlm CONTRIBUTING.md (https://github.com/Blaizzy/mlx-vlm/blob/main/CONTRIBUTING.md) is structurally identical — replace single file with sub-package, add a vision-side smoke test.

## Don't forget — the common pitfalls still apply

The pitfalls in `common-pitfalls.md` are framework-agnostic. They bite manual ports too:

- Constructor defaults must match the **trained** config, not the framework default. (#1)
- `attention_head_dim` misnomer in diffusers — still a trap when porting a VLM's vision tower if it's a diffusers UNet. (#2)
- Fused QKV `cat→view→chunk` interleaving — handle inside `sanitize(weights)` if upstream stores fused, or run the interleaved reshape at forward time. (#3)
- Conv `(O, I, *K)` → `(O, *K, I)` — transpose inside `sanitize()`. Vision sub-packages need this constantly. (#4)
- Norm semantics, `qk_norm`, activation choice — every config flag → field on `ModelArgs`. (#5, #6)
- Tekken / Pixtral tokenizer skipping BOS — check `tokenizer_config.json` `add_bos_token` before upload. (#8)

The official converters call `Model.sanitize(weights)` automatically before saving, so put **all** weight remapping in `sanitize()` — never in the constructor, never at first forward pass. That keeps weights serialization-safe (no lazy-zero bug, see weight-conversion.md) and keeps the converter pure.
