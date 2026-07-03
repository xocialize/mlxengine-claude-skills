# mlx-community Conventions

This is the publishing contract that downstream Swift / Python consumers assume. If you upload to `mlx-community/<name>-<quant>`, they expect this exact repo layout, model-card frontmatter, and quant-suffix grammar. Deviate and the loader silently picks wrong defaults or fails on key remap.

## Repo naming

```
mlx-community/<UpstreamRepoName>-<quant>
```

Preserve case from the upstream repo. `Qwen/Qwen3-Coder-Next` → `mlx-community/Qwen3-Coder-Next-6bit`, not `qwen3-coder-next-6bit`.

## Quant suffix grammar

Use only these — they map to the official converters' CLI flags or learned-quant subcommands.

| Suffix | What it means | How to produce |
|---|---|---|
| `-4bit`, `-6bit`, `-8bit` | Uniform affine quant (default `group_size=64`) | `mlx_lm.convert -q --q-bits {4,6,8}` |
| `-bf16`, `-fp16` | No quant, just dtype conversion. **Pick by parity at that dtype, both ways:** narrow audio nets sometimes *need* fp16 (bf16 collapses them); **high-magnitude-activation nets (FFT/FFC, e.g. LaMa) REQUIRE bf16 — fp16 collapses them to garbage** (mean err 0.55 → 0.004 switching to bf16). See `parity-testing.md` "Choose the publish dtype". | `mlx_lm.convert --dtype bfloat16` (no `-q`) / `--dtype float16` |
| `-mxfp4`, `-nvfp4`, `-mxfp8` | Microscaling FP quant | `mlx_lm.convert -q --q-mode {mxfp4,nvfp4,mxfp8}` |
| `-MXFP4-Q4` | gpt-oss-style mixed (MXFP4 weights, INT4 outliers) | Custom per-component; see https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q4 |
| `-mixed_2_6`, `-mixed_3_4`, `-mixed_3_6`, `-mixed_4_6` | Mixed-bit recipes via predicate | `mlx_lm.convert -q --quant-predicate mixed_X_Y` |
| `-AWQ-4bit`, `-DWQ-4bit`, `-OptiQ-4bit` | Learned / calibrated quants | `mlx_lm.awq`, `mlx_lm.dwq`, `mlx_lm.dynamic_quant` |

**Never use:** `Q4_K_M`, `Q5_K_S`, `q4_0`, `IQ3_XS`, or any other llama.cpp / GGUF suffix. They're a different ecosystem and the mlx-community loader doesn't recognize them.

Per `LEARNED_QUANTS.md` (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md): dynamic quantization is the fastest to compute, DWQ takes longer but typically yields better quality. Pick DWQ if you have the time budget.

## Quant collections (group multi-variant publishes)

A model published at more than one quant should not leave its variants scattered across the org. Gather them into an **`mlx-community` Hugging Face Collection** so a user finds the whole family — `-4bit`, `-6bit`, `-8bit`, `-bf16`, learned quants — on one page instead of guessing which suffixes exist. This is "when appropriate" grouping: do it once you have two or more variants of the same base model; skip it for a lone one-off.

Create the collection under the **org namespace**, not your personal one, and add each repo as it's published:

```python
from huggingface_hub import create_collection, add_collection_item

# namespace="mlx-community" puts the collection on the org page (requires org write rights)
collection = create_collection(
    title="Lance-3B",
    description="MLX quant variants of Lance-3B (Wan2.2 MoT).",
    namespace="mlx-community",
)
for repo, note in [
    ("mlx-community/Lance-3B-bf16", "Reference precision; parity baseline."),
    ("mlx-community/Lance-3B-8bit", "Near-bf16 quality, ~half memory."),
    ("mlx-community/Lance-3B-4bit", "Smallest footprint; check parity at 5e-2."),
]:
    add_collection_item(collection.slug, item_id=repo, item_type="model", note=note)
```

Items are added one at a time; the optional `note` (≤500 chars) is the right place to state the quant tradeoff so a browsing user doesn't have to open each repo. You can also create the collection in the browser via **+ New** on the mlx-community org page and add items from each repo's dropdown.

**Permission caveat:** creating an *org* collection requires mlx-community membership with write rights. If you don't have them, the variants still publish fine under the naming grammar above — leave the collection uncreated, link them with a shared `base_model`, and flag that the grouping step is still owed.

## Model card frontmatter

`mlx_lm.convert --upload-repo` auto-generates this via `create_model_card()` at `mlx_lm/utils.py` lines 608–632. For hand uploads (bf16 audio / pipeline models, multi-component diffusion), reproduce the same shape:

```yaml
---
library_name: mlx
license: <copy from upstream>
license_link: <copy from upstream>
pipeline_tag: <text-generation | image-to-image | text-to-speech | image-text-to-text | ...>
base_model: <upstream-org>/<upstream-repo>
tags:
  - mlx
---
```

Older uploads (Mistral-7B-Instruct-v0.3-4bit era) used only `tags: - mlx` — accept that form when validating existing repos but produce the current form for new uploads.

## Model card body — LLM

```markdown
# mlx-community/<name>-<quant>

This model [mlx-community/<name>-<quant>](https://huggingface.co/mlx-community/<name>-<quant>) was
converted to MLX format from [<upstream-org>/<upstream-repo>](https://huggingface.co/<upstream-org>/<upstream-repo>) using mlx-lm version **<version>**.

## Use with mlx

\`\`\`bash
pip install mlx-lm
\`\`\`

\`\`\`python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/<name>-<quant>")
prompt = "hello"
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )
response = generate(model, tokenizer, prompt=prompt, verbose=True)
\`\`\`
```

## Model card body — VLM (swap the snippet)

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/<name>-<quant>")
config = model.config
formatted = apply_chat_template(processor, config, "Describe this image.", num_images=1)
output = generate(model, processor, formatted, ["image.jpg"], verbose=True)
```

## Model card body — TTS / STT / audio (swap the snippet)

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/<name>-bf16")
for result in model.generate(text="Hello, world."):
    audio = result.audio   # mx.array, write with soundfile or torchaudio
```

For STT, swap `mlx_audio.tts.utils` → `mlx_audio.stt.utils`.

**Audio model cards should include a Parity section** stating the task-specific metric vs the upstream PyTorch reference — SDR (dB) for separation, codebook-index match % for codec encoders, label-agreement % for classifiers. This is the audio analog of vision ports quoting PSNR / LPIPS. Concrete examples downstream consumers expect: `Mel-RoFormer-Kim-Vocal-2-bf16` quotes "66.08 dB SDR vs reference"; `Mel-RoFormer-ZFTurbo-fp16` documents why bf16 was rejected (parity dropped to 21.96 dB on a 33.7M-param model). The Parity line is what tells a user how close the published preset is to original quality.

## Mirror policy for xocialize packages

The Swift consumer side under `xocialize/<package>-mlx` should reference the mlx-community repo, not host its own copy:

- **Single-stack LLM/VLM/audio model** → `mlx-community/<name>-<quant>` is the canonical home, and `xocialize/<package>-mlx` (the Swift package) loads it via `from_pretrained`. Examples: Mel-RoFormer (`mlx-community/Mel-RoFormer-bf16` + `xocialize/mlx-mel-roformer-swift`).
- **Multi-component diffusion pipeline** → `xocialize/<model>-mlx` hosts the converted weights directly (no clean single-`model_type` slot in mlx-community). Examples: `xocialize/ltx-2-mlx`, `xocialize/lance-mlx`, `dgrauet/ltx-2.3-mlx` (HF).
- **Single bf16 publish of a model that mlx-community accepts** → publish to both, with mlx-community as primary and `xocialize` referencing it. This is Dustin's Lance pattern (`mlx-community/Lance-3B-bf16`, `mlx-community/Lance-3B-Video-bf16`, `mlx-community/Ming-omni-tts-16.8B-A3B-bf16`).

## Browser-side fallback

For one-off conversions where you don't want to set up the local environment: https://huggingface.co/spaces/mlx-community/mlx-my-repo. It runs `mlx_lm.convert` server-side and uploads to mlx-community for you. Useful for LLMs only; doesn't cover VLM / audio / diffusion.
