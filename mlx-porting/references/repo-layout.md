# Standard `-mlx` Fork Repo Layout

Convention developed across LTX-2, Hunyuan3D-2.1, CogVideoX-Fun, Matrix-Game and others. Following this layout makes every port look the same — contributors can orient quickly, tests find the right paths, and `mlx-forge` recipes plug in predictably.

## Naming

- GitHub fork / repo: `<OriginalModelName>-mlx`. Example: `Hunyuan3D-2.1-mlx`, `LTX-2-mlx`, `CogVideoX-Fun-mlx`.
- Python package: `<original_model_name>_mlx` (snake_case). Example: `hunyuan3d_mlx`, `ltx_core_mlx`.
- HF weights repo: `<your-hf-user>/<original-model-name>-mlx`.

## Single-package layout (smaller models)

For models with a single main component + VAE + text encoder. Example: Qwen-Image, Mistral-Small.

```
<model>-mlx/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── <model>_mlx/
│   ├── __init__.py
│   ├── pipeline_mlx.py              # high-level from_pretrained entry
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py           # main module
│   │   ├── attention.py             # attention blocks
│   │   ├── vae.py
│   │   └── text_encoder.py
│   ├── config.py                    # dataclasses mirroring source config.json
│   ├── scheduler.py                 # if applicable
│   └── utils/
│       ├── __init__.py
│       ├── weights.py               # load_split_safetensors, HF download
│       └── memory.py                # aggressive_cleanup, peak_memory helpers
├── tests/
│   ├── parity/                      # PT vs MLX (torch = optional dep)
│   ├── smoke/                       # shapes / config / e2e no-numeric
│   └── fixtures/                    # golden npy / png
└── examples/
    ├── minimal.py
    └── README.md
```

## Monorepo layout (complex pipelines)

For pipelines with multiple independent packages (core model, pipelines, optional trainer). Example: LTX-2-mlx.

```
<model>-mlx/
├── README.md
├── pyproject.toml                   # workspace / root
├── packages/
│   ├── <model>-core-mlx/
│   │   ├── pyproject.toml
│   │   └── src/<model>_core_mlx/
│   │       ├── model/
│   │       │   ├── transformer/
│   │       │   ├── video_vae/
│   │       │   ├── audio_vae/
│   │       │   └── upsampler/
│   │       ├── conditioning/
│   │       ├── loader/
│   │       │   ├── sd_ops.py        # safetensors ops
│   │       │   └── fuse_loras.py
│   │       ├── text_encoders/
│   │       │   └── gemma/
│   │       └── utils/
│   │           ├── weights.py
│   │           └── memory.py
│   ├── <model>-pipelines-mlx/
│   │   └── src/<model>_pipelines_mlx/
│   │       ├── ti2vid_t2v.py
│   │       ├── ti2vid_i2v.py
│   │       ├── scheduler.py
│   │       └── cli.py               # `<model>-mlx` command
│   └── <model>-trainer/             # optional
│       └── src/<model>_trainer_mlx/
│           ├── trainer.py
│           └── training_strategies/
└── tests/
    ├── <model>-core-mlx/
    ├── <model>-pipelines-mlx/
    └── <model>-trainer/
```

## HF auto-download pattern

Standard single entrypoint:

```python
# <model>_mlx/pipeline_mlx.py
import os
from huggingface_hub import snapshot_download

class Pipeline:
    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        local_dir = os.environ.get(
            f"{cls.ENV_NAME}_MLX_WEIGHTS_DIR",  # e.g. HUNYUAN3D_MLX_WEIGHTS_DIR
            snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.txt"]),
        )
        return cls._load_from_dir(local_dir, **kwargs)
```

- `<MODEL>_MLX_WEIGHTS_DIR` env var overrides the HF download — for local dev / offline use. Standardize this across ports.
- Use `snapshot_download` with `allow_patterns` to skip PyTorch-format weights if they exist in the repo.
- Lazy-load safetensors via `mx.load` — don't read into host memory first.

**Existing inconsistency:** LTX-2-mlx uses a conftest-based lookup instead of an env var. New ports should standardize on the env-var pattern above.

## pyproject.toml template

```toml
[project]
name = "<model>-mlx"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.30",
    "mlx-arsenal>=0.1",
    "huggingface-hub>=0.20",
    "safetensors>=0.4",
    "numpy",
    "Pillow",       # if image I/O
]

[project.optional-dependencies]
parity = ["torch>=2.3", "diffusers>=0.30"]
dev = [
    "pytest",
    "pytest-xdist",
    "<model>-mlx[parity]",
    "ruff",
]
forge = ["mlx-forge"]  # only if users need to re-convert from PT

[project.scripts]
"<model>-mlx" = "<model>_mlx.cli:main"   # if a CLI exists
```

## README.md sections (in order)

1. **Header** — one-line description, badges (Python, MLX, HF repo link).
2. **Features** — what works, what doesn't (be honest — "Stage 1 bit-exact, Stage 2 within PSNR 32 dB").
3. **Requirements** — Apple Silicon M-series, Python ≥ 3.11, macOS version, expected RAM.
4. **Installation** — `pip install <model>-mlx`, optional `[parity]` / `[forge]` extras.
5. **Quick Start** — shortest possible working example (CLI + Python code).
6. **Model Card / Architecture** — brief description, link to original paper / repo.
7. **Converting Weights Yourself** — link to mlx-forge + recipe name (if applicable).
8. **Performance** — table of peak memory / wallclock on M2, M3, M4 for representative inputs.
9. **Known Limitations** — things that don't match reference (e.g. `texture_size=2048` due to Metal command-buffer budget).
10. **Citation** — original paper BibTeX.
11. **License** — match upstream.

## .gitignore essentials

```
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
.venv/
build/
dist/
# generated
*.safetensors
*.mlx
!tests/fixtures/*.safetensors  # keep small golden fixtures
# large media
*.mp4
*.png
!tests/fixtures/*.png
# user dirs
weights/
outputs/
```

## CI (GitHub Actions)

Minimum pipeline:

- **Lint** (ruff).
- **Type check** (mypy or pyright, optional).
- **Smoke tests** (shape + config, no numerics) — run on every push.
- **Parity tests** (needs torch) — run on PR to main only, gated behind `[parity]` extra install.
- **E2E** (downloads weights, runs small golden input) — nightly, not per-push.

Self-hosted macOS runner required — GitHub's Linux runners can't exercise MLX.

## Release convention

- Tag semver: `v0.1.0`, `v0.2.0`, etc.
- Pin MLX minor version in `pyproject.toml` — MLX numerics shift between minor releases; don't let users surprise themselves on upgrade.
- HF weights repo versioned via branches: `main` (latest), `v0.1.x` (compat branch).
- Release notes should always include: MLX version tested, peak memory measured, any breaking API changes.

## Swift consumer side — `xocialize/<package>-mlx`

The published HF repo (whether mlx-community or a private repo under `xocialize`) must conform to the layout `mlx_lm.convert` / `mlx_vlm.convert` / `mlx_audio.convert` produce so Swift consumers (`mlx-swift-examples`, `mlx-audio-swift`) load it with zero custom code.

**Swift package layout** (per Dustin's environment convention):

- Top-level **Xcode workspace** (`<Package>.xcworkspace`) at the repo root.
- `Package.swift` is the **dependency manifest only**. Sub-projects and packages organize beneath the workspace.
- Build via `xcodebuild` (Xcode is the build tool for anything touching MLX, CoreML, Metal, or signing — SwiftPM CLI is **not** used).
- CI runs `xcodebuild` against the workspace; self-hosted Apple Silicon runner required.

Reference layout: https://github.com/xocialize/lance-mlx (single-stack model, references `mlx-community/Lance-3B-bf16`) and https://github.com/xocialize/ltx-2-mlx (multi-component pipeline, hosts its own weights).

**MLX-Swift consumer idioms** (apply on the Swift side when loading published weights):

- **Canonical weight-load three-step:** `MLX.loadArrays(url:)` → `ModuleParameters.unflattened(loaded)` → `model.update(parameters:..., verify: .noUnusedKeys)`. The `verify: .noUnusedKeys` is what surfaces a remap bug as a test failure instead of silently zero-initializing a layer.
- **Safetensors-key ↔ Swift-property remap.** MLX-Swift's `Module` reflects Swift property names, which can't contain dots. Upstream `mlx-lm` / `mlx-vlm` safetensors record indexed lists as `block.0.weight`, `block.1.weight`, … but the Swift port exposes them as `block_0`, `block_1`, … as explicit named properties (you can't write `block.0` in Swift). Remap dotted keys at load time before `unflattened`:
  ```swift
  let remapped = Dictionary(uniqueKeysWithValues: loaded.map { (k, v) in
      (k.replacing(/(\w+)\.(\d+)\./, with: { "\($0.output.1)_\($0.output.2)." }), v)
  })
  ```
  Concrete past instance: ForgeOptimizer's LiteFlowNet port — safetensors `matching.2.…`, Swift property `matching_2.…`. Without the remap, `verify: .noUnusedKeys` either fails (good — caught at boot) or silently no-ops (bad — module runs with uninitialized weights).
- **GPU-state classes are `@unchecked Sendable` with manual locking.** Anything holding `MLXArray` cached state, an `MTLCommandQueue`, or a long-lived `MLXFastKernels` handle gets `@unchecked Sendable` + an internal `NSLock` (or `os_unfair_lock`) around the mutable region. Strict-concurrency-checked `Sendable` will fight you otherwise — MLX-Swift arrays aren't Sendable-marked upstream and the unified-memory pointers move under you.
- **SPM CLI cannot compile Metal shaders — workaround when you need `swift test`.** `swift run` / `swift test` against an MLX-Swift package fail with `MLX error: Failed to load the default metallib. library not found`. The build tool of record is `xcodebuild`; if a CI lane genuinely needs SPM CLI (e.g. cross-target test that doesn't touch GPU), seed Xcode DerivedData by building any workspace target from Xcode once, then copy the bundle into the SPM build dir and rename the metallib so the MLX runtime finds it:
  ```bash
  cp -R "$HOME/Library/Developer/Xcode/DerivedData/<Workspace>-*/Build/Products/Debug/mlx-swift_Cmlx.bundle" \
      .build/arm64-apple-macosx/debug/
  mv .build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib \
     .build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle/Contents/Resources/mlx.metallib
  ```
  Hit by every Swift MLX port that has tried `swift test`: Demucs-mlx-swift, Mel-RoFormer-mlx-swift, Qwen3-TTS, anime-studio root.

**Mirror policy** — the gist: single-stack LLM/VLM/audio weights live in `mlx-community/<name>-<quant>` and the `xocialize/<package>-mlx` Swift package references them via `from_pretrained`; multi-component diffusion pipelines host their own weights in the `xocialize/<model>-mlx` repo (no clean single-`model_type` slot in mlx-community). `mlx-community-conventions.md` "Mirror policy" is the single source of truth, including the publish-to-both bf16 case — defer there rather than duplicating the rule.

## Tier-3 pipeline-mlx repos

For multi-component pipelines (LTX-2, Qwen-Image, Hunyuan3D-2.1, Matrix-Game), the Python side uses the monorepo layout above; the Swift side mirrors with an Xcode workspace at the top, `Package.swift` listing MLX-Swift + the Python-side HF repo as the weight source. See `xocialize/ltx-2-mlx` for the canonical pattern.
