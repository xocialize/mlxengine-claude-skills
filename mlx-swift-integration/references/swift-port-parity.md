# Stage 1 — The Python-MLX → Swift-MLX port itself (workflow + numerics doctrine)

`porting-conformance.md` defines what a finished Stage-1 package looks like; THIS file is how you
get there. Distilled from the `bernini-r-mlx-swift` port (2026-06-12: Wan2.2-A14B dual-expert
t2v/t2i/r2v/v2v/rv2v — S0–S6 all parity-locked in one pass, samplers bit-exact), which itself
built on `lance-mlx-swift` / `qwen25vl-mlx-swift` / `longcat-avatar-mlx-swift`. The `mlx-porting`
skill's doctrine (isomorphism, CPU-stream parity, fixture injection) carries over — this file is
the Swift-side delta.

## The big advantage: same array semantics, same RNG

Python-MLX → Swift-MLX is a far gentler port than PyTorch → MLX: both bindings wrap the SAME
mlx::core. Exploit that:

- **RNG seed streams are bit-identical across bindings** (verified: `mx.random.seed(42)` +
  `normal(shape)` produces byte-equal tensors in Python and Swift). `seed:` parameters are
  cross-binding-reproducible; keep fixture *injection* as belt-and-braces, but you can mirror
  Python signatures that seed internally without redesigning them. Pin this with a one-shot test
  (dump a seeded normal from Python; assert max_abs == 0 in Swift).
- **Scalar-coefficient code can be bit-exact.** Swift `Double` == numpy float64: schedulers
  (UniPC incl. its Gaussian-elimination corrector solves) and full sampler trajectories achieved
  **max_abs 0.0** vs the oracle. If a scheduler/sampler gate shows ~1e-6 drift, suspect a
  float32-vs-float64 spot in YOUR translation, not "numerics".
- Compute frequency tables (rope, inv_freq) in Swift `Double` loops before the fp32 cast,
  mirroring numpy-float64-then-astype — don't build them with fp32 MLX ops.
- Expect: component forwards on real weights ≤ the mlx-porting thresholds with big margin;
  4-step e2e goldens ≤ 0.05; quant cosines agreeing with Python to the 4th decimal.

## Phase-gated workflow (the S0–S7 pattern)

One phase = one parity gate = one commit. Each phase independently shippable; the PORTING-SPEC.md
in the repo carries the phase table with PASSED stamps.

```
S0  key contract        generated key sets == actual safetensors headers (all variants)
S1  substrate           per-component forwards vs oracle fixtures (real weights, CPU stream)
S2  core pipeline       few-step e2e golden, injected noise/contexts (both experts/branches hit)
S2b prompt entry + GPU  tokenizer wiring + ONE real generation on GPU — the eyeball gate
S3+ model deltas        the novel surfaces, each with oracle fixtures + a no-op identity test
S5  memory machinery    streaming/chunked paths gated BIT-IDENTICAL vs the plain path
S6  quantized variant   per-pass cosine vs the ORACLE'S OWN gate (cross-validate same-fixture)
S7  engine wrap         → porting-conformance.md / integration-lessons.md
```

Fixture generation is a `tools/dump_*.py` script run with the ORACLE'S venv, dumping `.npy`
(bf16 saved as fp32; ids as int32) into the test resources. Promote the minimal `.npy` reader
into the main target so CLI gates share it.

## S0 — the key contract (before any model code)

- Read the actual safetensors **headers** (8-byte length + JSON; pure Foundation — no MLX) and
  pin: per-component key sets, dtypes, config-derived shape spot-checks, for EVERY published
  variant (bf16 + quantized). Generate regular key sets programmatically (blocks × paths +
  globals); pin irregular ones (VAE sequential indices) as a fixture file.
- The converted checkpoint's `config.json` is usually the fully **resolved** config (the
  conversion wrote it) — decode it into one Codable struct and treat its values as oracle truths.
- Quantized variants: expect `.scales`/`.biases` only on the recipe's quantized scope, and
  expect **stray serialized buffers** (e.g. a rope `freqs` table the Python quantize script
  picked up). Handle via explicit `toleratedExtras` in the loader — and remember to SUBTRACT the
  tolerated key from the expected set, or the contract check contradicts itself.
- Loader rule (refuse-partial-loads): after dropping tolerated extras, on-disk keys must equal
  the expected set — 0 missing / 0 unused — or throw. `update(verify: .noUnusedKeys)` alone
  tolerates unset module params (see the silent-failure class in `integration-lessons.md`).

## Donor reuse: verify key paths, not architecture

A parity-proven Swift implementation of the same architecture (a donor) is a huge accelerator —
but **architecture match ≠ key-name match**. Decide lift-vs-translate by comparing the donor's
flattened parameter paths against YOUR checkpoint's key contract:

- Donor paths match the checkpoint → **lift** (keep its debugged-the-hard-way choices verbatim).
- Donor uses a different naming lineage (e.g. diffusers `norm1/conv1/conv_shortcut` vs
  mlx-video sequential `residual.0/.2`, `gamma`) → **translate 1:1 from the Python upstream**
  instead, and use the donor only as the Swift-idiom reference (cache patterns, cpu-stream
  spots, MLXFast usage). Native key match beats a remap layer.
- Donor *conventions* may not transfer either: check the reference's LOAD-TIME dtype policy
  (mlx-video upcasts the T5 to fp32 at load; a donor running it bf16 is not your convention).

The structural gate that makes all this cheap: **instantiate the module weight-free and assert
`Set(parameters().flattened().map(\.0)) == contract`**. Lazy init means no eval, no Metal, runs
in milliseconds offline. Keep non-parameter buffers (rope tables, inv_freq) in a plain holder
class so Module reflection can't see them — or they pollute the key set.

## The Metal-watchdog family (≈10 s command-buffer ceiling)

One root cause, four disguises met in a single port. The rule: **a Metal command buffer must
never wait on slow non-GPU work** (disk IO, long CPU-stream chains, paging under memory
pressure). `kIOGPUCommandBufferCallbackErrorTimeout` is the watchdog firing.

1. **Weight loads ride the CPU stream.** Lazy `Load` ops bind to the stream current at
   *creation*; on the GPU stream a multi-GB read holds one buffer open past the watchdog.
   Wrap `loadArrays` + chunked eval in `Device.withDefaultDevice(.cpu)`. Chunked GPU-side eval
   is NOT sufficient on a slow disk. **Corollary — materialize a dtype UPCAST before the
   forward.** If you `loadArrays(...).mapValues { $0.asType(.float32) }` (e.g. a bf16 weight
   file run as the production fp32 DiT, or an fp32-parity umT5), the upcast is *lazy*: without
   an `eval(model)` after `update`, the multi-GB cast folds into the first forward's command
   buffer and times out the watchdog. `eval` the model right after load. (Helios S2b: an
   11 GB umT5 bf16→fp32 upcast un-`eval`'d → timeout on the very first encode.)
2. **Quantized forwards run wholly on the GPU stream.** Quantized matmuls route to Metal even
   under a CPU pin, so a CPU-pinned quantized graph becomes one Metal buffer fenced on CPU ops
   at every block. (For a quant-quality cosine gate this is fine doctrinally: GPU float noise
   ~1e-3 is negligible against int4 error at a ≥0.99 gate.) **Load on CPU stream, but run the
   forward OUTSIDE the CPU pin** — wrap only `applyQuantization`+`loadArrays`+`update`+`eval` in
   `withDefaultDevice(.cpu)`, return the model, then call it on the default (GPU) stream.
   **Symptom if you don't:** at small gate seqLen the per-block CPU fence does NOT trip the
   watchdog — it just **grinds** (process state `R`, 100+ min CPU time, zero output), which
   masquerades as a hang or a reaped task. Triage with `ps -Ao pid,stat,etime,time` — `R` +
   huge CPU time = alive-but-CPU-pinned (not `Z`/reaped, not `S`-QoS-starved). Helios S6 lived
   this exactly: CPU-pin → 100 min no output; GPU → seconds, int4-vs-bf16 cosine 0.9965.
3. **Never eagerly eval giant constant fills.** Zero-filling params before `MLXNN.quantize`
   (shapes are all that matter — every value is replaced at load) is correct; *evaluating* those
   zeros materialized ~57 GB fp32 per expert and produced a 161 GB swap-storm whose paging
   stalls also read as the watchdog.
4. **Scope big models so ARC frees them.** Holding a bf16 expert while constructing its int4
   sibling (plus init buffers) is memory pressure; wrap phase-1 in a closure returning only the
   eval'd output, then `GPU.clearCache()`.

**Telling the disguises apart at bring-up.** All four present as "<10 % GPU, long wall-clock," so
don't guess — run the smoke under `MLX_PROFILE=1` (the shared `MLXProfiling` instrument,
`MetalToolBox/PROD/mlx-profiling`, `mlx-swift`-only, zero overhead when unset). Its live
`[MLXPROF]` rows separate **compile** (one slow first step, flat memory) from **paging**
(⚠PAGING flag = `phys_footprint > GPU.maxRecommendedWorkingSetSize`, cache balloons) from
**encoder-stall** (fires AFTER the pipeline returns — instrument the encode too). Read `phys_footprint`,
not `Memory.peakMemory` (which counts cumulative allocs and misleads under a `cacheLimit` cap). Adopt it
rather than hand-rolling a per-package profiler — it's the same instrument used for the manifest footprint
and the efficiency sweep (see `memory-harness.md`, `package-efficiency.md`).

Also inherited from mlx-porting, still true in Swift: per-step `eval` + cache clearing in
denoise loops; chunked materialization at natural boundaries.

## Where gates can actually run (this environment)

- **Plain `swift run` executables DO GPU inference** — the metallib resolves from the products
  dir via mainBundle. The old workspace-wide "live inference under Xcode only" rule is stale for
  current mlx-swift. Real generations, GPU smokes, and heavy gates all work as CLI runs.
- **The SPM test product is the fragile context.** The nested `mlx-swift_Cmlx.bundle` assembly
  can corrupt (`missing creator for mutated node` build warning) and STAY corrupt across product
  wipes; then every MLX-touching test dies on "Failed to load the default metallib" — including
  CPU-only suites, because mlx initializes Metal on first op regardless of the default device.
  A global `Device.setDefault(device: .cpu)` pin does not avoid the init.
- **Therefore: every Metal-context gate is a CLI mode of the executable target** —
  `swift run RunModel --s4-gate` etc., reading fixtures from the source tree. Keep the
  swift-testing variants env-gated as repros, but the CLI is canonical. Tests that never eval
  (key-path/structural/scheduler-scalar) stay in `swift test` happily.
- **Ops discipline for heavy gates:** launch detached (`nohup … & disown` — harness-tracked
  tasks die with the panel); `print` to a redirected file is block-buffered, so progress/bisect
  markers go to **stderr** via `FileHandle.standardError`; debug-build CPU forwards are several×
  slower than release — a "20 min" gate can be 45+ (build the gate `-c release` when iterating);
  when a crash location is unclear, bisect with staged stderr markers before theorizing.

## Gate thresholds: use the oracle's own gates, cross-validate same-fixture

When your measured number differs from a published one (int4 cosine 0.9977 vs the oracle's
published 0.9992), don't loosen by judgement — (1) find the oracle's actual GATE (here: ≥ 0.99,
in its quantize script docstring), and (2) run the oracle's measurement **on your identical
fixture** (a 20-line Python script in `tools/`). Same-fixture agreement to the 4th decimal
converts "we loosened a gate" into "we verified equivalence; the published number was a
different input distribution."

## Misc Swift-side gotchas (this port)

- Published weight repos may ship NO tokenizer files — fetch from the canonical upstream repo
  (e.g. `google/umt5-xxl`, exactly what mlx-video loads); swift-transformers `AutoTokenizer`
  pulls only the tokenizer files.
- `@main` cannot live in `main.swift` (rename the file when adding an entry-point struct).
- Mirror the reference's eval-laziness: structural/key-path tests never eval, so they run under
  plain `swift test` even when the metallib is unavailable.
- A 1:1 translation should let a reader diff the Swift file against the Python file and see only
  syntax. Same file/class/function decomposition, same constants, same comments where they carry
  constraints. (The mlx-porting isomorphism rule, unchanged.)

### Key-naming idioms — the checkpoint key is the contract (scail-2-mlx-swift, 2026-06-20)

The fastest way to break a clean weight load (`verify: .all`, 0 missing/0 unused) is a Swift
container whose flattened keys don't match the checkpoint. Three idioms that recur:

- **`MLXNN.Sequential` for `mlp.layers.N` keys**, NOT dot-keyed `@ModuleInfo(key: "layers.0")`.
  A dot in a ModuleInfo key creates a *literal* component `layers.0` that does NOT round-trip
  through `ModuleParameters.unflattened` (which splits on `.` into `layers`→`0`) → keyNotFound at
  load. `Sequential { Linear; GELU; Linear }` produces native `layers.0`/`layers.2` (the GELU at
  index 1 carries no params), matching the Python `nn.Sequential` exactly. Same fix for
  `img_emb.proj.layers.{0,1,3,4}`, `text_embedding.layers.{0,2}`, `time_projection.layers.1`.
- **RMSNorm/any normalized weight must be a `Module`, not a bare `MLXArray`.** The checkpoint key
  is `self_attn.norm_q.weight`; a bare `@ModuleInfo var normQ: MLXArray` flattens to
  `self_attn.norm_q` (no `.weight`) → load fails. Wrap it: a 6-line `RMSNormW: Module` with a
  `weight` property loads under `<name>.weight`. (A gate that hand-bridges names — loading a
  `w_norm_q__weight` fixture into key `norm_q` — passes while hiding this; only a real-key load
  with `verify: .all` catches it. Always load by the module's OWN flattened keys.)
- **Parameterless-but-present norms = `Identity()`.** Python `WanLayerNorm(elementwise_affine=
  False)` has NO checkpoint keys; representing it as `Identity` under the same ModuleInfo key
  yields zero params (matches), where a real LayerNorm would invent phantom `weight`/`bias` keys.
  Only the affine norm (`norm3`) is a real `LayerNorm`.

### Two more cross-binding numerics notes

- **Pin the fixture DUMPER to `mx.cpu` too, not just the Swift gate.** `mx.fast`/`MLXFast` SDPA
  (and quantized matmuls) route to Metal even under a CPU default-device, so a GPU-dumped fixture
  carries the machine's reduced-precision SDPA noise (~1e-3 on M5). A correct translation then
  reads ~1.7e-3 and "fails" a 1e-3 gate; re-dumping on `mx.cpu` → bit-exact 0.0, no code change.
  Stream-match BOTH sides.
- **RoPE/freq tables: the oracle may be float32 where upstream + wan-core are float64.** If a
  freq-table reuse check drifts ~2e-4, suspect the Python-MLX oracle took a float32 shortcut
  (`mx.arange(...).astype(float32)`) while the canonical path (PyTorch `torch.polar`, wan-core
  `ropeParams`'s `Double` loop) is float64. float64 is correct (skill doctrine: build freq tables
  in `Double` before the fp32 cast) — align the FIXTURE to float64 rather than degrade the reused
  `ropeParams`; don't "match the oracle" into a less-accurate table.

### MLX-Swift 0.31.x API gaps & convention traps (trellis2-mlx-swift, 2026-06-26)

Porting a 6-core sparse-3D / DiT / ViT pipeline (sparse conv, decode ops, DiT blocks, VAE
decoders, DINOv3) mapped where mlx-swift 0.31.4 diverges from Python mlx 0.31.2 and where silent
numeric drift hides. All cores landed bit-identical (`0.000e+00`) or float-epsilon (~1e-7).

**API gaps — plan around these for ANY sparse/dedup/dynamic-topology port:**
- **No `unique`, no Swift-level scatter-add, no `mx.nonzero`/`argwhere`, no boolean-mask gather.**
  Only Cmlx `mlx_scatter_add` (C-only). For data-dependent / dynamic-size compaction (mesh
  topology, NMS, sparse downsample, token filtering) the idiom is **host-compaction**:
  `eval(x); let v = x.asArray(Float.self)` → compact in a plain-Swift loop, preserving the same
  ascending order Python's `np.where(mask)[0]` produces → re-wrap `MLXArray`. Exact (index work),
  and how the sparse decode ops port bit-identically.
- **`scatter-ASSIGN` IS available** (`a[idxArray] = values`, internal `mlx_scatter`) — so any
  lookup-table build over UNIQUE keys (sparse-conv neighbor map, flat-key hashmap) ports cleanly
  with assign, no scatter-add needed (coords-unique guarantees it).
- **`.broadcast(to:)` is `internal`** → use the FREE func `broadcast(arr, to: [Int])`.
- Column-slice `x[:, :k]` → `take(x, MLXArray([Int32](0..<k)), axis: 1)` (avoids 2-D range
  subscripts). `mx.repeat(x, n, axis)` → `repeated(x, count: n, axis:)`. Unstack `x[:, i]` →
  `x.split(parts:, axis:)` + `.squeezed(axis:)`. `mx.linalg.cross` lives in `MLXLinalg`.
- **Module products are SEPARATE SPM libraries** — add each to the target's `dependencies` as you
  reach for it: `MLX` (core ops, `conv3d`, `broadcast`), `MLXNN` (`silu`, `gelu`,
  `geluApproximate`), `MLXFast` (`rmsNorm`, `layerNorm`, `scaledDotProductAttention`), `MLXLinalg`
  (`cross`). `MLXFast.layerNorm`/`rmsNorm` accept `weight:nil`/`bias:nil` (≡ Python `None`).
- `conv3d` weight layout is `[C_out, D, H, W, C_in]` — transpose PyTorch `(Co,Ci,kD,kH,kW)` via
  `.transposed(0,2,3,4,1)`. `conv3d`/SDPA/rmsNorm on the CPU stream match Python `mx.*` bit-for-bit.

**Convention traps that drift ~1e-3 silently (a WRONG answer, not a crash):**
- **GELU has two forms ~1e-3 apart.** `nn.GELU(approx="precise"|"tanh")` = the TANH approximation =
  Swift `geluApproximate`. `nn.gelu` / `approx="none"` = the ERF form = Swift `gelu`. Match the
  reference's exact choice — the SAME codebase used precise/tanh in the DiT FFN and exact-erf in the
  DINOv3 MLP. Grep the activation before porting.
- **RoPE has two incompatible conventions, sometimes in ONE model.** Interleaved-pair (reshape
  D→(D/2,2), complex-multiply) vs rotate_half (`[-x2,x1]`, cos/sin tiled to full head_dim). Read
  the actual `apply_rope` — don't assume. RoPE may also apply to a token SUBSET (DINOv3 skips the
  CLS+register prefix): split at the prefix, rope only the rest, concat back.
- **Which LayerNorm matters.** A hand-rolled TWO-PASS LayerNorm (mean→center→var→rsqrt) is often
  chosen for PyTorch parity over `mx.fast.layer_norm` (single-pass, drifts ~1.4e-6/call, compounds
  over 90+ norms × many steps). `nn.LayerNorm` itself == `mx.fast.layer_norm`. Match whichever the
  reference uses; `SparseMultiHeadRMSNorm` = `MLXFast.rmsNorm(x, weight: ones(D))` then `× gamma`.

**Component-gate methodology (fast front-runner before the keyed Module):** for parity probes, port
each block as a FREE FUNCTION taking weights via a `[String: MLXArray]` dict
(`block(x, w: ["self_attn.to_qkv.weight": ...])`), NOT a checkpoint-keyed `nn.Module`. This gates
the MATH in isolation with zero key plumbing, and a full-block test transitively covers its sub-ops.
Dump goldens from the Python-MLX oracle pinned to `mx.cpu` (inputs+weights+outputs → JSON), gate
`max_abs < 2e-4`. The keyed-`Module` + `verify:.all` real-key load (above) is still required for the
SHIPPING package — the dict-functional probe just de-risks the translation first, with a runnable
`swift-probe` executable (CPU stream, no metallib) accumulating one `runCore<N>()` per block.
