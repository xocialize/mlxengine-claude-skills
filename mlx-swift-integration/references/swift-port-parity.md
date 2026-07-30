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

**The one place "same semantics" is a LIE — subscripts. `.newAxis` needs EVERY axis named, and
silently no-ops otherwise** (Audio8-TTS, 2026-07-30). Python/NumPy imply trailing axes; Swift-MLX
does not. On a 3-D array:

```swift
r[.newAxis, 0..., .newAxis]              // → shape UNCHANGED [8,32,2]. No error. No warning.
r[.newAxis, 0..., .newAxis, 0..., 0...]  // → [1,8,1,32,2]  ✅ every real axis named
r.expandedDimensions(axes: [0, 2])       // → [1,8,1,32,2]  ✅ and unambiguous
```

Every `[:, None]` / `[None, :]` broadcasting idiom you transcribe from the Python rung is a
candidate. Two failure modes, both bad:

- **Shapes still broadcast** ⇒ wrong numbers, no error at all.
- **Shapes don't** ⇒ a broadcast error at the CONSUMER, pointing at innocent code
  (`[broadcast_shapes] Shapes (1,8,16,32) and (8,32) cannot be broadcast` was thrown three call
  frames away from the bad subscript, inside SDPA).

**Rule: port `[:, None]` as `expandedDimensions(axis:)`, never as a `.newAxis` subscript.** Keep
`.newAxis` subscripts only where the list demonstrably covers every axis (usually 1-D operands).
When unsure, probe it — a 5-line `swift run` that prints two shapes costs seconds; this cost an
hour at the S0 gate and would have been silent numerical damage if the shapes had happened to
broadcast.

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

> 🚨 **Write the PASSED stamp AFTER the run, never while scaffolding the table — and the same for
> every "measured" number.** Scaffolding a spec invites filling in plausible results (the phases
> are known, the thresholds are known, the numbers "will be" fine). Two instances in one port
> (Audio8-TTS, 2026-07-30): a phase table stamped `PASSED 2026-07-30 (…102/102 frames…)` before a
> single gate had run — with invented per-gate figures — and manifest footprint constants
> commented `MEASURED via --validate` when they were arithmetic on weight bytes. Both were caught
> and corrected by the author, but nothing in the process would have caught either: the numbers
> were *individually plausible*, the tests built on them passed (a footprint assertion happily
> validates a guess), and a reviewer reads a spec as a record, not a forecast.
>
> The spec and the manifest are **documents of record** — their entire value is that an entry
> means someone ran the thing. Concretely: seed new rows as `pending` / constants as
> `PROVISIONAL — not yet measured`, and when a number is missing, **go build the measurement**
> (that is where the `--validate` harness came from) rather than softening the wording. If you
> catch yourself typing a result you have not read out of a tool result, that is the signal.

## The gate matrix must span the input envelope — largest production grid + decoded output per tier

The S1/S2/S2b fixtures are generated once at whatever size keeps them small — and that silently
defines the port's validated envelope. Position-dependent machinery (RoPE scaling, pos-embed
interpolation, vision-grid indexing) runs a DIFFERENT code path at larger grids, so small-grid
cosines validate nothing above them. The qwen3vl-mlx-swift conditioner was cos 0.998 at its
576-token golden grid (768² input) and **0.84 at the ~1024-token grid** (1024² input) — filed as
"minor residual," it produced glitch-banded Boogu edits that only surfaced months later in
in-app validation. Full doctrine (incl. the Anima 512² case) → `mlx-porting`
`references/parity-testing.md` ("largest production grid"). Swift-side rules:

- **Fixture matrix includes the max grid the product will run** (max resolution / max
  vision-token count / max sequence), not just the golden-friendly small one.
- **S2b's eyeball gate runs per shipping resolution tier** — one decoded image at each tier the
  wrapper's configs expose, not one image total.
- **A cosine that sags monotonically with grid size is a structural bug** — blocker until
  root-caused, never "not catastrophic" (conditioning error is amplified by the denoise loop).
- **Record the validated input envelope** (e.g. "edit inputs ≤ ~600 vision tokens / ≤0.6 MP") in
  the workspace APP-VALIDATION.md so app-side testers know where the tested region ends — the
  in-app failure was simply the first time anyone ran a 1 MP input.

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

## The NAX split-K GEMM bug (mlx-swift ≤ 0.31.6, M5-class GPUs) — check FIRST for "bf16 breaks above a size"

Before debugging any Swift-side "bf16/fp16 garbage or NaN that switches on above a token/resolution threshold (edit paths first)": mlx-swift builds with `MLX_METAL_JIT=ON`, and its JIT path mis-instantiates `steel_gemm_splitk_axpby_nax` with the output dtype (mlx#3797, fixed upstream by #3810, in no release ≤ 0.31.6). Dispatch window: half precision, M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N) — in practice only the FFN **down-projection** (K = 4×hidden). Python MLX is AOT-compiled and unaffected, so "works in the Python rung, breaks in Swift" fits this bug.

- Fix: **row-chunk that one GEMM at ≤896 rows** (output rows independent → exact; bf16 stays ~2× faster than fp32). Do NOT flip the whole model to fp32. Reference impls: `qwen3vl-mlx-swift` `MLP.downProjected`, `mage-flow-swift` `MageFeedForward.downProjected`, `boogu-image-swift` `LuminaFeedForward.downProjected`.
- Probe after every mlx-swift bump (`--nax-probe` gate modes; strict thresholds — near the boundary corruption is cos ≈ 0.998, NOT NaN); on PASS delete the chunks. `boogu-image-swift/tools/check_mlx_swift_3810.sh` checks a tag's vendored mlx.
- Full registry entry + history (this one bug stranded LTX → Boogu → Mage as separate mysteries): mlx-porting skill, `common-pitfalls.md` #35.

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
2. **Quantized forwards run wholly on the GPU stream. ⚠ HIGHEST-COST MEMBER — recurs.** This one
   has now bitten twice (Helios 100 min, Z-Image **10 hours**) precisely because it does NOT error
   and does NOT trip the watchdog — the default "pin the whole gate to `.cpu` like the fp32 gates"
   instinct is the trap. Quantized matmuls route to Metal even under a CPU pin, so a CPU-pinned
   quantized graph becomes one Metal buffer fenced on CPU ops at every block. (For a quant-quality
   cosine gate the GPU is fine doctrinally: GPU float noise ~1e-3 is negligible against int4 error at
   a ≥0.99 gate.) **Load on CPU stream, but run the forward OUTSIDE the CPU pin** — wrap only
   `applyQuantization`+`loadArrays`+`update`+`eval` in `withDefaultDevice(.cpu)`, return the model,
   then call it on the default (GPU) stream. And since the SPM **test target's** metallib is
   unreliable for GPU, put the quant gate in the **CLI lane** (`swift run … --quant-gate`), not an
   XCTest (Z-Image's P7 gate is a CLI subcommand for exactly this reason). **Symptom if you don't:**
   at small gate seqLen the per-block CPU fence does NOT trip the watchdog — it just **grinds**
   (process state `R`, 100+ min CPU time, zero output), which masquerades as a hang or a reaped task.
   Triage with `ps -Ao pid,stat,etime,time` — `R` + huge CPU time = alive-but-CPU-pinned (not
   `Z`/reaped, not `S`-QoS-starved). Helios S6: CPU-pin → 100 min no output; GPU → seconds,
   int4-vs-bf16 cosine 0.9965. Z-Image P7: CPU-pin → 10 h `R` at 99% before kill; moved to the GPU
   CLI lane → int8 cos 0.9998 / int4 0.976 in seconds.
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

## Window attention (Swin family): both porting errors are SHAPE-SAFE — gate the TABLES (SCUNet, 2026-07-27)

Porting `WMSA` (cyclic roll + shifted windows + relative-position bias) is the first genuinely new
shape in the image fleet, and it unlocks the whole Swin family. What makes it different from a
conv-net port is that its two characteristic mistakes **produce a plausible image rather than an
error** — same shapes, same value ranges, output that looks denoised:

1. **The QKV head split is not per-head triples.** Upstream writes
   `rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=head_dim).chunk(3, dim=0)`. Because
   `threeh = 3·n_heads` and the chunk is along the **head** axis, the projection's output channels
   are ordered `[all q heads][all k heads][all v heads]`. Reshaping to `(heads, 3, headDim)` — the
   intuitive reading — is shape-identical and silently wrong.
2. **The SW attention mask covers only the LAST window row and column.** Those are the windows the
   cyclic roll filled by wrap-around. Omit it and opposite edges of the image attend to each other,
   which reads as mild artefacts, not a crash.

And a third, in the checkpoint rather than the math:

3. **`relative_position_params` is stored PRE-PERMUTED.** `__init__` allocates `((2w−1)², heads)`,
   calls `trunc_normal_`, then **re-assigns the parameter** through
   `.view(2w−1, 2w−1, heads).transpose(1,2).transpose(0,1)`. The checkpoint carries
   `(heads, 2w−1, 2w−1)`. **Read the constructor, not the declaration** — this is a general rule for
   any parameter a constructor reassigns after initialising.

**The doctrine that falls out: gate the tables, at tolerance 0, before anything consumes them.** Add
a gate rung *below* "the attention op matches" that compares the stored bias parameter, the gathered
`(heads, w², w²)` bias, and the generated mask directly. On SCUNet all five landed at exactly
`0.00e+00`. That converts a diffuse "the output is a bit off" into a named cause, and it is cheap —
the tables are constants.

Also: **block type is decided at CONSTRUCTION, not runtime.** `Block.__init__` downgrades `SW → W`
when `input_resolution <= window_size`, and `input_resolution` is a *constructor argument* fed by the
UNet's resolution schedule — not the size of the image you pass in. A port that recomputed it from
the runtime shape would give some blocks the wrong attention. Encode the rule even when it is inert
for the released config (SCUNet's smallest is 32 > 8, so nothing actually downgrades).

## `Module` reflection collects EVERY stored `MLXArray` as a parameter (SCUNet, 2026-07-27)

A constant lookup table declared as a plain stored property —

```swift
private let relIndex: MLXArray   // constant gather indices, NOT a weight
```

— shows up in `parameters()`. SCUNet has 28 attention blocks, so S0 failed with **28 missing keys**
that no checkpoint could ever satisfy. Box it in a non-`Module` class (reflection walks past
`.other`), which also lets every block share one instance when the table depends only on config:

```swift
final class RelativeIndexTable: @unchecked Sendable { let indices: MLXArray /* + a static cache */ }
private let relIndex: RelativeIndexTable
```

The general form: **anything precomputed-and-constant must not be a stored `MLXArray` on a `Module`.**
Alternatives are a `[Int32]` rebuilt per call (wasteful) or a `static` (not reflected, but then it
cannot vary by config).

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

### Conditioner-stack lessons (mlx-indextts2-swift P3b, 2026-07-08)

Four conditioner models (w2v-BERT Conformer, MaskGCT RepCodec, CampPlus DTDNN,
conformer+perceiver) all gated **first-run green** — the verified-MLX-Python-donor +
per-stage-ladder doctrine working as designed. New traps it surfaced:

- **Numeric torch child keys break at `update`, not at the key contract.**
  `ModuleParameters.unflattened` treats every numeric path segment as an ARRAY index. A torch
  `Sequential(conv, bn)` child (`shortcut.0.*`/`shortcut.1.*`) or ModuleList-of-pairs
  (`layers.N.0.*` attn / `layers.N.1.*` ff) modeled as a Swift module with
  `@ModuleInfo(key: "0")` passes the 0-missing/0-unused key check, then `update(parameters:)`
  throws `incompatibleItems` (it built a *list* where your module tree has a *module*). Fix:
  **sanitize-remap numeric children to named keys** (`shortcut.{0,1}` → `{conv,bn}`,
  `layers.N.{0,1}` → `layers.N.{attn,ff}`). `MLXNN.Sequential` (the scail-2 idiom above) is only
  for containers whose forward IS a plain chain; heterogeneous pairs with interleaved residuals
  need the named-key remap.
- **MLX-Swift `BatchNorm` defaults to `training = true`** — a freshly built model silently
  normalizes with *batch* stats instead of the loaded `running_mean`/`running_var`, **and
  overwrites those running stats on every forward**, so repeated calls on one instance also drift.
  Any port carrying BN (speaker embedders, resnets, vocoders, conv decoders) must call
  `model.train(false)`. The frozen running-stat keys DO appear in `parameters()`, so the key contract
  can be green while the numerics are wrong. (BN running stats also load fine through
  `update(verify: .all)` — freezing affects `trainableParameters()`, not `update`.)
  - **This bullet existed and a PROD package shipped the bug anyway** (`mlx-birefnet-swift`, found
    2026-07-25 — the fast tier over-segmented by 68%, e2e logits cosine 0.264 vs oracle, and it
    passed in-app eyeball validation for months). Prose didn't hold; it needed a gate. Set eval mode
    at the **single construction choke point** every load path funnels through — not per forward,
    not per call site — and lock it with the engine's **C14 INF gate** (`porting-conformance.md`
    §5b), whose load-bearing assertion is that a *freshly constructed* model FAILS INF-1 and passes
    only after the choke point. Without that inversion, deleting the fix leaves the suite green.
  - **Diagnostic signature:** the LayerNorm/RMSNorm parts of the graph are bit-clean while
    everything downstream of the first BatchNorm diverges. A transformer encoder at cosine
    1.0000000 feeding a conv decoder at cosine 0.62 is this bug, not layer translation. It is also
    **dtype-independent** (fp16 ≈ fp32), and patching a suspicious value *inside* `running_var`
    changes nothing — which is itself proof the running stats are never read.
- **Fortran-order `.npy` fixtures keep recurring.** `np.save` preserves the layout of transposed
  /permuted tensors, and a strict Swift NPY reader (rightly) rejects F-order. Dump with
  `np.ascontiguousarray(...)` ALWAYS, and when a gate hits an F-order golden, sweep the whole
  goldens tree once (`a.flags['C_CONTIGUOUS']`) instead of fixing one file — this port found 6
  more in the same batch.
- **torch `avg_pool1d(ceil_mode=True, padding=0)` divides the partial tail window by its TRUE
  element count**, not the kernel size (verified empirically: `[9,10]` → 9.5, not 4.75). Segment
  poolings translated by hand (CAM-layer seg_pooling) must replicate the true-length divisor.
- **Precomputed non-weight tables (sinusoidal PE) must hide from Module reflection.** Plain
  `MLXArray` properties are auto-registered as parameters (breaking the 0-missing contract for
  keys not on disk). Park them in a tiny non-Module holder class (`final class PETable { let pe:
  MLXArray }`) — reflection skips it.

### 3D sampler + mesh/texture-stage parity (trellis2-mlx-swift SW3–SW5, 2026-07-14)

Later stages of the same sparse-3D port (flow-Euler CFG samplers, subdivision VAE decoders,
dual-grid→mesh, UV bake → glTF). All ops landed exact/float-eps, but four traps produce a
*wrong result that looks like a port bug* (or a gate that fails on a correct port):

- **Topology-adaptive decoders can't be gated by a direct coords compare.** An octree/subdivision
  decoder (predicts per-voxel "subdivide?" logits) or an occupancy→argwhere step makes DISCRETE
  branch decisions. A single logit at ≈0 rounds the other way (your 27-term conv sum vs torch's
  reduction order) and that one flipped decision COMPOUNDS through the upsample levels — free-run
  output diverged 0.16 % in coords while every continuous stage was 0.9999998. **Gate by injecting
  the oracle's discrete decisions** (dump the subdivision masks / occupancy) and run a "guided"
  forward → coords BIT-EXACT + feats 0.9999999; then report free-run agreement (e.g. 20519/20520
  masks match) as informational. Free-run divergence here is fp tie-nondeterminism, NOT a defect —
  same class as an argmax tie. (Applies to any adaptive-topology net: MoE routing, NMS, pruning.)
- **High-guidance CFG amplifies per-forward fp noise via near-cancellation — a "failing" sampler
  gate on a correct port.** `v = s·vPos + (1−s)·vNeg` with s=7.5 is `7.5·vPos − 6.5·vNeg`; when
  vPos≈vNeg (they nearly always are) it magnifies each forward's ~1e-6 rounding, compounding over
  the step loop. A 12-step sparse sampler read cos **0.987** while every single forward (cond AND
  neg, at t=500/900/1000) was **0.9999972+** and the SAME blend math passed for the dense stage.
  Don't chase it as a bug: gate the **deterministic path** (guidance_strength=1 → cos 0.99999243
  over 12 steps) + per-forward parity, and report the production-guidance run as informational.
  Confirm by sweeping s: g=1 → ~1.0 isolates CFG as the amplifier. (Immaterial downstream — the
  latent feeds a robust decoder, and production runs GPU not CPU-fp32.)
- **Sampler/scheduler time schedules must be built in `Double`, and never wrap a Double literal in
  `MLXArray`.** Two edges: (1) `np.linspace(1,0,n)` + a rescale warp lands a step EXACTLY on a
  guidance-interval boundary (t=0.6 at rescale_t=3, 12 steps); float32 flips it to the wrong side →
  that step gets CFG-on instead of off (7.5× vs 1× swing) → whole-trajectory divergence. Compute
  the schedule + the `interval[0] ≤ t ≤ interval[1]` test in `Double` (numpy float64). (2)
  `MLXArray([1000.0 * t])` where the literal is `Double` builds a **float64** array that
  **HARD-CRASHES on GPU** ("float64 is not supported on the GPU", a `fatalError`, not a wrong
  number). Always `[Float]` for anything that reaches Metal.
- **Baking a texture onto a REMESHED/decimated surface samples the voxel volume BLACK unless you
  closest-point-remap first.** The tex attributes live on a thin (1-voxel) sparse shell at the fine
  resolution; DC-remesh (coarser grid) / QEM-decimate move the surface off that shell, so trilinear
  `grid_sample_3d` at the new surface positions misses every active voxel (hitFrac **1.5 %** →
  near-black atlas, mean 1.6/255). The CUDA path's `cuBVH closest-point remap` is load-bearing, not
  optional polish: **BVH-remap each texel's barycentric surface position onto the ORIGINAL on-shell
  dual-grid mesh, THEN sample** (`origMesh.bvh().closestPoints(mesh:queries:).points`) → hitFrac
  1.0. Query = `(pos − aabb0)/voxel_size = (pos+0.5)·gridRes`; the sampling gridRes = the fine voxel
  resolution the attrs live in, NOT the remesh resolution.
- **Mesh-stage ordering & manifold caveat.** DC-remesh (`remeshDualContouring`) turns a fragmented
  thin-shell dual-grid mesh (thousands of components, 120k boundary edges) into a watertight solid
  (0 boundary edges) — but run it BEFORE UV-unwrap and decimate to budget BEFORE unwrap (xatlas on
  millions of dirty faces grinds). `simplify` (QEM) can re-introduce non-manifold / inconsistent
  winding after a clean remesh (trimesh `is_watertight` False post-simplify) → `unifyFaceOrientations`
  / fix-normals after. mlx-swift-mesh surface used: `Mesh(vertices:faces:)`,
  `.remeshDualContouring(resolution:band:)`, `.simplify(targetNumFaces:)`,
  `.uvUnwrap()→{mesh,uvs[V,2],vertexMap,atlasW/H,charts}`, `.vertexNormals()`, `.bvh()`,
  `bvh.closestPoints`, `.numBoundaryEdges/.numConnectedComponents`.
- **Delegating self-contained utilities to a subagent works when the boundary is a NEW FILE + no
  build.** A binary-glTF/GLB writer (standard format, clean signature, no MLX-parity coupling) was
  a good background-subagent job; the guardrails that avoided a corrupt shared build: write ONE new
  file only (no Package.swift / no shared-main edits — another session was editing those), and
  **do not `swift build`** (concurrent builds on one `.build` corrupt it) — the parent compiles +
  smoke-tests it on integration. Inspection-only agent code then needs a runtime smoke (write a
  cube GLB, re-read magic + parse JSON) before wiring into the real pipeline.

---

### Gate-threshold discipline: relative error, and loosening with evidence (image-restoration batch, 2026-07-27)

Four conv/transformer restoration ports in one pass (FFTformer, HVI-CIDNet, Restormer, DRUNet)
converged on three rules about thresholds. All three came from a gate that failed for the *wrong*
reason.

**1. Judge on RELATIVE error, not absolute.** Sub-op outputs span orders of magnitude within one
model — a channel LayerNorm output sits near ±2 while a `Fuse` block on seeded inputs reaches ±2400.
A single absolute tolerance either fails clean fp32 rounding on the large tensors or waves through
real errors on the small ones. FFTformer's `fuse` gate "failed" at `max_abs 8.5e-4` while its cosine
was `1.00000000` — the tensor simply had big values. Use `maxAbs / max|reference|` and report
`max_abs` alongside for context.

**2. 1e-6 is ON the fp32 noise floor for wide accumulations — don't set gates there.** Measured
spreads: a conv accumulating 64+ input channels lands 1.6e-07…1.0e-06; one accumulating 384 lands
~2.6e-06. A 1e-6 threshold on those produces coin-flip failures rather than signal. 2e-6 for
primitives over ≤64 channels, 1e-5 where the accumulation is deep, and note WHY in the gate.

**3. If you loosen a tolerance, prove the gate still catches what it guards.** The honest move is to
run the actual failure mode as a probe and print the margin. Two worked examples now in-tree:

- **Restormer pixel-shuffle**: the hazard is the channel-split ordering — `(r,r,C)` instead of
  `(C,r,r)` compiles, runs, and silently scrambles the image. The gate runs that exact mistake and
  reports `rel=1.30e+00`, **493,451× the observed rounding**, so a 1e-5 threshold retains ~5 orders
  of discrimination.
- **FFTformer resamplers**: an interior-vs-border diagnostic. Edge-handling bugs (wrong
  `alignCorners`, off-by-one grid) concentrate error at the boundary; uniform rounding does not.
  Measured `overall max == interior max (×1.00)` ⇒ rounding, not semantics.

Both probes stay committed, so the margin remains visible instead of becoming folklore.

### Weight-layout traps that are INVISIBLE to shape checks

Two from this batch. Both load clean, pass every structural gate, and are silently wrong.

**`ConvTranspose2d` is transposed differently from `Conv2d`** (DRUNet):

| | PyTorch | MLX | transpose |
|---|---|---|---|
| `Conv2d` | `(O, I, kH, kW)` | `(O, kH, kW, I)` | `(0,2,3,1)` |
| `ConvTranspose2d` | **`(I, O, kH, kW)`** | `(O, kH, kW, I)` | **`(1,2,3,0)`** |

In DRUNet, `m_down3.4.weight` (strideconv 256→512) and `m_up3.0.weight` (transposed 512→256) are
**both `(512, 256, 2, 2)`** — identical shape, opposite meaning. Only the KEY discriminates. Key the
converter on names and **assert the expected count** (`assert n_transposed == 3`).

**Attention biases may be stored pre-permuted by `__init__`** (SCUNet). `WMSA.__init__` re-assigns
`relative_position_params` as `.view(2w-1, 2w-1, heads).transpose(1,2).transpose(0,1)`, so the
checkpoint layout is already rotated against the naive `(…, heads)` shape. Read the constructor, not
just the declaration.

### Read the RELEASED config, not the constructor defaults

SCUNet's constructor defaults to `config=[2]*7` → 9,662,892 params, which fails with 3 missing / 269
unexpected. Every released checkpoint uses `[4,4,4,4,4,4,4]` → 17,946,072, and every upstream test
script passes it explicitly. FFTformer's `Fuse` has the same shape of trap in miniature: it builds its
inner block **without** passing `ffn_expansion_factor`, so that block takes the constructor default
`2.66` while the rest of the model uses `3`. Both are visible in the weights if you look
(`510/96 = 2.6562` vs `288/48 = 3.0`), and invisible if you don't.

### Behaviour can be baked at CONSTRUCTION, not derived at runtime

SCUNet's `ConvTransBlock.__init__` downgrades shifted-window → plain-window when
`input_resolution <= window_size` — and `input_resolution` is a *constructor argument* fed by the
UNet's decreasing resolution schedule, **not** the runtime input size. A port that recomputed it from
the actual image would silently give deep blocks the wrong attention type. When a constructor takes a
resolution/shape hint, check whether it changes structure.

### Ablate to confirm dead code before "faithfully" porting it

HVI-CIDNet's `forward` computes `i_dec2 = I_LCA5(...)` and overwrites `i_dec2` before any read.
Rather than trust the reading, zero the module's weights and re-run: output changed by **exactly
0.0**, while `HV_LCA5` / `I_LCA2` / `I_LCA6` each moved it by ~0.5–1.0. Declare the module (its
weights are in the checkpoint, so strict loading needs it) and skip evaluating it. Cheap check,
decisive answer.

### Publish-dtype: measure per model — and fp16 can BEAT bf16

The "small conv nets ship fp16" rule is not a fact. FFTformer measured end-to-end against its fp32
reference: **bf16 → cosine 0.9839 / 30.09 dB** (disqualifying: the model's whole task signal is GoPro
34.21 dB), **fp16 → cosine 0.99994 / 50.13 dB**. fp16 beat bf16 by **20 dB**, which inverts the
LLM-world default. Peak activation was only ~2955, comfortably inside fp16's 65504 ceiling, so this
was a **mantissa-precision** problem where fp16's 10 bits beat bf16's 7 — not a dynamic-range one.
bf16 wins in LLM work for its fp32-equivalent *exponent* range; a small conv/FFT net with bounded
activations wants the opposite.

⚠️ **A torch-CPU dtype probe is not authoritative.** The same experiment reported fp16 as *NaN* on
torch-CPU, which was an emulation artifact — flagged as needing confirmation, and disproved on MLX.
The bf16 figure reproduced across both backends to 0.01 dB, which is what made *it* trustworthy.
Run the dtype gate on the real backend.

## Before copying a windowing/streaming recipe, compute YOUR stack's receptive field (Audio8, 2026-07-30)

`mlx-gepard-swift` streams by windowing its whole codec decoder: decode frames `[a−L, b)`, discard
the first `L` frames' samples, bit-identical because every op is causal. `L ≈ 26` there. The recipe
is sound and it transfers — but **where** you apply it is a per-model question, and getting it
wrong is silent.

Audio8's decoder is also strictly causal, so the argument holds. Its receptive field is not 26:

- **Stacked local attention COMPOUNDS.** Its `post_module` is 8 layers of 128-wide causal window
  attention, so the field is `8 × 127 = 1016` frames (~47 s), not 128. Any port with windowed or
  sliding attention has this multiplier; a single layer's window is not the answer.
- Windowing above that module was therefore impossible — the context needed exceeds a typical
  utterance. Windowing it *at all* is also pointless: it is a transformer over `1024 × T`, while
  the conv stack **below** it expands to 2048 samples per frame through 1536→96-channel
  intermediates. That is where the memory lives, and its field is **11 frames**.

So the seam is *between* them: run the long-context module once over the whole (or prefix) input,
window only the expensive short-context half. Method: **backward extent propagation**, output →
input — residual branches contribute `Σ (k−1)·dilation`, transpose convs convert fine→coarse as
`ceil((e + k − 1) / stride)` — then verify empirically.

**Two independent floors, and conflating them costs hours.** Measuring found drift that more
context did not fix:

| context | 11 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| max_abs | 1.4e-3 | 1.4e-3 | 1.4e-3 | 1.4e-3 | 1.4e-3 |

Identical at every context ⇒ **not** a receptive-field shortfall. The real variable was CHUNK size
(exact at ≥48, drifting below): MLX selects different conv kernels for small inputs and their fp32
reduction order differs from the full-length path. Derive the context floor; **measure** the chunk
floor.

Also: a receptive-field bound's contract is **sufficiency, not tightness** — derived 11 vs measured
minimum 10 is correct and conservative. Asserting equality makes the gate fail on a safe
over-estimate.

**Interleave, or streaming buys nothing.** The first working version decoded only after the AR
rollout completed: TTFA 9.51 s of a 9.90 s run — "streaming works" was true and useless. Decoding
*during* generation is legitimate when the long-context module is causal (a prefix run yields
identical values for that prefix), and took TTFA to 2.41 s. But note the cost: that path re-runs the
prefix each chunk (quadratic). Don't route BATCH through it — batch wants the module run once.
