# Custom lowerings and Metal kernels

**Status: DOCUMENTED, not yet exercised.** Everything here is read from
`apple.github.io/coreai-torch/main/guides/` (2026-08-29) and is **INHERITED** until a port
proves it. This closes the single largest OPEN in the skill — but reading is not measuring.

Primary target: **`birefnet`**, blocked on `deform_conv2d`. The upstream docs use
`deform_conv2d` as their own example of an unsupported ATen op, so this is a well-trodden path.

---

## `register_torch_lowering` is an INSTANCE-METHOD DECORATOR

Not a global function. Lowerings are stored **on the `TorchConverter` instance** and do not
affect other converters or the global tables.

```python
converter = TorchConverter()

@converter.register_torch_lowering("namespace::op_name.overload")
def lower_op(values_map, node, loc):
    ...
```

| Param | Type | What it is |
|---|---|---|
| `values_map` | `dict[str, Value]` | FX node name → CoreAI `Value`, for tensor operands |
| `node` | `torch.fx.Node` | the FX node; **tensor args are `fx.Node`, scalars are plain Python values** |
| `loc` | `Location` | pass through to every CoreAI op constructor |

Helpers: `get_operand(values_map, node, index, loc)` and
`get_operands(values_map, node, [indices], loc)` from `coreai_torch._utils`.

**Name format:** `"namespace::op_name.overload"`. Custom ops always use `.default`; ATen ops in
FX graphs carry the overload suffix explicitly.

**Reserved namespaces** — `aten`, `higher_order`, `coreai`, `coreaix` — require
`allow_override=True`:

```python
@converter.register_torch_lowering("aten::deform_conv2d.default", allow_override=True)
def lower_deform_conv2d(values_map, node, loc):
    ...
```

**Multiple returns:** return a Python list of `Value`s; the converter stores them as
`"node_name#0"`, `"node_name#1"`, ….

**Private-API warning (upstream's own):** `coreai._compiler.dialects` and `coreai_torch._utils`
carry leading underscores — they are private and **may change across `coreai-core` releases**.
Any lowering we write is version-fragile. Pin, and re-verify on every toolchain bump.
→ the version-scoping trap in `runtime-api.md`.

### Worked override from the docs — static `_adaptive_avg_pool2d`

Useful as a template because it shows real CoreAI ops (`sumpool2d`,
`broadcasting_divide`, `cast`, `constant`) and static-shape arithmetic:

```python
@converter.register_torch_lowering("aten::_adaptive_avg_pool2d.default", allow_override=True)
def lower_adaptive_avg_pool2d_static(values_map, node, loc):
    x = get_operand(values_map, node, 0, loc)
    output_h, output_w = node.args[1]
    input_h, input_w = x.type.shape[2], x.type.shape[3]
    stride_h, stride_w = input_h // output_h, input_w // output_w
    kernel_h = input_h - (output_h - 1) * stride_h
    kernel_w = input_w - (output_w - 1) * stride_w
    return coreai.broadcasting_divide(
        coreai.sumpool2d(x,
            kernel_size=np.array([kernel_h, kernel_w], dtype=np.uint32),
            strides=np.array([stride_h, stride_w], dtype=np.uint32),
            dilation=coreai.constant([1, 1], dtype=np.uint32)),
        coreai.cast(float(kernel_h * kernel_w), x.type.element_type))
```

---

## `TorchMetalKernel` — raw Metal shader as a torch op

Construction takes: `name`, `input_names` / `result_names` (must match the Metal source),
`src` (**the shader body only** — the function signature is auto-generated), `torch_defn` (a
PyTorch reference implementation, **required for shape inference during export**),
`metal_params` (e.g. `thread_position_in_grid`), and optional `template_dtypes`.

Call sites must specify `threads_per_grid`, `threads_per_thread_group`, and `result_shapes`.

Registration differs from lowerings — it is a separate call, **before** `add_exported_program`:

```python
converter.register_custom_kernels([kernel])
```

Custom kernels are **preserved through `run_decompositions()`**.

**Use it when:** the op has no standard PyTorch equivalent, you want to fuse several ops into one
dispatch, or you need Metal-specific control. **Otherwise prefer a custom lowering** built from
CoreAI ops.

### ⚠️ The question the docs do not answer — and it decides Phase 2

**OPEN, and load-bearing:** the documentation **does not state whether a Metal kernel forces GPU
execution and thereby excludes the ANE.**

If it does, then unblocking `birefnet` with a Metal kernel would *defeat the purpose of the
port* — we would have converted a model to CoreAI and lost the only reason to be there.

**Therefore, for `birefnet` and anything like it:**

1. Try a **custom lowering from CoreAI ops** first (option 4 in the triage ladder).
2. Treat `TorchMetalKernel` as the fallback, **and measure placement on the result** before
   calling it a success — three lanes, stderr scan, GPU-idle check.
3. If a Metal kernel does force GPU, record it here as MEASURED. It reshapes the whole
   custom-op strategy and belongs in `mlx-vs-coreai-fit.md` too — a model needing a Metal
   kernel may simply belong on MLX.

---

## Composite ops — the transformer signal

`get_decomp_table()` **preserves** these rather than decomposing them:

- **Module-class:** `GatherMM`, `GatedDeltaUpdate`, `RMSNormImpl`, `RoPE`, `SDPA`
- **ATen-derived:** `batch_norm`, `group_norm`, `hard_sigmoid`, `instance_norm`, `layer_norm`,
  `linalg_vector_norm`, `log_softmax`, `pixel_shuffle`

Two things follow:

1. **`scaled_dot_product_attention` is preserved as a composite.** Attention may be far better
   supported than our Moebius experience suggested — that model used a *custom λ-attention* with
   hand-rolled einsums, not SDPA. **A model that uses stock `F.scaled_dot_product_attention` is
   a materially different proposition** and Wave 3 should test that specifically before
   generalizing from Moebius.
2. `RoPE`, `RMSNorm`, `GatedDeltaUpdate` and `GatherMM` are **LLM/transformer inference
   primitives**. The toolchain is being actively built for transformer decoding — which bears
   directly on the "CoreAI is for convnets, MLX is for LLMs" assumption in
   `mlx-vs-coreai-fit.md`. That assumption is now **less safe than when we wrote it.**

Also: `generate_composite_decl` and `ExternalizeSpec` exist as API surface. Externalization has
its own guide (`guides/externalization.html`) — **still unexercised by us.**

---

## Supported-ops list — triage before failure

`api/supported-aten-ops.html` lists what `TorchConverter` lowers **out of the box**. Consult it
*before* an export rather than discovering gaps by failure.

| Op | Listed? |
|---|---|
| `avg_pool2d` | ✅ (lowered as a composite) |
| `_adaptive_avg_pool2d.default` | ✅ |
| `scaled_dot_product_attention` | ✅ (preserved as composite) |
| `grid_sampler_2d` | ❌ |
| `as_strided` | ❌ |
| `_upsample_bicubic2d_aa` | ❌ |
| `deform_conv2d` | ❌ |
| `einsum` | ❌ |

### A contradiction to resolve — good first A/B

Upstream lists **`_adaptive_avg_pool2d.default` as supported**. LibreYOLO replaces
`AdaptiveAvgPool2d(1)` with a spatial mean, commenting that it *"decomposes to
`aten.as_strided`, which the Core AI converter cannot lower."*

Both can be true — if `get_decomp_table()` decomposes it to `as_strided` **before** the
converter's own lowering ever sees it, the supported lowering never fires. But that is a
hypothesis.

**This is a clean, cheap experiment** and a good rehearsal for the sealed protocol: export a
model containing `AdaptiveAvgPool2d(1)` three ways — untouched, LibreYOLO's spatial-mean swap,
and the docs' `allow_override` static lowering — and measure all three on all three lanes.
Per the A/B rule, "it works" is not the answer we want; we want to know which is *better*, and
why.
