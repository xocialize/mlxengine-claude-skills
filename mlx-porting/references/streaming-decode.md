# Memory-bound video VAE decode — stream it (temporal-chunked, lossless)

**When this applies:** any video model whose VAE decodes a `[B, C, T_lat, H, W]` latent to a
`[B, 3, T_out, H, W]` clip in **one whole-sequence pass** — Wan (2.1/2.2), LTX, CogVideoX, Hunyuan-Video,
and every fine-tune riding them (Phantom-Wan, Bernini-R, LongCat, Lance…). The DiT is usually *not* the
memory bottleneck; the VAE decode is.

## The problem (measure it before deciding)

Whole-sequence decode materializes the highest-resolution decoder activations for **all temporal frames
at once**, so peak memory grows ~linearly in frame count and OOMs fast. Measured on a 128 GB M5 (Wan2.1
16-ch VAE, 832×480, decode only):

| frames | whole-seq peak | streaming peak |
|--------|----------------|----------------|
| 17 | 47.5 GB | 19.7 GB |
| 49 | 115.9 GB | ~20 GB |
| 81 | **OOM** | 20.4 GB |
| 121 | — | 20.6 GB |

So the *standard* 81-frame output is impossible whole-sequence, and the existing lossy `decode_tiled`
doesn't save you — it tiles **spatially** while the blow-up is **temporal**. Streaming (temporal-chunked)
decode is a **capability unblock, not an optimization**: flat ~20 GB at any length.

> **Measurement hygiene:** an OOM "ceiling" measured while another process (a training run, another
> model) holds unified memory is confounded — the *clean* ceiling sits higher. The per-frame GB and the
> linear scaling are still valid; re-measure the absolute cliff on an idle machine, but it doesn't change
> the streaming-vs-linear conclusion.

## The fix

Decode one (or `chunk_lat`) latent frame(s) at a time, threading the decoder's **`CausalConv3d`
cross-chunk cache** — exactly the pattern the encoder side already ships (`WanVAE.encode` loops chunks
with `feat_cache`/`feat_idx`). The decode side usually just never wired it. Consumer-side, **no fork**:
the building blocks (`ResidualBlock`, downsample `Resample`) already accept `feat_cache=`; only the
`upsample3d` temporal conv and the top-level chunk loop are missing.

```python
CACHE_T = 2  # causal cache depth for a kernel-3 conv

def _conv_cached(conv, x, fc, fi):           # CausalConv3d w/ cross-chunk cache (NCHWD: T axis = 2)
    idx = fi[0]
    cache_x = x[:, :, -CACHE_T:]
    if cache_x.shape[2] < 2 and fc[idx] is not None:
        cache_x = mx.concatenate([fc[idx][:, :, -1:], cache_x], axis=2)
    out = conv(x, cache_x=fc[idx]); fc[idx] = cache_x; fi[0] += 1
    return out

def decode_streaming(vae, z, chunk_lat=1):
    z = z / vae.inv_std.reshape(1,-1,1,1,1) + vae.mean.reshape(1,-1,1,1,1)   # un-normalize
    fc, outs = [None] * 64, []                                               # generous; 1 slot/cached conv
    for s in range(0, z.shape[2], chunk_lat):
        xc = vae.conv2(z[:, :, s:s+chunk_lat])                               # kernel-1 -> per-frame, no cache
        oc = _decoder_chunk(vae.decoder, xc, fc, [0])                        # walk conv1/middle/upsamples/head
        mx.eval(oc, *[c for c in fc if isinstance(c, mx.array)])             # see "lazy cache" gotcha
        outs.append(oc)
    return mx.clip(mx.concatenate(outs, axis=2), -1, 1)
```

`_decoder_chunk` mirrors the whole-seq `Decoder3d.__call__`, but routes each temporal-mixing op through
the cached variant (`_conv_cached` for `conv1`/`head`, the block's own `feat_cache=` path for
`ResidualBlock`, the cached `upsample3d` below) and leaves per-frame ops (Attention, `upsample2d`
spatial conv, kernel-1 convs) untouched.

## The cross-implementation trap — the `upsample3d` "Rep" sentinel is NOT universal

The temporal-upsample frame bookkeeping **differs by VAE implementation** and is the one place you must
read the *specific* reference, not copy another port's streaming code:

- **diffusers / Wan22 / LongCat** `Resample.upsample3d`: the first chunk's frame-0 is **not doubled** (a
  `"Rep"` sentinel skips `time_conv` on the first call, frames 1.. are time-conv'd). Streaming must
  replicate the first-chunk skip. (This is what Lance's `vae_stream.py` implements.)
- **mlx-video stock** `wan_2/vae.py` `Resample.upsample3d`: **always doubles every frame** (no first-chunk
  skip). So the temporal op is *just* a cache-threaded `time_conv` — **no `"Rep"` logic** — and copying
  the diffusers/Lance version here produces a **wrong frame count** (e.g. `4*T_lat-3` instead of `4*T_lat`).

Symptom of getting it wrong: streaming output has a *different T_out* than `vae.decode(z)`. Check frame
count first, values second.

> ⚠️ **But mlx-video stock's always-doubling is itself a BUG vs the real model — not just "a variant to
> match."** It emits `4*T_lat` frames where true Wan2.2/diffusers emit `(T_lat−1)*4+1 = 4*T_lat−3`
> (e.g. **20 vs 17** for a 17-frame request — the count can't even round-trip; the first latent frame
> expands to 4 output frames instead of 1). The **newer mlx-video (helios branch) FIXED it** to the
> first-chunk-skip behavior. So the trap cuts **both ways**: a port riding *old* mlx-video that matches
> its always-double whole-decode has a `streaming == whole_decode` gate that is **bit-identical to a
> flawed reference** — it passes while BOTH diverge from the real model's output length. When output
> length must match the true model, port the **first-chunk skip** and add a **semantic-invariant gate**
> (`assert T_out == (T_lat−1)*4+1`) checked against the *authoritative* reference (diffusers /
> helios-branch mlx-video), NOT the convenient already-ported oracle. *(bernini-r `E11`: shipped
> 20-frame clips for 17-frame requests; S5 bit-identity couldn't catch it because the bernini Python
> oracle inherited the same old-mlx-video flaw — a self-consistency gate against a flawed reference is
> no gate at all.)*

## Gotchas

- **Lazy cache aliasing:** `feat_cache` stores slice-views into the chunk's buffers. Across the chunk
  boundary (where you `mx.eval` the output and free buffers) those views go stale → wrong results from
  chunk ≥3. Materialize the carried cache each chunk: `mx.eval(oc, *live_caches)`.
- **`chunk_lat=1`** is the minimum-memory / flat-in-length floor (Lance's finding; confirmed here).
  Larger chunks trade memory for fewer boundary recomputes — sweep if you need throughput.

## Correctness gate — bit-identity ON CPU (not GPU)

Streaming runs *more, smaller* convs than whole-seq, so on the Apple **GPU** (tf32-like fp32 conv, see
`parity-testing.md`) chunked-vs-whole differs by reduction-order noise (~1e-2) — **not a bug**. Gate
correctness on `mx.set_default_device(mx.cpu)`, where it is **bit-exact** (`max|Δ| == 0.0`):

```python
@pytest.mark.parametrize("t_lat", [1, 2, 3, 5])           # 1,2 chunks AND ≥3 (catches cache aliasing)
def test_streaming_bit_exact_cpu(t_lat):
    mx.set_default_device(mx.cpu)
    z = mx.array(rng.standard_normal((1, 16, t_lat, 16, 16)).astype("float32"))
    assert float(mx.abs(decode_streaming(vae, z) - vae.decode(z)).max()) == 0.0
```

Weights-free (random-init VAE works) and ~1 min on CPU. Add a GPU **flat-memory** assertion too (peak at
81 frames within ~15 % of peak at 17 frames). The debugging tell if it *isn't* bit-exact: every
sub-component (`CausalConv3d`, norm, shortcut) is bit-exact in isolation but the *composition* diverges
and the error **grows downstream** → that's reduction-order numerics, confirm on CPU — not a logic bug.

## Cross-port reuse

Ports sharing the same VAE module reuse the file **verbatim** — phantom-wan → bernini-r was a literal
copy (both on mlx-video's 16-ch `wan_2.vae`), bit-exact on the second VAE with zero edits. Confirm the
shared VAE first (`encode` returning `[1,16,1,…]` ⇒ 16-ch `wan_2.vae`, not the 48-ch `vae22`). A VAE on a
*different schema* (LongCat's forked diffusers-layout) needs real adaptation, not a copy.

> **The flip side of verbatim reuse: a flaw propagates verbatim too.** The E11 frame-count bug above
> rode old-mlx-video into bernini-r *and* phantom-wan by the same literal copy. This is the real argument
> for a **single shared substrate package** (a `wan-core` the consumers depend on) over N copies: one
> fix lands everywhere at once, and there's one place to gate against the authoritative reference — but
> only if you **verify the shared file against that reference, not just propagate it.** A copy that's
> "bit-exact on the second VAE with zero edits" proves the two ports *agree*, never that either is
> *correct*. Extract to share **after** the shared file is parity-locked against the upstream of record;
> sharing an unverified file just industrializes the bug.

## Optional Phase 2 — spatial halo-tile

Phase-1 temporal streaming already caps the floor at ~one-chunk extent (~20 GB), which unblocks the full
81-frame envelope. The high-res suffix (last upsample stages + head) can be further **spatially**
halo-tiled (real-neighbour halo + crop, lossless — *not* the lossy trapezoidal `decode_tiled` blend) to
push the floor lower, only if you need 4K / very long clips. Defer until a measured need.

---

## Generalize: deep sparse/conv decoders OOM the same way — stream them too (TRELLIS.2, 2026-06-26)

The video-VAE lesson above is one instance of a broader MLX trap. Any **deep many-block decoder**
(here: TRELLIS.2's 512³ sparse-VAE shape decoder — ~30 ConvNeXt/up blocks growing 5.8K → 2.2M voxels)
hits the same wall, and the diagnosis order matters because the obvious culprit is usually wrong.

**Trap 1 — the whole decoder evals as ONE deferred graph → peak = Σ(all blocks), not max(block).**
MLX is lazy: if you only `eval` at the very end of the decoder, MLX holds *every* block's intermediates
live simultaneously during that final eval. Measured: TRELLIS.2 shape decoder peaked **44.24 GB**; with
an `eval(feats)` after **each block** (free this block's workspace before the next) it dropped to
**11.93 GB** — and got *faster* (80s → 71s). This is the single highest-leverage memory fix; it is the
decoder-scale version of "insert `mx.eval` at natural boundaries." **Always eval at block boundaries in
a long decode loop.**

**Trap 2 — don't attribute the peak to dtype before you measure per-stage.** Casting the decoder to
fp16 changed the peak by **zero** (still 44 GB) because the real driver was Trap 1, not precision.
**Instrument per-stage first:** wrap each stage with `MLX.GPU.resetPeakMemory()` + read `GPU.peakMemory`
(`Memory.peakMemory`). `peakMemory` is a global high-water mark that is never auto-reset — a single
end-of-run read tells you nothing about *which* stage spiked. Reset-per-stage turns it into a profiler
and points at the real op in one run. Only then pick the lever.

**Trap 3 — memory-bound K-reductions: stream over K, don't materialize the stacked tensor.** A
submanifold sparse conv is `gather (K,N,Ci) → matmul → (K,N,Co) → sum over K`. Materializing the full
`(K,N,Ci)` gather **and** the `(K,N,Co)` matmul spiked ~37 GB at the 512³ channel-expansion conv
(128→512 on 1.1M voxels). Reformulate as an **accumulation over the K kernel taps** — gather only
`(N,Ci)` per tap, `acc += matmul(gather_k, w_k)` — so peak ≈ `(N,Ci)+(N,Co)`. Same math; the explicit
add-chain reorders the K-reduction by ~fp-rounding (bit-exact `0` → `~1e-7`, still ≪ tol). This is the
sparse-conv analog of temporal streaming.

**Counterintuitive — finer eval is often FASTER at scale.** `eval` *per K-tap* (810 evals in the decode)
beat a lazy 27-chain (71s vs 111s). A big deferred graph at 512³ incurs memory-pressure stalls; keeping
each step's workspace tiny avoids them. Don't assume "fewer evals = faster" for memory-bound graphs —
measure. Final TRELLIS.2 config = per-tap eval (conv) **and** per-block eval (decoder): 71s / 11.93 GB.

**The dtype lever is on ACTIVATIONS, not weights.** MLX promotes `fp32-activation × bf16/f16-weight →
fp32` in matmul/conv, so loading weights as f16 alone doesn't shrink the working set — the activations
do. Cast the *activations* to the compute dtype at the stage entry (and back to fp32 for the final
param-free LN). For TRELLIS.2 the 512³ activations are the driver; weights (~8 GB resident) are not.
