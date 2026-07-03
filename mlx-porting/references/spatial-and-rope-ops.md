# Hand-rolled spatial / RoPE ops in MLX (video & flow models)

Ops that **`mlx_arsenal` and `mx.fast` do not cover** and recur across video ViTs,
optical-flow / frame-interpolation, and 3D-patch models. Each recipe below was
**parity-locked vs PyTorch on `mx.cpu` fp32** in a shipped port. All are **NHWC**
(or NDHWC) — MLX's native layout. Reach for these before writing a Metal kernel.

> `mlx_arsenal.spatial` has `interpolate_nearest`, `pixel_shuffle`, `PatchEmbed2d/3d`
> — but **NOT** `grid_sample`, **NOT** bilinear `interpolate`, **NOT** 3D-RoPE.
> Don't assume an op exists because a near-neighbor does.

---

## 1. Bilinear `grid_sample` (warp) — `rife-mlx`

`torch.nn.functional.grid_sample(mode='bilinear', padding_mode='border', align_corners=True)`.
Backs RIFE's backward-warp. Weights come from the **unclamped** continuous coordinate;
the 4 neighbor **indices are clamped** (border).

```python
def grid_sample_bilinear(inp, grid, align_corners=True):   # inp [N,H,W,C], grid [N,gH,gW,2] in [-1,1]
    N, H, W, C = inp.shape; _, gH, gW, _ = grid.shape
    gx, gy = grid[..., 0], grid[..., 1]
    if align_corners: ix, iy = (gx+1)*0.5*(W-1), (gy+1)*0.5*(H-1)
    else:             ix, iy = ((gx+1)*W-1)*0.5, ((gy+1)*H-1)*0.5
    x0, y0 = mx.floor(ix), mx.floor(iy); x1, y1 = x0+1, y0+1
    wx1, wy1 = ix-x0, iy-y0; wx0, wy0 = 1-wx1, 1-wy1          # weights from UNCLAMPED coord
    cx = lambda a: mx.clip(a,0,W-1).astype(mx.int32); cy = lambda a: mx.clip(a,0,H-1).astype(mx.int32)
    flat = inp.reshape(N, H*W, C)
    def g(yc, xc):
        idx = mx.broadcast_to((yc*W+xc).reshape(N,gH*gW,1), (N,gH*gW,C))
        return mx.take_along_axis(flat, idx, axis=1).reshape(N,gH,gW,C)
    return (g(cy(y0),cx(x0))*(wy0*wx0)[...,None] + g(cy(y0),cx(x1))*(wy0*wx1)[...,None]
          + g(cy(y1),cx(x0))*(wy1*wx0)[...,None] + g(cy(y1),cx(x1))*(wy1*wx1)[...,None])
```
*Trap:* clamp the **indices**, not the weights. The identity grid → identity output;
an integer flow → `np.roll`. Verify both as smoke tests.

## 2. Bilinear `interpolate` (resize) — `rife-mlx`

`F.interpolate(mode='bilinear')`. Separable 1-D resample per axis. **Pin `align_corners`
from the source** (RIFE IFNet uses `False`); the source coordinate differs:
`align_corners=True → dst*(in-1)/(out-1)`, else `(dst+0.5)*(in/out) - 0.5`.

```python
def _axis(x, axis, out, align):
    in_ = x.shape[axis]
    if in_ == out: return x
    dst = mx.arange(out, dtype=mx.float32)
    src = dst*((in_-1)/(out-1) if (align and out>1) else 0.0) if align else (dst+0.5)*(in_/out)-0.5
    i0 = mx.floor(src); w1 = src-i0; w0 = 1-w1
    i0c = mx.clip(i0,0,in_-1).astype(mx.int32); i1c = mx.clip(i0+1,0,in_-1).astype(mx.int32)
    sh = [1]*x.ndim; sh[axis] = out
    return mx.take(x,i0c,axis=axis)*w0.reshape(sh) + mx.take(x,i1c,axis=axis)*w1.reshape(sh)
def interpolate_bilinear(x, scale, align=False):   # x [N,H,W,C]
    oH, oW = int(round(x.shape[1]*scale)), int(round(x.shape[2]*scale))
    return _axis(_axis(x,1,oH,align),2,oW,align)
```
*Trap (coarse-to-fine pyramids):* when a block does `interp(x, 1/scale)` then `interp(·, scale)`,
the round-trip only closes if input dims are a multiple of `base/scale` — pad
**to a multiple of `base/scale`**, not a fixed base. (1080→1088 at /64 fails at
scale 0.5; needs 1152.) Caught by benchmarking, not unit tests.

## 3. 3-D RoPE for video ViTs — `vjepa2-mlx`

Rotary over a `(depth, height, width)` token grid; `head_dim` split per axis. **Not
serialized in config** — read from the model source (resolved-config trap). Two
V-JEPA2 specifics to match *exactly* (don't "fix" to standard NeoX rope):

- Per-axis dim = `2*((head_dim//3)//2)` (head 64 → 20/20/20, **+4 passthrough**).
- `cos`/`sin` are **concat-tiled** (`emb = concat([e,e])`, index `i` uses `freq[i mod D/2]`),
  but the rotated half is **interleaved** (`y[2j]=-x[2j+1]`, `y[2j+1]=x[2j]`).
- Position ids decompose the **flattened** token index using the **config** grid
  size (not the actual input size): `frame=id//(g*g)`, `h=(id-g*g*frame)//g`, `w=rest`.

```python
def rotate(x, pos):                                  # x [...,N,D], pos [N] or [B,1,N]
    D = x.shape[-1]; om = 1.0/(10000.0**(mx.arange(D//2).astype(mx.float32)/(D/2.0)))
    f = pos.astype(mx.float32)[...,None]*om
    s = mx.concatenate([mx.sin(f),mx.sin(f)],-1); c = mx.concatenate([mx.cos(f),mx.cos(f)],-1)
    y = x.reshape(*x.shape[:-1], D//2, 2); y = mx.stack([-y[...,1], y[...,0]],-1).reshape(x.shape)
    return x*c + y*s
def apply_3d_rope(qk, pos_dhw, dims):                # qk [B,H,N,hd]; rotate d/h/w slices, cat remainder
    s=0; parts=[]
    for d,p in zip(dims, pos_dhw): parts.append(rotate(qk[...,s:s+d], p)); s+=d
    if s < qk.shape[-1]: parts.append(qk[...,s:])
    return mx.concatenate(parts,-1)
```
`mx.fast.rope` is 1-D only. The same `rotate` serves both the encoder (`pos` = `arange`,
broadcasts over B/H) and a masked predictor (`pos` = `[B,1,N]` token-index masks) —
generalize with `pos[...,None]`, not `pos[:,None]`.

## 4. 3-D tubelet patch embed (Conv3d) — `vjepa2-mlx`

`nn.Conv3d(kernel=stride=(tubelet,patch,patch))` over video. MLX Conv3d is **NDHWC**.

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, cfg):
        self.proj = nn.Conv3d(cfg.in_chans, cfg.hidden_size,
                              kernel_size=(cfg.tubelet_size,cfg.patch_size,cfg.patch_size),
                              stride=(cfg.tubelet_size,cfg.patch_size,cfg.patch_size))
    def __call__(self, video_bthwc):                 # permute upstream (B,T,C,H,W) -> (B,T,H,W,C)
        x = self.proj(video_bthwc); B,Dp,Hp,Wp,Cc = x.shape
        return x.reshape(B, Dp*Hp*Wp, Cc)            # token order d·H'·W' matches torch flatten(2).transpose
```
Weight transpose: torch Conv3d `(O,I,kT,kH,kW)` → MLX `(O,kT,kH,kW,I)` (already in
`weight-conversion.md`). Reshape `(B,D',H',W',C)→(B,N,C)` reproduces torch's
`flatten(2).transpose(1,2)` token order for free.

---

## Gating note

Video ViTs (V-JEPA2: encoder activations ~43, no final scaling) are another case
for **relative-error gating** — see `parity-testing.md`. fp16 holds for ViT
embeddings (rel ~5e-3, argmax stable) but **flow / coarse-to-fine nets are
fp16-sensitive** (RIFE fp16 diverged 3.1e-2 vs fp32 1.4e-3) — ship those fp32.
