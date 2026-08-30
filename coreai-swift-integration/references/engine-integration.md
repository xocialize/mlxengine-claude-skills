# Engine integration — a CoreAI core inside MLXEngine

---

## MLXToolKit only

**MEASURED.** An MLXEngine `ModelPackage` wrapping a CoreAI core depends on **MLXToolKit alone**
— the engine's dependency-free contract layer. **No MLX is linked.**

This is the whole reason the engine can host both runtimes: the coordinator's contract is about
lifecycle and capability, not about arithmetic.

---

## The macOS-26 host / macOS-27 package conflict

**MEASURED.** CoreAI requires macOS 27. A host package with a macOS 26 floor **cannot** declare
a dependency on a 27-floored package — SPM refuses at resolution time, not at runtime.

Raising the host's floor to 27 is the wrong fix: it drags every consumer of that host onto 27,
including the MLX packages that don't need it.

### The ExternalRegistration seam

```text
App (deployment target decides)
  └── if #available(macOS 27) { ForgeCore.ExternalRegistration.register { CoreAIBackend() } }
        │
ForgeCore (macOS 26 floor)  ← never names the CoreAI package
  └── holds a registration closure slot
        │
coreai-<model>-swift (macOS 27 floor)  ← linked ONLY by the app
  └── registers beside the MLX sibling under its own PackageID
```

The app's own deployment target is the only place that knows whether CoreAI exists. Everything
below stays portable.

**Result:** the CoreAI backend appears in the engine's registry alongside the MLX sibling for
the same capability, distinguished by `PackageID`.

---

## Two packages, one capability

MEASURED once (Real-ESRGAN): the CoreAI and MLX packages serve **different tiers of the same
capability** — CoreAI/ANE for the low-energy fixed-tile path, MLX/GPU for the flexible-geometry
whole-frame path.

**Documentation duty:** the two ports have **inverted geometry contracts** — CoreAI tile
geometry is build-time (one executable per static shape); MLX geometry is runtime-injectable.
**Say so in both packages**, or a maintainer will eventually "fix" one toward the other and
break it.

→ `coreai-porting`'s `mlx-vs-coreai-fit.md` for how to decide which tier a model belongs on.

---

## OPEN

- **Multi-package capability routing by `PackageID`** — how the engine should *choose* between
  the CoreAI and MLX package at runtime (energy state? geometry? host memory pressure?) is
  **undesigned**. One shipped pair is not enough to generalize.
- **QuantFootprint equivalent for CoreAI.** The MLX side declares measured resident + activation
  bytes. CoreAI's memory shape is fundamentally different — activations stay on-die, so the
  host-side number is small and the meaningful budget is elsewhere. **What the engine should be
  told is an open question.**
- Cancellation (the MLX side's CAN gate) and auto-materialization (MAT gate) have **no CoreAI
  analogue yet**.
