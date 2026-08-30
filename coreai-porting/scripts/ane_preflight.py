"""ANE pre-flight — answer "is this graph ANE-resident?" BEFORE committing to a port.

The routing question that actually matters is not "CoreAI or MLX", it is "is this
graph ANE-resident, and does that matter for this workload?" Only the first half
has a capability answer, and it is measurable in about a minute.

Usage:
    from ane_preflight import preflight
    report = await preflight(model, {"x": example_tensor}, dtype="fp16")

Returns a verdict plus the exact ops the ANE rejected, so a no is actionable
rather than just discouraging.

WHY THIS WORKS
    ANE validation fires at LOAD, naming the failing torch ops with source
    attribution. Validation hits are the RELIABLE signal and they do reappear on
    later loads.

    Cold-load DURATION is a weaker, one-shot signal: a first-ever load of a new
    asset specializes for seconds, but that is NOT restored by clearing
    ~/Library/Caches/coreai-cache or ~/Library/Caches/com.apple.e5rt.e5bundlecache.
    MEASURED: a never-loaded asset took 3.25 s then 0.00 s; a much-loaded asset
    stayed at 0.18 s no matter what was cleared. To get a fresh duration reading,
    RE-EXPORT TO A NEW PATH. Never treat a fast load on a known asset as evidence.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import placement

# MEASURED rejections, accumulated across our ports. Not exhaustive -- upstream
# moves, and eligibility is SCALE-DEPENDENT (see the 16^2 vs 128^2 finding in
# case-deformable-conv.md). Treat as a hint list, never as a substitute for a run.
KNOWN_REJECTIONS = {
    "gather": "scattered read — deformable conv, grid sample, any learned-offset sampling",
    "bitwise_and": "bool mask chain; also SILENTLY MISCOMPILES on CPU (coreai-torch#74)",
    "slice_scatter": "in-place write into a slice, e.g. mask[:, :n, m:] = ...",
    "scaled_dot_product_attention": "preserved as a composite, but NOT always ANE-eligible in fp16",
    "index_put": "scatter by index",
    "nonzero": "data-dependent output shape",
}

# Ops that block CAPTURE rather than ANE placement — different failure, different fix.
CAPTURE_BLOCKERS = ("item", "nonzero", "_local_scalar_dense")


def graph_prescan(exported_program) -> dict:
    """Static checks that need no load. Catches rank>5 and host reads early."""
    max_rank, ranked = 0, None
    host_reads = []
    ops = set()
    for n in exported_program.graph.nodes:
        if n.op != "call_function":
            continue
        t = str(n.target)
        ops.add(t.split(".")[-2] if "." in t else t)
        v = n.meta.get("val")
        if hasattr(v, "shape") and len(v.shape) > max_rank:
            max_rank, ranked = len(v.shape), t
        if any(b in t for b in CAPTURE_BLOCKERS):
            host_reads.append(t)
    hints = {o: why for o, why in KNOWN_REJECTIONS.items()
             if any(o in x for x in ops)}
    return {
        "max_tensor_rank": max_rank,
        "max_rank_op": ranked,
        "rank_ok": max_rank <= 5,
        "host_reads": host_reads,
        "suspect_ops": hints,
    }


def clear_cache() -> None:
    """The specialization cache caches failure-then-fallback. Diagnostics exist
    on the cold run only."""
    shutil.rmtree(Path.home() / "Library/Caches/coreai-cache", ignore_errors=True)


async def probe_asset(asset: str | Path, feed) -> dict:
    """Load on the ANE lane with a cold cache and report what it rejected."""
    from coreai.runtime import AIModel

    clear_cache()
    opts = placement.options_for("ane")
    with placement.capture_native() as cap:
        t0 = time.perf_counter()
        model = await AIModel.load(str(asset), specialization_options=opts)
        fn = model.load_function(next(iter(model.function_names)))
        cold = time.perf_counter() - t0
        await fn(feed())
        scan = placement.scan_stderr(cap.read())

    rejected = [o for o in scan["rejected_ops"] if o not in ("torch", "coreai")]
    n = scan["n_validation_hits"]
    if n == 0 and cold >= 1.0:
        verdict = "ANE-RESIDENT (probable) — no rejections, and the load specialized"
    elif n == 0:
        verdict = ("NO REJECTIONS — likely resident. Load was fast, but that is only "
                   "meaningful on a first-EVER load of a new asset; confirm with the GPU-idle oracle")
    elif rejected:
        verdict = "PARTIAL / NOT RESIDENT — graph partitions and falls back"
    else:
        verdict = "NOT RESIDENT"
    return {
        "verdict": verdict,
        "cold_load_s": round(cold, 2),
        "n_validation_hits": n,
        "rejected_ops": rejected,
        "reasons": scan["reject_reasons"],
        "why": {o: w for o, w in KNOWN_REJECTIONS.items()
                if any(o in r for r in rejected)},
    }


def render(pre: dict, probe: dict | None = None) -> str:
    L = ["ANE PRE-FLIGHT", "=" * 52]
    L.append(f"max tensor rank : {pre['max_tensor_rank']} ({'OK' if pre['rank_ok'] else 'REJECTED — ANE limit is 5'})")
    if not pre["rank_ok"]:
        L.append(f"  worst op      : {pre['max_rank_op']}")
    if pre["host_reads"]:
        L.append(f"host reads      : {len(pre['host_reads'])} — CAPTURE will fail, not just placement")
        for h in pre["host_reads"][:4]:
            L.append(f"  {h}")
    if pre["suspect_ops"]:
        L.append("suspect ops (from measured rejections):")
        for o, w in pre["suspect_ops"].items():
            L.append(f"  {o:32s} {w}")
    if probe:
        L += ["-" * 52, f"VERDICT         : {probe['verdict']}",
              f"cold load       : {probe['cold_load_s']} s",
              f"validation hits : {probe['n_validation_hits']}"]
        if probe["rejected_ops"]:
            L.append(f"rejected ops    : {', '.join(probe['rejected_ops'][:10])}")
        for o, w in probe["why"].items():
            L.append(f"  {o:32s} {w}")
    return "\n".join(L)
