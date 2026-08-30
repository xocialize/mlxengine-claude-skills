"""Canonical Core AI placement helpers — build it, pass it, prove it.

MEASURED 2026-08-29, macOS 27.0 (26A5421a), M5 Max, coreai-core==1.0.0b2.

Three API traps this module exists to neutralise. All were measured, not read:

1. ``ComputeUnitKind.cpu`` / ``.gpu`` / ``.neural_engine`` are **staticmethod
   factories, not constants**. You must CALL them. Passing the uncalled
   attribute raises ``RuntimeError: Invalid ComputeUnitKind in
   preferred_compute_unit_kind`` — loud, but easy to swallow in a try/except
   that then falls back to default placement and mislabels every number.

2. ``ComputeUnitKind.available_kinds()`` returns a **non-deterministic order
   across processes** (stable within one process, shuffled between them).
   Indexing it — ``available_kinds()[0]`` — selects a different compute unit
   on every run. Never index it; select by name.

3. ``from_preferred_compute_unit_kind`` sets a **preference**, and
   ``allowed_compute_unit_kinds`` stays all three. There is no
   ``neural_engine_only()``; ``cpu_only()`` is the only restriction primitive
   the API exposes. **Fallback off the ANE cannot be forbidden**, so a
   completed run is never evidence of ANE execution — you need the control
   lanes and stderr in ``run_lane`` below.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

VALID_LANES = ("cpu", "gpu", "ane")
_FACTORY = {"cpu": "cpu", "gpu": "gpu", "ane": "neural_engine"}
_EXPECT = {"cpu": "CPU", "gpu": "GPU", "ane": "Neural Engine"}


def options_for(lane: str):
    """SpecializationOptions for a lane, verified to actually carry it.

    Raises rather than returning something mislabeled. A caller that catches
    this and continues is producing false data.
    """
    if lane not in VALID_LANES:
        raise ValueError(f"lane must be one of {VALID_LANES}, got {lane!r}")
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if not SpecializationOptions.is_supported():
        raise SystemExit(
            "Delegate specialization unsupported here. Refusing to run: "
            "default placement would produce mislabeled data."
        )
    # CALL the factory. Do not pass the attribute. Do not index available_kinds().
    kind = getattr(ComputeUnitKind, _FACTORY[lane])()
    opts = SpecializationOptions.from_preferred_compute_unit_kind(kind)

    got = str(opts.preferred_compute_unit_kind)
    if got != _EXPECT[lane]:
        raise RuntimeError(
            f"asked for {_EXPECT[lane]!r}, options carry {got!r} — refusing to proceed"
        )
    return opts


def describe(opts) -> str:
    """Echo placement OFF THE OPTIONS OBJECT, never off an argv string."""
    allowed = sorted(str(a) for a in opts.allowed_compute_unit_kinds)
    return f"preferred={opts.preferred_compute_unit_kind} allowed={allowed}"


class _NativeCapture:
    """Handle yielded by capture_native(). Call .read() at any time."""

    def __init__(self, tmp):
        self._tmp = tmp

    def read(self) -> str:
        """Flush the native fds, then read everything written so far.

        Must seek explicitly: the file position sits at the END after writes,
        so a bare read() returns "" — a bug that silently reported 0 diagnostics
        on a run that emitted 96 of them.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        pos = self._tmp.tell()
        self._tmp.seek(0)
        data = self._tmp.read()
        self._tmp.seek(pos)
        return data


@contextlib.contextmanager
def capture_native():
    """Capture BOTH fd 1 and fd 2 at the FILE-DESCRIPTOR level.

    MEASURED 2026-08-29, and it invalidated two earlier versions of this module:

    1. The ANE validation diagnostics are written by NATIVE code straight to the
       process file descriptors, and some go to STDOUT rather than stderr.
       `contextlib.redirect_stderr` only rebinds Python's `sys.stderr` object,
       so it captures NONE of them.
    2. Reading the temp file inside the with-block without seeking returns "" —
       the position is at the end. Use the yielded object's .read().

    Usage:
        with capture_native() as cap:
            ...                      # convert / load / run
            text = cap.read()        # any time, inside or before exit
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        sys.stdout.flush()
        sys.stderr.flush()
        so, se = os.dup(1), os.dup(2)
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield _NativeCapture(tmp)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(so)
            os.close(se)


_ANE_FAIL = re.compile(r"ANECCompile\(\)\s*FAILED|_ANECompiler", re.I)
_ANE_VALID = re.compile(r"ane_validation_message|ANE I/O op|ane_?validation", re.I)
_ANE_REJECT = re.compile(r"failed: ([^\"<>]{6,120})")


def scan_stderr(text: str) -> dict:
    """Extract the ANE signals buried in megabytes of MLIR `warning: loc(...)`."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("warning: loc")]
    # The interesting text is buried in megabytes of MLIR; pull the reasons and
    # the torch op identifiers out rather than keeping the blobs.
    reasons = {}
    for m in _ANE_REJECT.finditer(text):
        reasons[m.group(1).strip()] = reasons.get(m.group(1).strip(), 0) + 1
    ops = sorted(set(re.findall(r'string<"([a-z_][a-z_0-9]*)"', text)))
    return {
        "anecompile_failed": bool(_ANE_FAIL.search(text)),
        "n_validation_hits": len(_ANE_VALID.findall(text)),
        "reject_reasons": reasons,
        "rejected_ops": ops[:25],
        "errors": [ln.strip() for ln in lines if "error" in ln.lower()][:20],
    }


def cache_dir() -> Path:
    """Specialization cache. It caches FAILURE-then-fallback: an ANE attempt
    that prints ANECCompile() FAILED and drops to GPU is cached AS GPU, and
    every later load is fast, silent and wrong. Clear to re-diagnose."""
    return Path.home() / "Library/Caches/coreai-cache"


async def run_lane(asset: Path, lane: str, feed, *, iters: int = 20, warmup: int = 3):
    """Load on `lane`, time it, and return evidence rather than a claim.

    `feed` is a zero-arg callable returning the inputs dict for one call.
    Returns a dict carrying the echoed placement, cold-load cost, median
    latency, and the stderr scan — everything a receipt needs.
    """
    from coreai.runtime import AIModel

    opts = options_for(lane)
    err = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(err):
        model = await AIModel.load(asset, specialization_options=opts)
        fn = model.load_function(next(iter(model.function_names)))
    cold_load_s = time.perf_counter() - t0

    for _ in range(warmup):
        await fn(feed())
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        await fn(feed())
        times.append((time.perf_counter() - t) * 1000.0)

    return {
        "lane": lane,
        "placement_echoed": describe(opts),
        "cold_load_s": round(cold_load_s, 3),
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "stderr": scan_stderr(err.getvalue()),
    }


def verdict(results: dict) -> list[str]:
    """Flag the failure modes that make a number a lie.

    Two lanes with equal latency ARE the same lane. This is the check that
    catches an inert placement flag, and it has caught one before.
    """
    out = []
    have = {k: v["median_ms"] for k, v in results.items() if v}
    for a in have:
        for b in have:
            if a < b and abs(have[a] - have[b]) / max(have[a], have[b]) < 0.05:
                out.append(
                    f"SUSPECT: {a} ({have[a]} ms) and {b} ({have[b]} ms) agree within 5% "
                    f"— two lanes with equal latency are the same lane"
                )
    if "ane" in results and results["ane"]:
        r = results["ane"]
        # accept either {"stderr": scan} or the scan fields inlined
        s = r.get("stderr") or r
        cold = r.get("cold_load_s")
        if cold is not None and cold < 1.0:
            out.append(
                f"SUSPECT: ANE cold load was {cold:.2f}s. Real ANE specialization is SECONDS "
                f"(measured 7-9 s for a 1.2M convnet, 254 s for a 226M UNet). A sub-second cold "
                f"load on a cold cache means the graph was NOT compiled for the ANE."
            )
        if s.get("reject_reasons"):
            for reason, n in s["reject_reasons"].items():
                out.append(f"ANE REJECTED {n}x: {reason}")
        if s.get("anecompile_failed"):
            out.append(
                "ANECCompile() FAILED on the ANE lane — this ran on the GPU with an "
                "'ANE' label. The cache has now cached that fallback; clear it to re-diagnose."
            )
        if s.get("n_validation_hits"):
            out.append(f"{s['n_validation_hits']} ANE validation hit(s) in captured native output")
    if not out:
        out.append("No contradiction found. This is NOT proof of residency — "
                   "cross-check the GPU idle clock (macmon) during sustained inference.")
    return out


if __name__ == "__main__":
    from coreai.runtime import ComputeUnitKind as K, SpecializationOptions as SO
    print(f"is_supported: {SO.is_supported()}")
    print(f"available_kinds() THIS process: {[str(k) for k in K.available_kinds()]}")
    print("  ^ order is non-deterministic across processes — never index this")
    for lane in VALID_LANES:
        print(f"  {lane:4s} -> {describe(options_for(lane))}")
    print(f"cache: {cache_dir()}")
