#!/usr/bin/env python3
"""Prove ANE residency for a .aimodel — with the oracle's own control built in.

    python residency.py --asset model.aimodel

Why this exists, in one paragraph: a compute-unit *preference* is not a placement,
zero ANE validation hits is not proof of residency (hits are a reliable NEGATIVE,
never a positive), and a harness that does not check its worker's exit status will
happily print a verdict for a process that already died. All three produced wrong
published numbers before this script existed. It bakes in the fixes.

The oracle is the GPU idle clock via `macmon`: an ANE-resident graph leaves the GPU
at its idle floor (~338 MHz on M5 Max) while a GPU-executing graph lifts it. Because
an absent signal is being read as a positive, the script ALWAYS runs a GPU-lane
control through the identical harness and refuses to report anything if that control
does not read busy — an oracle that cannot say "no" is not evidence.

Budget: a process dies after 2^14 output NDArray allocations (coreai-torch#75), i.e.
floor(16384 / len(output_names)) inferences, warmup included. The default paced run
stays well under it; the script computes the cap from the asset and says so.
"""
import argparse, asyncio, json, os, subprocess, sys, time
from pathlib import Path

CAP = 16384                 # coreai-torch#75, per process
IDLE_MHZ_DEFAULT = 500      # below this = idle; M5 Max floor measured at 338


# ───────────────────────────── worker ─────────────────────────────

async def _worker(a):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    import placement
    from coreai.runtime import AIModel, NDArray

    opts = placement.options_for(a.lane)
    model = await AIModel.load(a.asset, specialization_options=opts)
    fn = model.load_function(next(iter(model.function_names)))
    d = fn.desc

    # Build inputs from the asset's own descriptors -- no guessing at names/shapes.
    feed = {}
    for n in d.input_names:
        idesc = d.input_descriptor(n)
        feed[n] = NDArray(np.random.rand(*idesc.shape).astype(idesc.dtype))
    n_out = max(1, len(d.output_names))

    for _ in range(a.warmup):
        await fn(feed)

    hb = open(a.heartbeat, "w") if a.heartbeat else None
    if a.ready:                       # tell the driver the steady loop is starting,
        Path(a.ready).write_text("1")  # so macmon samples the loop, not the load.

    limit = a.max if a.to_failure else a.iters
    t0, n = time.perf_counter(), 0
    while n < limit:
        if a.seconds:                 # paced: hold a wall-clock window open on a
            due = t0 + (n / limit) * a.seconds   # hard inference budget
            slack = due - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)
            if time.perf_counter() - t0 >= a.seconds:
                break
        await fn(feed)
        n += 1
        # The #75 SIGTRAP is uncatchable: the count must be durable BEFORE the
        # call that dies, so fsync -- coarsely, then every iteration near the cap.
        if hb and (n % 100 == 0 or n >= a.fine_from):
            hb.seek(0); hb.write(f"{n}\n"); hb.flush(); os.fsync(hb.fileno())
    dt = time.perf_counter() - t0
    if hb:
        hb.seek(0); hb.write(f"{n}\n"); hb.flush(); os.fsync(hb.fileno()); hb.close()
    print(json.dumps({"ok": True, "iters": n, "rate": n / dt if dt else 0,
                      "n_out": n_out, "inputs": list(d.input_names),
                      "outputs": list(d.output_names)}), flush=True)


# ───────────────────────────── driver ─────────────────────────────

def _sample_gpu_mhz(seconds, interval_ms=800):
    """Median GPU clock over a window. Returns None if macmon is unavailable."""
    n = max(2, int(seconds * 1000 / interval_ms))
    try:
        out = subprocess.run(["macmon", "pipe", "-s", str(n), "-i", str(interval_ms)],
                             capture_output=True, text=True, timeout=seconds + 20).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(json.loads(line).get("gpu_usage", [0, 0])[0])
        except Exception:
            pass
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def _run_lane(a, lane, tmp):
    """Spawn a worker on one lane and sample the GPU clock during its steady loop."""
    ready = tmp / f"ready_{lane}"
    ready.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, __file__, "--worker", "--asset", a.asset, "--lane", lane,
         "--iters", str(a.iters), "--seconds", str(a.seconds),
         "--warmup", str(a.warmup), "--ready", str(ready)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    deadline = time.time() + 300      # first load can pay E5RT specialization
    while not ready.exists() and proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    mhz = _sample_gpu_mhz(min(a.seconds * 0.7, 6)) if proc.poll() is None else None
    out, err = proc.communicate(timeout=600)

    payload = None
    for line in out.splitlines():                 # coreai prints a banner to stdout
        try:
            payload = json.loads(line)
        except Exception:
            pass
    return {"lane": lane, "rc": proc.returncode, "mhz": mhz,
            "payload": payload, "stderr": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--lane", default="ane")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--idle-mhz", type=int, default=IDLE_MHZ_DEFAULT)
    ap.add_argument("--to-failure", action="store_true",
                    help="unbounded run to locate the #75 cap (worker mode)")
    ap.add_argument("--max", type=int, default=60000)
    ap.add_argument("--fine-from", type=int, default=10**9)
    ap.add_argument("--heartbeat", default="")
    ap.add_argument("--ready", default="")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        asyncio.run(_worker(a))
        return

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "coreai-residency"
    tmp.mkdir(parents=True, exist_ok=True)

    test = _run_lane(a, a.lane, tmp)
    control = _run_lane(a, "gpu", tmp) if a.lane != "gpu" else None

    # ---- refuse to report on anything that did not survive ----
    for r in filter(None, (test, control)):
        if r["rc"] != 0 or r["payload"] is None:
            print(f"NO VERDICT — the {r['lane']} lane did not survive "
                  f"(rc={r['rc']}). A GPU clock reading from a dead process is "
                  f"the idle floor, not residency.")
            tail = [l for l in r["stderr"].splitlines() if l.strip()][-3:]
            for l in tail:
                print("   ", l[:160])
            # subprocess.returncode is NEGATIVE for a signal (-5 = SIGTRAP);
            # a shell reports the same death as 128+5 = 133. Accept both.
            rc = r["rc"]
            if rc < 0 or rc > 128:
                sig = -rc if rc < 0 else rc - 128
                print(f"    died on signal {sig}"
                      + ("  — SIGTRAP: see coreai-torch#75, the 2^14 "
                         "output-allocation cap." if sig == 5 else ""))
            sys.exit(2)

    p = test["payload"]
    cap = CAP // max(1, p["n_out"])
    print(f"asset      {Path(a.asset).name}")
    print(f"inputs     {p['inputs']}   outputs {p['outputs']} "
          f"(n={p['n_out']})  ->  #75 cap {cap} inferences/process")
    print(f"ran        {p['iters']} inferences on the {a.lane} lane "
          f"({p['iters'] * p['n_out']}/{CAP} of budget)")
    print()

    if test["mhz"] is None:
        print("NO VERDICT — macmon unavailable, so the residency oracle could not run.")
        sys.exit(2)

    # ---- verify the verifier: the oracle must still be able to say "no" ----
    if control is not None:
        if control["mhz"] is None or control["mhz"] < a.idle_mhz:
            print(f"NO VERDICT — the GPU-lane control read "
                  f"{control['mhz']} MHz, i.e. idle. Under this pacing the oracle "
                  f"cannot distinguish lanes, so a low reading on the test lane "
                  f"proves nothing. Raise --iters or lower --seconds and re-run.")
            sys.exit(2)
        print(f"control    gpu lane {control['mhz']} MHz = busy — oracle discriminates")

    verdict = "ANE-RESIDENT" if test["mhz"] < a.idle_mhz else "NOT RESIDENT (GPU busy)"
    print(f"test       {a.lane} lane {test['mhz']} MHz -> {verdict}")


if __name__ == "__main__":
    main()
