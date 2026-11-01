"""Study: does background prefetch shorten an exact_lm data pass over an HDF5 source?

Times ``--passes`` full data passes of ``nlsq.streaming.exact_lm.run_pass`` over an
h5py-backed ``(x, y)`` source with ``_PREFETCH`` off and on (interleaved, order
alternating, one untimed warm-up pass each), prints per-pass wall times and medians and
the machine state. Keep prefetch only if the median pass time is >= 10% lower.

    python benchmarks/studies/exact_prefetch.py --n 100000000 --path /data/exp.h5

The file (datasets ``x``, ``y``, float64, chunked, exp_decay data, fixed seed) is written
in slabs and reused if its ``n``/``seed`` attributes match. Page-cache state changes the
numbers: compare runs only against runs with the same free RAM / file residency.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED, SLAB = 0, 1 << 22
P_TRUE, P0, X_RANGE, NOISE = (2.5, 1.3, 0.5), (1.0, 1.0, 0.0), (0.0, 4.0), 0.05


def _h5py():
    try:
        import h5py
    except ImportError:
        raise SystemExit(
            "h5py is required for this study (not a project dependency): "
            "install it in the environment running the study, e.g. `pip install h5py`."
        ) from None
    return h5py


def make_file(path: Path, n: int) -> None:
    """Write ``x``, ``y`` slab by slab (memory stays at one slab) unless ``path`` matches."""
    h5py = _h5py()
    if path.exists():
        with h5py.File(path, "r") as f:
            if (f.attrs.get("n"), f.attrs.get("seed")) == (n, SEED) and f.attrs.get(
                "complete"
            ):
                print(f"reusing {path}")
                return
    print(f"writing {path} ({n:,} points)")
    a, b, c = P_TRUE
    with h5py.File(path, "w") as f:
        chunks = (min(n, 1 << 20),)
        dx = f.create_dataset("x", (n,), np.float64, chunks=chunks)
        dy = f.create_dataset("y", (n,), np.float64, chunks=chunks)
        for k, s in enumerate(range(0, n, SLAB)):
            e = min(s + SLAB, n)
            x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * np.arange(s, e) / (n - 1)
            noise = np.random.default_rng([SEED, k]).standard_normal(e - s)
            dx[s:e], dy[s:e] = x, a * np.exp(-b * x) + c + NOISE * noise
        f.attrs.update(n=n, seed=SEED, complete=True)


def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def machine_state() -> dict[str, str]:
    import jax

    def read(p: str) -> str:
        try:
            return Path(p).read_text().strip()
        except OSError:
            return "unavailable"

    mem = {
        k: v.split()[0]
        for k, v in (
            line.split(":", 1)
            for line in read("/proc/meminfo").splitlines()
            if ":" in line
        )
    }
    gpu = (
        _sh(
            "nvidia-smi",
            "--query-gpu=name,clocks.sm,clocks.max.sm,utilization.gpu",
            "--format=csv,noheader",
        )
        if shutil.which("nvidia-smi")
        else "nvidia-smi not found"
    )
    return {
        "platform": platform.platform(),
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "load_average": " ".join(f"{v:.2f}" for v in os.getloadavg()),
        "mem_free_available_kB": f"{mem.get('MemFree')} / {mem.get('MemAvailable')}",
        "page_cache_kB": mem.get("Cached", "unavailable"),
        "gpu": gpu,
        "jax_backend": f"{jax.default_backend()} {jax.devices()}",
        "jax_version": jax.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000_000)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--path", type=Path, default=Path("exact_prefetch.h5"))
    ap.add_argument("--chunk-size", type=int, default=1 << 22)
    ap.add_argument("--prefetch", choices=("on", "off", "both"), default="both")
    args = ap.parse_args()

    h5py = _h5py()
    import jax.numpy as jnp

    import nlsq.streaming.exact_lm as lm
    from benchmarks.compare.problems import PROBLEMS

    exp = PROBLEMS["exp_decay"]
    make_file(args.path, args.n)
    stats = lm.make_chunk_stats(lambda x, a, b, c: exp.model(jnp, x, a, b, c))
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.prefetch]
    times: dict[bool, list[float]] = {m: [] for m in modes}

    print("machine state (before):")
    for k, v in machine_state().items():
        print(f"  {k}: {v}")
    with h5py.File(args.path, "r") as f:
        source = (f["x"], f["y"], None)
        chunk = min(args.chunk_size, args.n)

        def one_pass(flag: bool) -> float:
            lm._PREFETCH = flag
            t = time.perf_counter()
            st = lm.run_pass(stats, P0, source, chunk)
            dt = time.perf_counter() - t
            assert st.n_seen == args.n
            return dt

        for flag in modes:  # untimed: JIT compile and first read of the file
            one_pass(flag)
        for i in range(args.passes):
            for flag in modes if i % 2 == 0 else modes[::-1]:
                times[flag].append(one_pass(flag))
        lm._PREFETCH = True
    print("machine state (after):")
    for k, v in machine_state().items():
        print(f"  {k}: {v}")

    med = {m: statistics.median(t) for m, t in times.items()}
    for m in modes:
        print(
            f"prefetch={'on' if m else 'off':3s} n={args.n:,} chunk={chunk:,} "
            f"passes(s)={[round(t, 3) for t in times[m]]} median={med[m]:.3f}s"
        )
    if len(modes) == 2:
        gain = 1.0 - med[True] / med[False]
        print(
            f"median pass time change with prefetch: {-100 * gain:+.1f}% "
            f"-> {'KEEP (>=10% faster)' if gain >= 0.10 else 'DROP (<10% faster)'}"
        )


if __name__ == "__main__":
    main()
