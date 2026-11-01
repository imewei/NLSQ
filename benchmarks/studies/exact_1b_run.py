"""Stage 3 manual gates: exact_streaming on exp_decay at up to 1e9 points from HDF5.

Usage::

    python -m benchmarks.studies.exact_1b_run generate PATH [--n N] [--noise S]
        [--seed 0] [--chunk-rows 4194304]
    python -m benchmarks.studies.exact_1b_run run --data-dir DIR [--n N] [--noise 0.05]
        [--seed 0] [--chunk-rows 4194304] [--out results.json]

The problem is ``benchmarks.compare.problems.PROBLEMS["exp_decay"]`` (the 100M benchmark
problem): ``x`` bitwise equal to ``np.linspace(0, 4, n)``, ``y = f(x; 2.5, 1.3, 0.5)``
plus ``noise * N(0, 1)``. Chunk ``k`` draws its noise from
``SeedSequence(seed, spawn_key=(k,))``, so the file depends on ``(n, seed, noise,
chunk_rows)`` only, never on write order, and is written without holding more than one
chunk. A file is written to ``PATH.tmp`` and renamed after its fingerprint attribute is
set: an existing ``PATH`` with a matching fingerprint is reused, any other is refused.

``run`` writes (or reuses) a noiseless and a noisy file in ``--data-dir`` and executes
the spec's gates ("Gates for the 1B run"): (1) noiseless recovery ``rtol 1e-6`` and cost
``< 1e-6 n``; (2) noisy ``|theta - theta_true| <= 5 sqrt(diag pcov)``; (3i) every pass
sees ``n`` points; (3ii) :func:`streaming_cost` equals the solver's cost to relative
``1e-9`` (on the noisy fit; the noiseless cost is ulp-level rounding, where host and
device ``exp`` differ, so it is reported only); (3iii) the sentinel; (4) peak RSS
``< 16 n B + 4 GiB``; (5) wall time, passes and per-pass throughput (reported). Solver
chunks equal ``--chunk-rows``.

Pass counts are not exposed by the solver (``n_points`` is ``len(x)``; ``n_seen`` is
checked on every pass), so the script observes every pass by rebinding
``nlsq.streaming.exact_lm.run_pass`` during each fit and by counting the ``y`` rows read
through :class:`TailOffsetY`. Costs: ``run`` adds one pass for ``H`` at the generating
parameters and one host pass for :func:`streaming_cost`.
"""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare.problems import PROBLEMS
from benchmarks.studies.conditioning_helpers import rel_size

PROBLEM = PROBLEMS["exp_decay"]
THETA_TRUE = np.asarray(PROBLEM.p_true, dtype=np.float64)
SENTINEL_ROWS = 1000
FORMAT_VERSION = 1
BYTES_PER_POINT = 16  # x and y float64


def model_np(x: np.ndarray, *theta: float) -> np.ndarray:
    return PROBLEM.model(np, x, *theta)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- data


def chunk_data(
    k: int, n: int, chunk_rows: int, seed: int, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``[k*chunk_rows, min((k+1)*chunk_rows, n))`` of ``(x, y)``."""
    s, e = k * chunk_rows, min((k + 1) * chunk_rows, n)
    lo, hi = PROBLEM.x_range
    x = np.arange(s, e, dtype=np.float64) * ((hi - lo) / (n - 1)) + lo  # as linspace
    if e == n:
        x[-1] = hi
    y = model_np(x, *THETA_TRUE)
    if noise:
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))
        y = y + noise * rng.standard_normal(e - s)
    return x, y


def fingerprint(n: int, seed: int, noise: float, chunk_rows: int) -> str:
    return json.dumps(
        {
            "format": FORMAT_VERSION,
            "model": PROBLEM.name,
            "p_true": list(PROBLEM.p_true),
            "x_range": list(PROBLEM.x_range),
            "n": n,
            "seed": seed,
            "noise": noise,
            "chunk_rows": chunk_rows,
        },
        sort_keys=True,
    )


def write_dataset(
    path: str | os.PathLike[str],
    n: int,
    seed: int,
    noise: float,
    chunk_rows: int,
    order: Sequence[int] | None = None,
) -> str:
    """Write ``x, y`` to HDF5 at ``path``; ``"exists"`` if a matching file is there.

    Raises ``FileExistsError`` for an existing file with another (or no) fingerprint and
    ``OSError`` when free disk space is below 1.25x the data size. ``order`` (chunk
    indices) only exists to test that the content is independent of write order.
    """
    import h5py  # type: ignore[import-untyped,import-not-found]

    if n < 2 or chunk_rows < 1:
        raise ValueError(f"need n >= 2 and chunk_rows >= 1, got {n}, {chunk_rows}")
    path = Path(path)
    fp = fingerprint(n, seed, noise, chunk_rows)
    if path.exists():
        try:
            with h5py.File(path, "r") as f:
                found = f.attrs.get("fingerprint")
        except OSError:
            found = None
        if found == fp:
            return "exists"
        raise FileExistsError(
            f"{path} exists with fingerprint {found!r}, expected {fp}; refusing to "
            "overwrite (delete it to regenerate)"
        )
    need = BYTES_PER_POINT * n
    free = shutil.disk_usage(path.parent).free
    if free < 1.25 * need:
        raise OSError(
            f"not enough disk space in {path.parent}: {free / 1e9:.1f} GB free, need "
            f"1.25 x {need / 1e9:.1f} GB"
        )
    nchunks = -(-n // chunk_rows)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        dx = f.create_dataset("x", (n,), "f8", chunks=(min(chunk_rows, n),))
        dy = f.create_dataset("y", (n,), "f8", chunks=(min(chunk_rows, n),))
        for k in range(nchunks) if order is None else order:
            x, y = chunk_data(k, n, chunk_rows, seed, noise)
            s = k * chunk_rows
            dx[s : s + len(x)] = x
            dy[s : s + len(y)] = y
        f.attrs["fingerprint"] = fp  # last: only a complete file carries it
    os.replace(tmp, path)
    return "written"


class TailOffsetY:
    """``y`` with ``offset`` added to its last ``rows`` points, read lazily by slice.

    Not an ``np.ndarray`` (the solver streams it, never copies it whole); counts the rows
    it serves in ``rows_read``.
    """

    def __init__(self, y: Any, offset: float = 0.0, rows: int = SENTINEL_ROWS):
        self.y, self.offset, self.rows = y, float(offset), rows
        self.shape = (int(y.shape[0]),)
        self.ndim, self.dtype = 1, np.dtype(np.float64)
        self.rows_read = 0

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, sl: slice) -> np.ndarray:
        if not isinstance(sl, slice) or sl.step not in (None, 1):
            raise TypeError("TailOffsetY supports contiguous slices only")
        n = self.shape[0]
        s, e, _ = sl.indices(n)
        out = np.asarray(self.y[s:e], dtype=np.float64)
        self.rows_read += max(e - s, 0)
        t = max(n - self.rows, s)
        if self.offset and t < e:
            out = out.copy()
            out[t - s :] += self.offset
        return out


def streaming_cost(
    source: tuple[Any, ...],
    model_np: Callable[..., np.ndarray],
    theta: Sequence[float] | np.ndarray,
    chunk: int,
) -> float:
    """``0.5 * sum(((y - model_np(x, *theta)) / sigma)**2)`` read chunk by chunk.

    ``source`` is ``(x, y)`` or ``(x, y, sigma)`` of sliceable 1-D sources with
    ``shape``; the convention is the solver's ``F`` (``PassStats.cost``). Pure NumPy
    float64; holds one chunk and its residual at a time.
    """
    x, y = source[0], source[1]
    sigma = source[2] if len(source) > 2 else None
    n = int(x.shape[0])
    th = [float(t) for t in theta]
    total = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xb = np.asarray(x[s:e], dtype=np.float64)
        r = np.asarray(y[s:e], dtype=np.float64) - model_np(xb, *th)
        if sigma is not None:
            r /= np.asarray(sigma[s:e], dtype=np.float64)
        total += 0.5 * float(np.dot(r, r))
    return total


# ----------------------------------------------------------------------- sentinel


def model_jnp(x: Any, *theta: Any) -> Any:
    import jax.numpy as jnp

    return PROBLEM.model(jnp, x, *theta)


def hessian_at(theta: np.ndarray, x: Any, y: Any, chunk: int) -> tuple[np.ndarray, int]:
    """``(J^T J, n_seen)`` at ``theta`` from one module pass (``H`` does not depend on y)."""
    from nlsq.streaming import exact_lm

    st = exact_lm.run_pass(
        exact_lm.make_chunk_stats(model_jnp), theta, (x, y, None), chunk
    )
    return st.H, st.n_seen


def calibrate_sentinel(
    theta_true: np.ndarray,
    x_block: np.ndarray,
    H: np.ndarray,
    target_rel: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """``(delta, predicted)``: offset of ``y`` on ``x_block`` and its displacement.

    With ``J_s`` the Jacobian of the block at ``theta_true`` and ``r = y - f``, adding
    ``delta`` to the block's ``y`` moves the linearized optimum by
    ``predicted = H^-1 J_s^T (delta * 1)``; ``delta`` is chosen so that
    ``rel_size(predicted, theta_true) == target_rel`` (the displacement is linear in
    ``delta``, so one evaluation at ``delta = 1`` fixes it).
    """
    import jax
    import jax.numpy as jnp

    xb = jnp.asarray(x_block, dtype=jnp.float64)
    th = jnp.asarray(theta_true, dtype=jnp.float64)
    Js = np.asarray(jax.jacfwd(lambda q: model_jnp(xb, *q))(th))
    unit = np.linalg.solve(H, Js.T @ np.ones(len(x_block)))
    delta = target_rel / rel_size(unit, theta_true)
    return delta, delta * unit


# ---------------------------------------------------------------- instrumentation


@contextmanager
def record_passes():
    """Rebind ``exact_lm.run_pass`` to log ``(n_seen, seconds)`` of every pass."""
    from nlsq.streaming import exact_lm

    orig, log = exact_lm.run_pass, []

    def wrapped(*a: Any, **k: Any) -> Any:
        t = time.perf_counter()
        st = orig(*a, **k)
        log.append((st.n_seen, time.perf_counter() - t))
        return st

    exact_lm.run_pass = wrapped
    try:
        yield log
    finally:
        exact_lm.run_pass = orig


def peak_rss_bytes() -> int:
    """Peak resident set: max of ``ru_maxrss`` and ``VmHWM`` (both high-water marks)."""
    try:
        import resource  # POSIX only; the helpers are imported by tests on Windows
    except ImportError:
        peak = 0
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KiB on Linux
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                peak = max(peak, int(line.split()[1]) * 1024)
    except OSError:
        pass
    return peak


def machine_state() -> dict[str, Any]:
    import jax

    def read(p: str) -> str | None:
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None

    mem = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "Cached"):
            mem[k] = v.strip()
    gpu = None
    if shutil.which("nvidia-smi"):
        q = "name,memory.total,memory.used,clocks.sm,clocks.mem,temperature.gpu"
        try:
            gpu = subprocess.run(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu = f"nvidia-smi failed: {exc}"
    return {
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "memory": mem,
        "gpu": gpu,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------------- run


def _fit(x: Any, y: TailOffsetY, chunk: int) -> dict[str, Any]:
    from nlsq.streaming.exact_lm import exact_lm_fit

    t = time.perf_counter()
    with record_passes() as log:
        res = exact_lm_fit(model_jnp, x, y, PROBLEM.p0, chunk_size=chunk)
    wall = time.perf_counter() - t
    n = len(y)
    secs = [s for _, s in log]
    return {
        "x": np.asarray(res["x"]),
        "cost": float(res["cost"]),
        "pcov": np.asarray(res["pcov"]),
        "nfev": int(res["nfev"]),
        "termination": res["termination"],
        "success": bool(res["success"]),
        "wall_s": wall,
        "pass_n_seen": [int(c) for c, _ in log],
        "pass_s": secs,
        "median_pass_s": float(np.median(secs)),
        "points_per_s": n / float(np.median(secs)),
        "y_rows_read": y.rows_read,
    }


def noisy_gate(x: np.ndarray, pcov: np.ndarray) -> tuple[bool, str]:
    """Gate 2: finite positive standard errors and ``|x - theta_true| <= 5 se``.

    An unavailable covariance (all-inf ``pcov``) gives ``z = 0`` and must not pass.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.diag(pcov))
        z = np.abs(x - THETA_TRUE) / se
    ok = bool(np.all(np.isfinite(se)) and np.all(se > 0) and np.all(z <= 5))
    return ok, f"|z| = {np.round(z, 3).tolist()}, se = {se.tolist()}"


def _count_ok(fit: dict[str, Any], n: int) -> bool:
    return (
        len(fit["pass_n_seen"]) == fit["nfev"]
        and all(c == n for c in fit["pass_n_seen"])
        and fit["y_rows_read"] == n * fit["nfev"]
    )


def run(a: argparse.Namespace) -> int:
    import h5py  # type: ignore[import-untyped,import-not-found]
    import jax

    jax.config.update("jax_enable_x64", True)
    n, chunk = a.n, a.chunk_rows
    if n % chunk and n % chunk < SENTINEL_ROWS:
        raise SystemExit(
            f"last chunk has {n % chunk} rows, fewer than the {SENTINEL_ROWS} sentinel "
            "rows; pick another --chunk-rows"
        )
    d = Path(a.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    # no local paths or host name in the record: basenames only, the --out path dropped
    args = {k: v for k, v in vars(a).items() if k != "out"}
    args["data_dir"] = Path(a.data_dir).name
    report: dict[str, Any] = {"args": args, "machine": machine_state()}
    paths = {
        "noiseless": d / f"exp_decay_n{n}_c{chunk}_s{a.seed}_noise0.h5",
        "noisy": d / f"exp_decay_n{n}_c{chunk}_s{a.seed}_noise{a.noise:g}.h5",
    }
    for key, noise in (("noiseless", 0.0), ("noisy", a.noise)):
        t = time.perf_counter()
        st = write_dataset(paths[key], n, a.seed, noise, chunk)
        report[f"generate_{key}"] = {
            "path": paths[key].name,
            "status": st,
            "s": time.perf_counter() - t,
        }
        print(f"[{key}] {paths[key]}: {st}", flush=True)

    gates: list[tuple[str, bool, str]] = []
    t_all = time.perf_counter()
    with h5py.File(paths["noiseless"], "r") as f0, h5py.File(paths["noisy"], "r") as f1:
        x0, y0, x1, y1 = f0["x"], f0["y"], f1["x"], f1["y"]

        clean = _fit(x0, TailOffsetY(y0), chunk)
        print(f"[noiseless] {clean['termination']} x={clean['x']}", flush=True)
        dev = np.abs(clean["x"] - THETA_TRUE)
        g1 = bool(np.all(dev <= 1e-6 * np.abs(THETA_TRUE)) and clean["cost"] < 1e-6 * n)
        gates.append(
            (
                "1 noiseless recovery",
                g1 and clean["success"],
                f"max rel err {rel_size(dev, THETA_TRUE):.2e}, cost {clean['cost']:.3e}",
            )
        )
        gates.append(
            ("3i count noiseless", _count_ok(clean, n), f"{clean['nfev']} passes")
        )

        t = time.perf_counter()
        H, n_h = hessian_at(THETA_TRUE, x0, y0, chunk)
        x_tail = np.asarray(x0[n - SENTINEL_ROWS : n], dtype=np.float64)
        delta, pred = calibrate_sentinel(THETA_TRUE, x_tail, H)
        report["sentinel_calibration"] = {
            "delta": delta,
            "predicted": pred.tolist(),
            "predicted_rel": rel_size(pred, THETA_TRUE),
            "hessian_pass_s": time.perf_counter() - t,
            "hessian_n_seen": n_h,
        }
        sent = _fit(x0, TailOffsetY(y0, delta), chunk)
        moved = sent["x"] - clean["x"]
        ratio = rel_size(moved, clean["x"]) / rel_size(pred, THETA_TRUE)
        proj = float(moved @ pred / (pred @ pred))
        gates.append(
            (
                "3iii sentinel",
                sent["success"] and ratio >= 0.5,
                f"delta {delta:.4g}, actual/predicted {ratio:.4f}, projection {proj:.4f}",
            )
        )
        gates.append(
            ("3i count sentinel", _count_ok(sent, n), f"{sent['nfev']} passes")
        )

        noisy = _fit(x1, TailOffsetY(y1), chunk)
        print(f"[noisy] {noisy['termination']} x={noisy['x']}", flush=True)
        ok2, detail2 = noisy_gate(noisy["x"], noisy["pcov"])
        gates.append(("2 noisy within 5 sigma", noisy["success"] and ok2, detail2))
        gates.append(("3i count noisy", _count_ok(noisy, n), f"{noisy['nfev']} passes"))

        t = time.perf_counter()
        sc = streaming_cost((x1, y1), model_np, noisy["x"], chunk)
        sc0 = streaming_cost((x0, y0), model_np, clean["x"], chunk)
        rc = abs(sc - noisy["cost"]) / abs(noisy["cost"])
        report["streaming_cost"] = {
            "noisy": sc,
            "noisy_rel_diff": rc,
            "noiseless_info": sc0,
            "noiseless_solver_cost": clean["cost"],
            "s": time.perf_counter() - t,
        }
        gates.append(("3ii streaming cost (noisy)", rc <= 1e-9, f"rel diff {rc:.2e}"))

    peak = peak_rss_bytes()
    limit = BYTES_PER_POINT * n + 4 * 1024**3
    gates.append(
        (
            "4 peak RSS",
            peak < limit,
            f"{peak / 1024**3:.2f} GiB < {limit / 1024**3:.2f} GiB",
        )
    )
    for key, fit in (("noiseless", clean), ("sentinel", sent), ("noisy", noisy)):
        report[f"fit_{key}"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()
        }
    report["total_s"] = time.perf_counter() - t_all
    report["peak_rss_bytes"] = peak
    report["gates"] = [{"gate": g, "pass": ok, "detail": s} for g, ok, s in gates]

    print(f"\n{'gate':32} {'result':6} detail")
    for g, ok, s in gates:
        print(f"{g:32} {'PASS' if ok else 'FAIL':6} {s}")
    print("\n5 reporting (not gated):")
    for key, fit in (("noiseless", clean), ("sentinel", sent), ("noisy", noisy)):
        print(
            f"  {key:9} wall {fit['wall_s']:.1f}s passes {fit['nfev']} median pass "
            f"{fit['median_pass_s']:.2f}s ({fit['points_per_s']:.3g} points/s)"
        )
    print(f"  total {report['total_s']:.1f}s (generation excluded)")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {a.out}")
    return 0 if all(ok for _, ok, _ in gates) else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=1_000_000_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--chunk-rows", type=int, default=1 << 22)
    g = sub.choices["generate"]
    g.add_argument("path")
    g.add_argument("--noise", type=float, default=0.0)
    r = sub.choices["run"]
    r.add_argument("--data-dir", required=True)
    r.add_argument("--noise", type=float, default=0.05)
    r.add_argument("--out", default=None, help="JSON report path")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        print(write_dataset(a.path, a.n, a.seed, a.noise, a.chunk_rows))
        return 0
    if a.noise <= 0 or not math.isfinite(a.noise):
        raise SystemExit("--noise for the noisy copy must be positive")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
