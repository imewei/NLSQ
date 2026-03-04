"""Performance Optimization Swarm — Profiling Benchmark

Instruments the NLSQ critical path to quantify bottlenecks:
- Residual evaluation throughput
- Jacobian computation (AD) timing
- SVD decomposition timing
- Trust region subproblem solve timing
- Python overhead in inner loop (update_tr_radius, check_termination)
- Host↔Device transfer overhead
- Memory peak during full solve
"""

import cProfile
import gc
import pstats
import statistics
import time
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np

from nlsq import CurveFit


def exponential(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def gaussian(x, amp, mu, sigma):
    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def multi_peak(x, a1, mu1, s1, a2, mu2, s2, offset):
    """7-parameter model for more iterations."""
    return (
        a1 * jnp.exp(-((x - mu1) ** 2) / (2 * s1**2))
        + a2 * jnp.exp(-((x - mu2) ** 2) / (2 * s2**2))
        + offset
    )


def benchmark_fn(fn, args, n_warmup=3, n_runs=20, label=""):
    """Benchmark with warmup and stats."""
    for _ in range(n_warmup):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    p95 = sorted(times)[int(0.95 * len(times))]
    print(
        f"  [{label}] median={med * 1000:.3f}ms | stdev={std * 1000:.3f}ms | p95={p95 * 1000:.3f}ms"
    )
    return {"median": med, "stdev": std, "p95": p95, "times": times}


def profile_end_to_end():
    """Profile full curve_fit with cProfile to find top hot functions."""
    print("=" * 80)
    print("PHASE 1: End-to-End cProfile (Top 30 Hot Functions)")
    print("=" * 80)

    sizes = [
        ("1K pts / 3 params", 1_000, exponential, [2.0, 0.5, 0.3]),
        ("10K pts / 3 params", 10_000, exponential, [2.0, 0.5, 0.3]),
        ("100K pts / 3 params", 100_000, exponential, [2.0, 0.5, 0.3]),
        (
            "10K pts / 7 params",
            10_000,
            multi_peak,
            [5.0, -2.0, 1.0, 3.0, 2.0, 1.5, 0.5],
        ),
    ]

    cf = CurveFit()

    for name, n, model, p0_true in sizes:
        print(f"\n--- {name} ---")
        np.random.seed(42)
        x = np.linspace(-10, 10, n)
        y_true = model(x, *p0_true)
        y = np.asarray(y_true) + 0.1 * np.random.randn(n)
        p0 = [p * 1.2 for p in p0_true]  # Offset initial guess

        # Warmup (JIT compile)
        _ = cf.curve_fit(model, x, y, p0=p0)

        # Profile
        profiler = cProfile.Profile()
        profiler.enable()
        t0 = time.perf_counter()
        result = cf.curve_fit(model, x, y, p0=p0)
        elapsed = time.perf_counter() - t0
        profiler.disable()

        success = result.get("success", "N/A") if hasattr(result, "get") else "N/A"
        print(f"  Total: {elapsed * 1000:.2f}ms | Success: {success}")

        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        print(s.getvalue())


def profile_component_timing():
    """Profile individual components of the solver pipeline."""
    print("\n" + "=" * 80)
    print("PHASE 2: Component-Level Timing")
    print("=" * 80)

    from nlsq.common_jax import CommonJIT, solve_lsq_trust_region_jax
    from nlsq.common_scipy import check_termination, update_tr_radius
    from nlsq.core.trf_jit import TrustRegionJITFunctions
    from nlsq.stability.svd_fallback import compute_svd_with_fallback

    cjit = CommonJIT()
    trjit = TrustRegionJITFunctions()

    for n_data, n_params in [(1_000, 3), (10_000, 5), (100_000, 3), (1_000_000, 3)]:
        print(f"\n--- {n_data:,} data points × {n_params} params ---")

        # Create representative arrays
        J = jnp.array(np.random.randn(n_data, n_params).astype(np.float64))
        f = jnp.array(np.random.randn(n_data).astype(np.float64))
        d = jnp.ones(n_params)
        x = jnp.array(np.random.randn(n_params).astype(np.float64))

        # 1. Gradient computation: J^T f
        benchmark_fn(trjit.compute_grad, (J, f), label="J^T*f (gradient)")

        # 2. SVD of J_h (J*d)
        if n_data <= 100_000:
            benchmark_fn(trjit.svd_no_bounds, (J, d, f), label="SVD(J_h)")
        else:
            print(f"  [SVD(J_h)] SKIPPED (n={n_data:,} too large for dense SVD)")

        # 3. Default loss function: 0.5 * ||f||^2
        benchmark_fn(trjit.default_loss_func, (f,), label="loss(f)")

        # 4. check_isfinite
        benchmark_fn(trjit.check_isfinite, (f,), label="isfinite(f)")

        # 5. Trust region subproblem solve (needs SVD output)
        if n_data <= 100_000:
            J_h = J * d
            U, s, V = compute_svd_with_fallback(J_h, full_matrices=False)
            uf = U.T.dot(f)
            benchmark_fn(
                solve_lsq_trust_region_jax,
                (n_params, n_data, uf, s, V, 1.0),
                label="TR subproblem",
            )

        # 6. Python overhead: update_tr_radius
        def _update_tr():
            return update_tr_radius(1.0, 0.5, 0.6, 0.1, True)

        benchmark_fn(_update_tr, (), label="update_tr_radius (Python)")

        # 7. Python overhead: check_termination
        def _check_term():
            return check_termination(0.001, 1.0, 0.01, 1.0, 0.8, 1e-8, 1e-8)

        benchmark_fn(_check_term, (), label="check_termination (Python)")

        # 8. Norm computations
        benchmark_fn(
            lambda f=f: jnp.linalg.norm(f).block_until_ready(), (), label="jnp.norm(f)"
        )
        benchmark_fn(
            lambda x=x: jnp.linalg.norm(x, ord=jnp.inf).block_until_ready(),
            (),
            label="jnp.norm(x,inf)",
        )

        # Memory estimate
        jac_bytes = n_data * n_params * 8
        res_bytes = n_data * 8
        print(
            f"  [Memory] Jacobian={jac_bytes / 1e6:.1f}MB | Residuals={res_bytes / 1e6:.1f}MB | Total≈{(jac_bytes + res_bytes) * 2 / 1e6:.1f}MB"
        )

        gc.collect()


def profile_host_device_transfers():
    """Quantify host↔device transfer overhead in the hot path."""
    print("\n" + "=" * 80)
    print("PHASE 3: Host↔Device Transfer Analysis")
    print("=" * 80)

    for n_data in [1_000, 10_000, 100_000]:
        print(f"\n--- {n_data:,} points ---")
        x_np = np.random.randn(n_data)
        x_jax = jnp.array(x_np)

        # NumPy → JAX
        benchmark_fn(
            lambda x_np=x_np: jnp.array(x_np).block_until_ready(),
            (),
            label="np→jax transfer",
        )

        # JAX → NumPy
        benchmark_fn(lambda x_jax=x_jax: np.array(x_jax), (), label="jax→np transfer")

        # JAX scalar → Python float (common in convergence checks)
        val = jnp.sum(x_jax**2)
        benchmark_fn(lambda val=val: float(val), (), label="jax_scalar→float")

        # Python bool from JAX (convergence check pattern)
        benchmark_fn(
            lambda x_jax=x_jax: bool(jnp.all(jnp.isfinite(x_jax))),
            (),
            label="jax→bool sync",
        )


def profile_full_solve_scaling():
    """Benchmark full curve_fit across dataset sizes."""
    print("\n" + "=" * 80)
    print("PHASE 4: Full Solve Scaling")
    print("=" * 80)

    cf = CurveFit()
    sizes = [100, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    print(
        f"\n{'N':>12s} | {'Median (ms)':>12s} | {'P95 (ms)':>12s} | {'Iters':>6s} | {'nfev':>6s}"
    )
    print("-" * 65)

    for n in sizes:
        np.random.seed(42)
        x = np.linspace(0, 10, n)
        y_true = 2.0 * np.exp(-0.5 * x) + 0.3
        y = y_true + 0.05 * np.random.randn(n)
        p0 = [2.5, 0.6, 0.4]

        # Warmup
        _ = cf.curve_fit(exponential, x, y, p0=p0)

        times = []
        last_result = None
        for _ in range(10):
            t0 = time.perf_counter()
            last_result = cf.curve_fit(exponential, x, y, p0=p0)
            times.append(time.perf_counter() - t0)

        med = statistics.median(times) * 1000
        p95 = sorted(times)[int(0.95 * len(times))] * 1000
        nit = last_result.get("nit", "?") if hasattr(last_result, "get") else "?"
        nfev = last_result.get("nfev", "?") if hasattr(last_result, "get") else "?"
        print(f"{n:>12,d} | {med:>12.2f} | {p95:>12.2f} | {nit!s:>6s} | {nfev!s:>6s}")

    gc.collect()


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    profile_end_to_end()
    profile_component_timing()
    profile_host_device_transfers()
    profile_full_solve_scaling()

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
