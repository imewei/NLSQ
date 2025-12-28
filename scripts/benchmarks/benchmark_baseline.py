#!/usr/bin/env python3
"""Baseline benchmark script for performance optimization validation.

This script measures baseline performance metrics before any optimizations
are applied. Results are saved to specs/002-performance-optimizations/baseline.json.

Usage:
    python scripts/benchmarks/benchmark_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import nlsq


def benchmark_large_dataset_fit() -> dict[str, Any]:
    """Benchmark large dataset fitting (1M points).

    Measures:
    - Total fit time
    - Average chunk time
    - Final chunk time (should be similar with static shapes)
    """
    # Create dataset where final chunk is smaller (not divisible by typical chunk size)
    n_points = 1_000_127
    x = np.linspace(0, 10, n_points)
    rng = np.random.default_rng(42)
    y = 2.5 * np.exp(-0.3 * x) + rng.normal(0, 0.1, len(x))

    def model(x: jax.Array, a: float, b: float) -> jax.Array:
        return a * jnp.exp(-b * x)

    # Warmup with small dataset to trigger JIT compilation
    x_small = x[:1000]
    y_small = y[:1000]
    nlsq.curve_fit(model, x_small, y_small, p0=[1.0, 1.0])

    # Benchmark full dataset
    start = time.perf_counter()
    popt, pcov = nlsq.curve_fit(model, x, y, p0=[1.0, 1.0])
    elapsed = time.perf_counter() - start

    return {
        "n_points": n_points,
        "elapsed_seconds": elapsed,
        "parameters": popt.tolist() if hasattr(popt, "tolist") else list(popt),
        "parameter_covariance_diag": np.diag(pcov).tolist(),
    }


def benchmark_svd_iterations() -> dict[str, Any]:
    """Benchmark SVD-intensive fit with many iterations.

    Uses an ill-conditioned problem that triggers many step rejections,
    making SVD caching beneficial.
    """
    x = np.linspace(0, 1, 1000)
    rng = np.random.default_rng(42)
    y = np.exp(-10 * x) + 0.001 * np.sin(100 * x) + rng.normal(0, 0.0001, len(x))

    def model(x: jax.Array, a: float, b: float, c: float) -> jax.Array:
        return jnp.exp(-a * x) + c * jnp.sin(b * x)

    # Warmup
    nlsq.curve_fit(model, x[:100], y[:100], p0=[1.0, 1.0, 0.001])

    # Benchmark
    start = time.perf_counter()
    popt, _pcov = nlsq.curve_fit(model, x, y, p0=[1.0, 1.0, 0.001])
    elapsed = time.perf_counter() - start

    return {
        "n_points": len(x),
        "elapsed_seconds": elapsed,
        "parameters": popt.tolist() if hasattr(popt, "tolist") else list(popt),
    }


def benchmark_memory_manager_overhead() -> dict[str, Any]:
    """Benchmark memory manager overhead with repeated fits.

    Measures the variance in fit times to detect psutil overhead.
    """
    x = np.linspace(0, 10, 100_000)
    rng = np.random.default_rng(42)
    y = 2.5 * np.exp(-0.3 * x) + rng.normal(0, 0.1, len(x))

    def model(x: jax.Array, a: float, b: float) -> jax.Array:
        return a * jnp.exp(-b * x)

    # Warmup
    nlsq.curve_fit(model, x[:1000], y[:1000], p0=[1.0, 1.0])

    # Benchmark multiple runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        nlsq.curve_fit(model, x, y, p0=[1.0, 1.0])
        times.append(time.perf_counter() - start)

    return {
        "n_points": len(x),
        "n_runs": len(times),
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
    }


def benchmark_condition_estimation() -> dict[str, Any]:
    """Benchmark condition number estimation.

    Compares full SVD vs singular values only for condition estimation.
    """
    n = 500
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))
    A = A @ A.T + 0.1 * jnp.eye(n)  # Ensure positive definite

    # Full SVD
    start = time.perf_counter()
    for _ in range(10):
        _U, s, _Vt = jnp.linalg.svd(A, full_matrices=False)
        cond = s[0] / s[-1]
    full_svd_time = (time.perf_counter() - start) / 10

    # Singular values only
    start = time.perf_counter()
    for _ in range(10):
        s = jnp.linalg.svdvals(A)
        cond = s[0] / s[-1]
    svdvals_time = (time.perf_counter() - start) / 10

    return {
        "matrix_size": n,
        "full_svd_seconds": full_svd_time,
        "svdvals_seconds": svdvals_time,
        "speedup": full_svd_time / svdvals_time if svdvals_time > 0 else float("inf"),
    }


def benchmark_cholesky_vs_eigenvalue() -> dict[str, Any]:
    """Benchmark Cholesky vs eigenvalue decomposition.

    For positive definite matrices, Cholesky should be ~3x faster.
    """
    n = 500
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))
    A = A @ A.T + 0.1 * jnp.eye(n)  # Ensure positive definite
    b = jax.random.normal(jax.random.PRNGKey(1), (n,))

    # Cholesky solve
    start = time.perf_counter()
    for _ in range(10):
        L = jnp.linalg.cholesky(A)
        x_chol = jax.scipy.linalg.cho_solve((L, True), b)
    chol_time = (time.perf_counter() - start) / 10

    # Eigenvalue solve
    start = time.perf_counter()
    for _ in range(10):
        eigvals, eigvecs = jnp.linalg.eigh(A)
        x_eigh = eigvecs @ (eigvecs.T @ b / eigvals)
    eigh_time = (time.perf_counter() - start) / 10

    return {
        "matrix_size": n,
        "cholesky_seconds": chol_time,
        "eigenvalue_seconds": eigh_time,
        "speedup": eigh_time / chol_time if chol_time > 0 else float("inf"),
    }


def main() -> None:
    """Run all baseline benchmarks and save results."""
    print("Running baseline benchmarks...")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print()

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }

    print("1/5 Benchmarking large dataset fit...")
    results["large_dataset_fit"] = benchmark_large_dataset_fit()
    print(f"    Elapsed: {results['large_dataset_fit']['elapsed_seconds']:.3f}s")

    print("2/5 Benchmarking SVD iterations...")
    results["svd_iterations"] = benchmark_svd_iterations()
    print(f"    Elapsed: {results['svd_iterations']['elapsed_seconds']:.3f}s")

    print("3/5 Benchmarking memory manager overhead...")
    results["memory_manager"] = benchmark_memory_manager_overhead()
    print(
        f"    Mean: {results['memory_manager']['mean_seconds']:.4f}s, Std: {results['memory_manager']['std_seconds']:.4f}s"
    )

    print("4/5 Benchmarking condition estimation...")
    results["condition_estimation"] = benchmark_condition_estimation()
    print(
        f"    Full SVD: {results['condition_estimation']['full_svd_seconds']:.4f}s, svdvals: {results['condition_estimation']['svdvals_seconds']:.4f}s"
    )

    print("5/5 Benchmarking Cholesky vs eigenvalue...")
    results["cholesky_vs_eigenvalue"] = benchmark_cholesky_vs_eigenvalue()
    print(
        f"    Cholesky: {results['cholesky_vs_eigenvalue']['cholesky_seconds']:.4f}s, Eigenvalue: {results['cholesky_vs_eigenvalue']['eigenvalue_seconds']:.4f}s"
    )

    # Save results
    output_path = (
        Path(__file__).parent.parent
        / "specs"
        / "002-performance-optimizations"
        / "baseline.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")
    print()
    print("Summary:")
    print(
        f"  Large dataset (1M points): {results['large_dataset_fit']['elapsed_seconds']:.3f}s"
    )
    print(f"  SVD iterations: {results['svd_iterations']['elapsed_seconds']:.3f}s")
    print(f"  Memory manager mean: {results['memory_manager']['mean_seconds']:.4f}s")
    print(
        f"  Condition estimation speedup potential: {results['condition_estimation']['speedup']:.2f}x"
    )
    print(
        f"  Cholesky speedup potential: {results['cholesky_vs_eigenvalue']['speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
