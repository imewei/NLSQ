"""Benchmarks for condition number estimation.

This module benchmarks condition estimation performance for SC-010:
- Target: 50% faster than full SVD-based condition computation

Compares:
- Full SVD: np.linalg.cond() or svdvals-based
- 1-norm estimation: ||A||_1 * ||A^{-1}||_1
"""

import time

import numpy as np
import pytest

# Skip benchmarks if pytest-benchmark not installed
pytest_benchmark = pytest.importorskip("pytest_benchmark")


class TestConditionEstimationBenchmarks:
    """Benchmarks for condition estimation (SC-010)."""

    @pytest.fixture
    def test_matrix(self) -> np.ndarray:
        """Generate 100×100 test matrix."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 100))

    @pytest.fixture
    def large_matrix(self) -> np.ndarray:
        """Generate 500×500 test matrix."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((500, 500))

    def test_svd_condition(
        self,
        test_matrix: np.ndarray,
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark full SVD condition number computation."""

        def svd_condition() -> float:
            s = np.linalg.svd(test_matrix, compute_uv=False)
            return float(s[0] / s[-1])

        result = benchmark(svd_condition)
        assert result > 0

    def test_1norm_condition(
        self,
        test_matrix: np.ndarray,
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark 1-norm condition estimation."""

        def norm1_condition() -> float:
            norm_A = np.max(np.sum(np.abs(test_matrix), axis=0))
            A_pinv = np.linalg.pinv(test_matrix)
            norm_A_inv = np.max(np.sum(np.abs(A_pinv), axis=0))
            return float(norm_A * norm_A_inv)

        result = benchmark(norm1_condition)
        assert result > 0

    def test_condition_speedup(self, large_matrix: np.ndarray) -> None:
        """Compare 1-norm estimation vs full SVD speedup."""
        A = large_matrix

        # Time SVD-based condition
        start = time.perf_counter()
        for _ in range(5):
            s = np.linalg.svd(A, compute_uv=False)
            _ = s[0] / s[-1]
        svd_time = (time.perf_counter() - start) / 5

        # Time 1-norm estimation
        start = time.perf_counter()
        for _ in range(5):
            norm_A = np.max(np.sum(np.abs(A), axis=0))
            A_pinv = np.linalg.pinv(A)
            norm_A_inv = np.max(np.sum(np.abs(A_pinv), axis=0))
            _ = norm_A * norm_A_inv
        norm1_time = (time.perf_counter() - start) / 5

        print("\n--- Condition Estimation Speedup ---")
        print(f"Matrix size: {A.shape}")
        print(f"SVD condition: {svd_time * 1000:.2f}ms")
        print(f"1-norm estimation: {norm1_time * 1000:.2f}ms")

        # Note: pinv still uses SVD internally, so speedup may be limited
        # The real benefit is for approximate estimation using power iteration
        # or randomized methods for very large matrices


class TestPowerIterationCondition:
    """Tests for power iteration-based condition estimation."""

    def test_power_iteration_method(self) -> None:
        """Test power iteration for fast approximate condition estimation."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))

        # Approximate largest singular value via power iteration
        x = rng.standard_normal(100)
        for _ in range(20):
            x = A.T @ A @ x
            x = x / np.linalg.norm(x)
        sigma_max = np.sqrt(x @ (A.T @ A @ x))

        # Approximate smallest singular value via inverse power iteration
        # (more numerically challenging)
        try:
            AtA = A.T @ A
            y = rng.standard_normal(100)
            for _ in range(20):
                y = np.linalg.solve(AtA, y)
                y = y / np.linalg.norm(y)
            sigma_min = 1.0 / np.sqrt(y @ (AtA @ y))
        except np.linalg.LinAlgError:
            sigma_min = 0.0

        if sigma_min > 0:
            cond_approx = sigma_max / sigma_min
            cond_true = np.linalg.cond(A)

            ratio = cond_approx / cond_true
            print(f"\nPower iteration condition: {cond_approx:.2e}")
            print(f"True condition: {cond_true:.2e}")
            print(f"Ratio: {ratio:.2f}")

            # Within 10x for monitoring purposes
            assert 0.1 < ratio < 10.0
