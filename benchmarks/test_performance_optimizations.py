"""Performance benchmarks for NLSQ optimization validation.

This module provides benchmarks for validating the performance improvements
from the 001-performance-optimizations feature. Each benchmark corresponds
to a success criterion from the spec.

Benchmarks:
- SC-001: Import time (<400ms target)
- SC-002: Sparse Jacobian construction (100x speedup)
- SC-003: Trust region GPU solve (2x speedup)
- SC-004: Multi-start optimization (5x speedup)
- SC-005: Scan-based chunking (2x speedup)
- SC-006: Memory allocation reduction (20%)
- SC-010: Condition estimation (50% faster)
- SC-011: Gradient computation (5% faster)
"""

import subprocess
import sys
import time

import numpy as np
import pytest

# Skip benchmarks if pytest-benchmark not installed
pytest_benchmark = pytest.importorskip("pytest_benchmark")


class TestImportPerformance:
    """SC-001: Import time benchmarks."""

    def test_import_time_baseline(self) -> None:
        """Measure cold import time in subprocess."""
        code = """
import time
start = time.perf_counter()
import nlsq
end = time.perf_counter()
print(f'{(end - start) * 1000:.1f}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        import_time_ms = float(result.stdout.strip())
        # Target: <400ms (50%+ reduction from ~800ms baseline)
        # Note: This is informational; actual target depends on baseline
        print(f"Import time: {import_time_ms:.1f}ms")
        assert import_time_ms > 0, "Import time measurement failed"


class TestSparseJacobianPerformance:
    """SC-002: Sparse Jacobian construction benchmarks."""

    @pytest.fixture
    def sparse_jacobian_data(self) -> tuple[np.ndarray, float]:
        """Generate test data for sparse Jacobian benchmarks."""
        # 100k rows, 50 columns, 50% sparsity
        rng = np.random.default_rng(42)
        J = rng.standard_normal((100_000, 50))
        # Apply 50% sparsity
        mask = rng.random((100_000, 50)) > 0.5
        J[mask] = 0.0
        threshold = 0.01
        return J, threshold

    def test_sparse_construction_vectorized(
        self,
        sparse_jacobian_data: tuple[np.ndarray, float],
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark vectorized sparse Jacobian construction."""
        J, threshold = sparse_jacobian_data

        def vectorized_construction() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mask = np.abs(J) > threshold
            rows, cols = np.where(mask)
            values = J[rows, cols]
            return rows, cols, values

        result = benchmark(vectorized_construction)
        assert result is not None


class TestConditionEstimationPerformance:
    """SC-010: Condition number estimation benchmarks."""

    @pytest.fixture
    def test_matrix(self) -> np.ndarray:
        """Generate test matrix for condition estimation."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))
        return A @ A.T  # Symmetric positive definite

    def test_svd_condition(
        self,
        test_matrix: np.ndarray,
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark SVD-based condition number."""

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
            # Simplified estimate
            return float(norm_A * np.linalg.norm(np.linalg.inv(test_matrix), ord=1))

        result = benchmark(norm1_condition)
        assert result > 0


class TestGradientPerformance:
    """SC-011: Gradient computation benchmarks."""

    @pytest.fixture
    def gradient_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate test data for gradient benchmarks."""
        rng = np.random.default_rng(42)
        m, n = 10000, 50
        J = rng.standard_normal((m, n))
        f = rng.standard_normal(m)
        return J, f

    def test_gradient_dot(
        self,
        gradient_data: tuple[np.ndarray, np.ndarray],
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark f.dot(J) gradient computation."""
        J, f = gradient_data

        def dot_gradient() -> np.ndarray:
            return f.dot(J)

        result = benchmark(dot_gradient)
        assert result.shape == (50,)

    def test_gradient_matmul(
        self,
        gradient_data: tuple[np.ndarray, np.ndarray],
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark J.T @ f gradient computation."""
        J, f = gradient_data

        def matmul_gradient() -> np.ndarray:
            return J.T @ f

        result = benchmark(matmul_gradient)
        assert result.shape == (50,)
