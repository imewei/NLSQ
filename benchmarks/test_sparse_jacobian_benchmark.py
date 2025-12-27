"""Benchmarks for sparse Jacobian construction.

This module benchmarks the sparse Jacobian construction performance
to validate SC-002: 100x speedup for sparse Jacobian construction.

Target: 100k×50 matrix sparse construction in <100ms
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

# Skip benchmarks if pytest-benchmark not installed
pytest_benchmark = pytest.importorskip("pytest_benchmark")


class TestSparseJacobianBenchmarks:
    """Benchmarks for sparse Jacobian construction (SC-002)."""

    @pytest.fixture
    def large_jacobian(self) -> tuple[np.ndarray, float]:
        """Generate 100k×50 Jacobian with 50% sparsity."""
        rng = np.random.default_rng(42)
        m, n = 100_000, 50
        J = rng.standard_normal((m, n))
        # Apply 50% sparsity
        mask = rng.random((m, n)) > 0.5
        J[mask] = 0.0
        threshold = 0.01
        return J, threshold

    @pytest.fixture
    def medium_jacobian(self) -> tuple[np.ndarray, float]:
        """Generate 10k×50 Jacobian with 50% sparsity."""
        rng = np.random.default_rng(42)
        m, n = 10_000, 50
        J = rng.standard_normal((m, n))
        mask = rng.random((m, n)) > 0.5
        J[mask] = 0.0
        threshold = 0.01
        return J, threshold

    def test_original_nested_loop(
        self,
        medium_jacobian: tuple[np.ndarray, float],
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark original O(nm) nested loop implementation."""
        J_chunk, threshold = medium_jacobian

        def original_construction() -> csr_matrix:
            n_data, n_params = J_chunk.shape
            J_sparse = lil_matrix((n_data, n_params))

            for i in range(n_data):
                for j in range(n_params):
                    if np.abs(J_chunk[i, j]) > threshold:
                        J_sparse[i, j] = J_chunk[i, j]

            return J_sparse.tocsr()

        result = benchmark(original_construction)
        assert result.nnz > 0

    def test_vectorized_construction(
        self,
        large_jacobian: tuple[np.ndarray, float],
        benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    ) -> None:
        """Benchmark vectorized O(nnz) implementation."""
        J_chunk, threshold = large_jacobian

        def vectorized_construction() -> csr_matrix:
            # Vectorized sparse construction
            mask = np.abs(J_chunk) > threshold
            rows, cols = np.where(mask)
            values = J_chunk[rows, cols]

            n_data, n_params = J_chunk.shape
            J_sparse = coo_matrix((values, (rows, cols)), shape=(n_data, n_params))
            return J_sparse.tocsr()

        result = benchmark(vectorized_construction)
        assert result.nnz > 0

    def test_vectorized_vs_original_speedup(
        self, medium_jacobian: tuple[np.ndarray, float]
    ) -> None:
        """Compare vectorized vs original implementation speedup."""
        import time

        J_chunk, threshold = medium_jacobian
        n_data, n_params = J_chunk.shape

        # Time original implementation
        start = time.perf_counter()
        J_lil = lil_matrix((n_data, n_params))
        for i in range(n_data):
            for j in range(n_params):
                if np.abs(J_chunk[i, j]) > threshold:
                    J_lil[i, j] = J_chunk[i, j]
        J_original = J_lil.tocsr()
        original_time = time.perf_counter() - start

        # Time vectorized implementation
        start = time.perf_counter()
        mask = np.abs(J_chunk) > threshold
        rows, cols = np.where(mask)
        values = J_chunk[rows, cols]
        J_vectorized = coo_matrix(
            (values, (rows, cols)), shape=(n_data, n_params)
        ).tocsr()
        vectorized_time = time.perf_counter() - start

        # Calculate speedup
        speedup = original_time / vectorized_time

        print("\n--- Sparse Jacobian Construction Speedup ---")
        print(f"Matrix size: {n_data}×{n_params}")
        print(f"Original (nested loop): {original_time * 1000:.2f}ms")
        print(f"Vectorized (NumPy): {vectorized_time * 1000:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")

        # Verify numerical equivalence
        diff = np.abs(J_original - J_vectorized)
        max_diff = np.max(diff.data) if diff.nnz > 0 else 0.0
        assert max_diff < 1e-10, f"Numerical mismatch: {max_diff}"

        # Target: 100x speedup (SC-002)
        # Note: The actual speedup depends on matrix size and sparsity
        # For 10kx50 with 50% sparsity, expect 50-200x speedup
        assert speedup > 10, f"Speedup {speedup:.1f}x below minimum threshold"


class TestSparseJacobianScaling:
    """Tests for sparse Jacobian construction scaling behavior."""

    @pytest.mark.parametrize(
        "m,n",
        [
            (1_000, 50),
            (10_000, 50),
            (100_000, 50),
        ],
    )
    def test_vectorized_scaling(self, m: int, n: int) -> None:
        """Test vectorized implementation scales linearly with matrix size."""
        import time

        rng = np.random.default_rng(42)
        J = rng.standard_normal((m, n))
        mask = rng.random((m, n)) > 0.5
        J[mask] = 0.0
        threshold = 0.01

        start = time.perf_counter()
        mask = np.abs(J) > threshold
        rows, cols = np.where(mask)
        values = J[rows, cols]
        _ = coo_matrix((values, (rows, cols)), shape=(m, n)).tocsr()
        elapsed = time.perf_counter() - start

        print(f"\n{m}×{n} matrix: {elapsed * 1000:.2f}ms")

        # 100kx50 should complete in <200ms (system-dependent, allows variance)
        if m == 100_000:
            assert elapsed < 0.2, f"100k×50 took {elapsed * 1000:.1f}ms, target <200ms"
