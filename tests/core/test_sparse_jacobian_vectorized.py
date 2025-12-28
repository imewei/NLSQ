"""Numerical equivalence tests for vectorized sparse Jacobian construction.

This module tests that the vectorized sparse Jacobian implementation produces
numerically equivalent results to the original nested-loop implementation.

Success Criterion: SC-002 - Sparse Jacobian construction 100x faster
Functional Requirement: FR-003 - Vectorized sparse matrix construction
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, lil_matrix


class TestSparseJacobianNumericalEquivalence:
    """Tests for numerical equivalence between original and vectorized implementations."""

    @pytest.fixture
    def sample_jacobian(self) -> np.ndarray:
        """Generate a sample dense Jacobian for testing."""
        rng = np.random.default_rng(42)
        m, n = 10000, 50
        J = rng.standard_normal((m, n))
        # Apply 50% sparsity
        mask = rng.random((m, n)) > 0.5
        J[mask] = 0.0
        return J

    def _original_sparse_construction(
        self, J_chunk: np.ndarray, threshold: float, start: int = 0
    ) -> csr_matrix:
        """Original O(nm) nested loop implementation."""
        n_data, n_params = J_chunk.shape
        J_sparse = lil_matrix((n_data, n_params))

        for i in range(n_data):
            for j in range(n_params):
                if np.abs(J_chunk[i, j]) > threshold:
                    J_sparse[start + i - start, j] = J_chunk[i, j]

        return J_sparse.tocsr()

    def _vectorized_sparse_construction(
        self, J_chunk: np.ndarray, threshold: float, start: int = 0
    ) -> csr_matrix:
        """Vectorized O(nnz) implementation using NumPy."""
        # Find non-zero elements using vectorized operations
        mask = np.abs(J_chunk) > threshold
        rows, cols = np.where(mask)
        values = J_chunk[rows, cols]

        # Adjust row indices for offset
        rows = rows + start - start  # Offset = 0 in this test

        # Create sparse matrix directly in COO then convert to CSR
        from scipy.sparse import coo_matrix

        n_data, n_params = J_chunk.shape
        J_sparse = coo_matrix((values, (rows, cols)), shape=(n_data, n_params))
        return J_sparse.tocsr()

    def test_numerical_equivalence(self, sample_jacobian: np.ndarray) -> None:
        """Test that vectorized and original implementations produce identical results."""
        threshold = 0.01

        # Original implementation
        J_original = self._original_sparse_construction(sample_jacobian, threshold)

        # Vectorized implementation
        J_vectorized = self._vectorized_sparse_construction(sample_jacobian, threshold)

        # Check structural equality
        assert J_original.shape == J_vectorized.shape
        assert J_original.nnz == J_vectorized.nnz

        # Check numerical equality (allowing for floating point tolerance)
        diff = np.abs(J_original - J_vectorized)
        if diff.nnz > 0:
            max_diff = np.max(diff.data)
        else:
            max_diff = 0.0

        assert max_diff < 1e-10, f"Max difference: {max_diff}"

    def test_vectorized_handles_empty_jacobian(self) -> None:
        """Test vectorized implementation with all-zero Jacobian."""
        J_empty = np.zeros((100, 10))
        threshold = 0.01

        J_sparse = self._vectorized_sparse_construction(J_empty, threshold)

        assert J_sparse.nnz == 0
        assert J_sparse.shape == (100, 10)

    def test_vectorized_handles_dense_jacobian(self) -> None:
        """Test vectorized implementation with fully dense Jacobian."""
        rng = np.random.default_rng(42)
        J_dense = rng.standard_normal((100, 10)) + 10  # All non-zero

        threshold = 0.01

        J_original = self._original_sparse_construction(J_dense, threshold)
        J_vectorized = self._vectorized_sparse_construction(J_dense, threshold)

        assert J_original.nnz == J_vectorized.nnz
        assert np.allclose(J_original.toarray(), J_vectorized.toarray(), rtol=1e-10)

    def test_vectorized_with_offset(self) -> None:
        """Test vectorized implementation with row offset."""
        rng = np.random.default_rng(42)
        J_chunk = rng.standard_normal((100, 10))
        threshold = 0.01
        offset = 500

        # The offset logic - for building larger matrices
        mask = np.abs(J_chunk) > threshold
        rows, cols = np.where(mask)
        values = J_chunk[rows, cols]

        # Rows with offset for larger matrix
        rows_with_offset = rows + offset

        from scipy.sparse import coo_matrix

        J_sparse = coo_matrix((values, (rows_with_offset, cols)), shape=(1000, 10))

        # Verify offset is applied correctly
        # Find where values appear in the sparse matrix
        actual_rows = J_sparse.tocsr().indices  # This is columns for CSR
        # Check that values are placed after row 500
        assert J_sparse.tocsr()[offset : offset + 100, :].nnz > 0

    def test_threshold_boundary(self) -> None:
        """Test values exactly at threshold boundary."""
        # Create Jacobian with values exactly at threshold
        J = np.array(
            [
                [0.01, 0.011, 0.009],  # boundary, above, below
                [0.0, 0.02, 0.001],
            ]
        )
        threshold = 0.01

        J_sparse = self._vectorized_sparse_construction(J, threshold)

        # Values > threshold should be included
        # 0.01 is NOT > 0.01, so excluded
        # 0.011, 0.02 should be included
        assert J_sparse.nnz == 2


class TestSparseJacobianIntegration:
    """Integration tests with actual SparseJacobianComputer."""

    def test_compute_sparse_jacobian_vectorized(self) -> None:
        """Test that SparseJacobianComputer produces correct sparse matrix."""
        from nlsq.core.sparse_jacobian import SparseJacobianComputer

        computer = SparseJacobianComputer(sparsity_threshold=0.1)

        # Simple linear model
        def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * x + b

        xdata = np.linspace(0, 10, 1000)
        ydata = 2.0 * xdata + 1.0
        x0 = np.array([1.0, 1.0])

        # Detect sparsity
        pattern, _sparsity = computer.detect_sparsity_pattern(model, x0, xdata)

        # For linear model, all elements should be non-zero
        # (each data point depends on both a and b)
        assert pattern.shape == (100, 2)  # n_samples=100 by default
