"""Additional sparse Jacobian edge-case coverage."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.core.sparse_jacobian import SparseJacobianComputer, SparseOptimizer


@pytest.mark.numerical
def test_detect_sparsity_pattern_empty_sample() -> None:
    """Empty sample should return zero sparsity without error."""

    def model(xdata: np.ndarray, a: float) -> np.ndarray:
        return a * xdata

    computer = SparseJacobianComputer()
    pattern, sparsity = computer.detect_sparsity_pattern(
        model, x0=np.array([1.0]), xdata_sample=np.array([]), n_samples=0
    )

    assert pattern.size == 0
    assert sparsity == 0.0


@pytest.mark.numerical
def test_should_use_sparse_small_problem_false() -> None:
    """Small problems should not trigger sparse path."""
    optimizer = SparseOptimizer(auto_detect=True)
    assert optimizer.should_use_sparse(n_data=10, n_params=10) is False


@pytest.mark.numerical
def test_should_use_sparse_force_for_huge_problem() -> None:
    """Huge problems should suggest sparse even if auto-detect is disabled."""
    optimizer = SparseOptimizer(auto_detect=False)
    assert optimizer.should_use_sparse(n_data=200_000, n_params=1_000) is True
