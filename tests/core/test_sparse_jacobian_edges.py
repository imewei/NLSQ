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


def _two_coord_jac(params, xdata, ydata, mask, _unused):
    """Dense Jacobian (len, 2) for y = a*x0 + b*x1 (xdata list or (k, n) array)."""
    cols = [np.asarray(c) for c in xdata]
    return np.stack(cols, axis=1)


@pytest.mark.numerical
def test_compute_sparse_jacobian_2d_ndarray_matches_list_form() -> None:
    """A (k, N) ndarray with N > chunk_size must be sliced on the points axis."""
    n = 2500
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.0, size=(2, n))
    X[0, ::3] = 0.0  # introduce structural zeros
    y = np.zeros(n)
    computer = SparseJacobianComputer(0.0)
    p = np.array([1.0, 1.0])
    J_list = computer.compute_sparse_jacobian(
        _two_coord_jac, p, [X[0], X[1]], y, chunk_size=1000
    )
    J_arr = computer.compute_sparse_jacobian(_two_coord_jac, p, X, y, chunk_size=1000)
    assert J_arr.nnz == J_list.nnz
    np.testing.assert_array_equal(J_arr.toarray(), J_list.toarray())


@pytest.mark.numerical
def test_optimize_with_sparsity_samples_points_axis(monkeypatch) -> None:
    """optimize_with_sparsity must index the points axis of a (k, N) ndarray."""
    n = 5000
    X = np.random.default_rng(0).uniform(size=(2, n))
    seen = {}
    optimizer = SparseOptimizer()
    monkeypatch.setattr(optimizer, "should_use_sparse", lambda *a, **k: True)

    def spy(func, x0, xdata_sample, n_samples):
        seen["shape"] = np.asarray(xdata_sample).shape
        raise RuntimeError("stop")

    monkeypatch.setattr(optimizer.sparse_computer, "detect_sparsity_pattern", spy)
    with pytest.raises(RuntimeError, match="stop"):
        optimizer.optimize_with_sparsity(
            lambda x, a, b: a * x[0] + b * x[1], np.ones(2), X, np.zeros(n)
        )
    assert seen["shape"] == (2, 1000)
