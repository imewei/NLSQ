"""Fast tests for sparse Jacobian fallback paths."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.core.sparse_jacobian import SparseJacobianComputer


@pytest.mark.numerical
def test_sparse_jacobian_requires_func_for_fallback() -> None:
    """Fallback without jac_func should require func."""
    computer = SparseJacobianComputer()
    x = np.array([1.0])
    xdata = np.array([0.0, 1.0])
    ydata = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="func parameter required"):
        computer.compute_sparse_jacobian(
            jac_func=None,
            x=x,
            xdata=xdata,
            ydata=ydata,
            func=None,
        )


@pytest.mark.numerical
def test_sparse_jacobian_finite_diff_fallback() -> None:
    """Finite-diff fallback should approximate derivative for simple model."""

    def model(xdata: np.ndarray, a: float) -> np.ndarray:
        return a * xdata

    computer = SparseJacobianComputer(sparsity_threshold=0.0)
    x = np.array([2.0])
    xdata = np.array([1.0, 2.0, 3.0])
    ydata = model(xdata, x[0])

    J_sparse = computer.compute_sparse_jacobian(
        jac_func=None,
        x=x,
        xdata=xdata,
        ydata=ydata,
        func=model,
        chunk_size=2,
    )

    dense = J_sparse.toarray()
    assert dense.shape == (len(xdata), len(x))
    assert np.allclose(dense[:, 0], xdata, rtol=1e-5, atol=1e-5)
