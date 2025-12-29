"""Fast tests for robust decomposition fallback helpers."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.stability
@pytest.mark.unit
def test_ensure_positive_definite_shifts() -> None:
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    matrix = jnp.array([[1.0, 2.0], [2.0, -3.0]])
    pd = rd._ensure_positive_definite(matrix, factor=1e-6)
    eigs = np.linalg.eigvalsh(np.array(pd))
    assert np.min(eigs) >= 0.0


@pytest.mark.stability
@pytest.mark.unit
def test_solve_least_squares_fallback_to_qr(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    monkeypatch.setattr(
        rd, "svd", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(rd, "qr", lambda A: (jnp.eye(2), jnp.eye(2)))

    A = jnp.eye(2)
    b = jnp.array([1.0, 2.0])
    x = rd.solve_least_squares(A, b)
    assert np.allclose(np.array(x), np.array(b))


@pytest.mark.stability
@pytest.mark.unit
def test_cholesky_via_eigen() -> None:
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    matrix = jnp.array([[2.0, 0.0], [0.0, 1.0]])
    L = rd._cholesky_via_eigen(matrix, lower=True)
    assert L.shape == (2, 2)
