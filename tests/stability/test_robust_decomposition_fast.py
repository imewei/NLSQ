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
def test_cholesky_all_tiers_nan_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every fallback_chain tier (and the eigendecomposition last resort)
    returning a NaN factor must raise RuntimeError, not silently return
    NaN. JAX's Cholesky commonly returns NaN rather than raising on a
    non-positive-definite input, so `result is not None` alone was not
    sufficient to detect failure."""
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    nan_2x2 = jnp.full((2, 2), jnp.nan)
    monkeypatch.setattr(rd, "fallback_chain", [("nan_tier", lambda *a, **k: nan_2x2)])
    monkeypatch.setattr(rd, "_cholesky_via_eigen", lambda *a, **k: nan_2x2)

    with pytest.raises(RuntimeError, match=r"non-finite|failed"):
        rd.cholesky(jnp.eye(2))


@pytest.mark.stability
@pytest.mark.unit
def test_solve_least_squares_catches_cholesky_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """solve_least_squares()'s normal-equations fallback must catch the
    RuntimeError cholesky() now raises when every tier fails, and continue
    to its own 'ultimate fallback' (direct regularized solve) instead of
    propagating the exception uncaught."""
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    monkeypatch.setattr(
        rd, "svd", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        rd, "qr", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        rd,
        "cholesky",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("all tiers failed")),
    )

    A = jnp.eye(2)
    b = jnp.array([1.0, 2.0])
    x = rd.solve_least_squares(A, b)
    assert bool(jnp.all(jnp.isfinite(x)))


@pytest.mark.stability
@pytest.mark.unit
def test_cholesky_via_eigen() -> None:
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    matrix = jnp.array([[2.0, 0.0], [0.0, 1.0]])
    L = rd._cholesky_via_eigen(matrix, lower=True)
    assert L.shape == (2, 2)


@pytest.mark.stability
@pytest.mark.unit
def test_numpy_decomp_cholesky_respects_lower() -> None:
    """_numpy_decomp's cholesky branch used to ignore the `lower` arg and
    always return the lower-triangular factor (np.linalg.cholesky's
    default), unlike the JAX/SciPy backends which honor it correctly."""
    module = importlib.import_module("nlsq.stability.robust_decomposition")
    rd = module.RobustDecomposition()

    matrix = jnp.array([[4.0, 2.0], [2.0, 3.0]])

    L = rd._numpy_decomp(matrix, "cholesky", True)
    U = rd._numpy_decomp(matrix, "cholesky", False)

    L_np, U_np = np.array(L), np.array(U)
    assert np.allclose(np.tril(L_np), L_np), "lower=True must be lower-triangular"
    assert np.allclose(np.triu(U_np), U_np), "lower=False must be upper-triangular"
    assert np.allclose(U_np, L_np.T)
    np.testing.assert_allclose(L_np @ L_np.T, np.array(matrix), atol=1e-10)
