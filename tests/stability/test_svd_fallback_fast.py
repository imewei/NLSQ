"""Fast unit tests for SVD fallback paths."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.stability
@pytest.mark.unit
def test_compute_svd_with_fallback_cpu_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.stability.svd_fallback")

    calls = {"count": 0}

    def _fake_svd(matrix, full_matrices=False):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("cuSolver error")
        return jnp.eye(2), jnp.array([2.0, 1.0]), jnp.eye(2)

    monkeypatch.setattr(module, "jax_svd", _fake_svd)

    with pytest.warns(RuntimeWarning):
        U, s, V = module.compute_svd_with_fallback(jnp.eye(2), full_matrices=False)

    assert np.allclose(np.array(s), np.array([2.0, 1.0]))
    assert U.shape == (2, 2)
    assert V.shape == (2, 2)


@pytest.mark.stability
@pytest.mark.unit
def test_compute_svd_with_fallback_numpy_last_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("nlsq.stability.svd_fallback")

    def _raise(*_a, **_k):
        raise RuntimeError("cuSolver INTERNAL")

    monkeypatch.setattr(module, "jax_svd", _raise)

    numpy_called = {"hit": False}

    orig_svd = np.linalg.svd

    def _numpy_svd(matrix, full_matrices=False):
        numpy_called["hit"] = True
        return orig_svd(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(np.linalg, "svd", _numpy_svd)

    with pytest.warns(RuntimeWarning) as warning_record:
        U, s, V = module.compute_svd_with_fallback(jnp.eye(2), full_matrices=False)

    assert U.shape == (2, 2)
    assert s.shape == (2,)
    assert V.shape == (2, 2)
    assert numpy_called["hit"] is True
    assert len(warning_record) >= 1


@pytest.mark.stability
@pytest.mark.unit
def test_compute_svd_adaptive_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.stability.svd_fallback")

    monkeypatch.setattr(
        module,
        "compute_svd_with_fallback",
        lambda *_a, **_k: (jnp.eye(1), jnp.array([1.0]), jnp.eye(1)),
    )

    with pytest.warns(DeprecationWarning):
        module.compute_svd_adaptive(jnp.eye(1), use_randomized=True)
