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
def test_compute_svd_with_fallback_cuda_ffi_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fallback triggers for JAX >=0.8 CUDA FFI errors (cusolver_gesvdj_ffi)."""
    module = importlib.import_module("nlsq.stability.svd_fallback")

    calls = {"count": 0}

    def _fake_svd(matrix, full_matrices=False):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError(
                "NOT_FOUND: No FFI handler registered for "
                "cusolver_gesvdj_ffi on a platform CUDA (canonical cuda)"
            )
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
class TestIsGpuError:
    """Unit tests for the _is_gpu_error helper."""

    def test_legacy_cusolver(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("cuSolver internal error") is True

    def test_ffi_cusolver_gesvdj(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        msg = (
            "NOT_FOUND: No FFI handler registered for "
            "cusolver_gesvdj_ffi on a platform CUDA (canonical cuda)"
        )
        assert _is_gpu_error(msg) is True

    def test_cublas_error(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("cuBLAS operation failed") is True

    def test_xla_internal_status(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("INTERNAL: solver failed") is True

    def test_generic_internal_error_not_matched(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("internal error in parsing") is False

    def test_lowercase_internal_not_matched(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("internal: something went wrong") is False

    def test_unrelated_error(self) -> None:
        from nlsq.stability.svd_fallback import _is_gpu_error

        assert _is_gpu_error("ValueError: shapes don't match") is False
