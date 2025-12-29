"""Targeted tests for TRF select_step branching (fast, deterministic)."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest


def _make_optimizer(monkeypatch: pytest.MonkeyPatch, trf_module):
    monkeypatch.setattr(
        trf_module.TrustRegionJITFunctions, "__init__", lambda self: None
    )

    class DummyCjit:
        def evaluate_quadratic(self, *_a, **_k):
            return 5.0

        def build_quadratic_1d(self, *_a, **_k):
            if _k.get("s0") is not None:
                return 1.0, 0.0, 0.0
            return 1.0, 0.0

    monkeypatch.setattr(trf_module, "CommonJIT", lambda: DummyCjit())
    return trf_module.TrustRegionReflective()


@pytest.mark.unit
def test_select_step_in_bounds_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    trf_module = importlib.import_module("nlsq.core.trf")
    optimizer = _make_optimizer(monkeypatch, trf_module)

    monkeypatch.setattr(trf_module, "in_bounds", lambda *_a, **_k: True)

    x = np.array([0.0, 0.0])
    J_h = jnp.eye(2)
    diag_h = jnp.zeros(2)
    g_h = jnp.array([1.0, -1.0])
    p = np.array([1.0, -1.0])
    p_h = jnp.array([1.0, -1.0])
    d = np.array([1.0, 1.0])
    Delta = 1.0
    lb = np.array([-np.inf, -np.inf])
    ub = np.array([np.inf, np.inf])
    theta = 0.8

    step, step_h, predicted = optimizer.select_step(
        x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta
    )

    assert np.allclose(step, p)
    assert np.allclose(np.asarray(step_h), np.asarray(p_h))
    assert predicted == -5.0


@pytest.mark.unit
def test_select_step_reflection_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    trf_module = importlib.import_module("nlsq.core.trf")
    optimizer = _make_optimizer(monkeypatch, trf_module)

    monkeypatch.setattr(trf_module, "in_bounds", lambda *_a, **_k: False)

    call_count = {"step_size": 0}

    def _step_size_to_bound(*_a, **_k):
        call_count["step_size"] += 1
        if call_count["step_size"] == 1:
            return 0.5, np.array([True, False])
        if call_count["step_size"] == 2:
            return 0.8, np.array([False, False])
        return 0.2, np.array([False, False])

    monkeypatch.setattr(trf_module, "step_size_to_bound", _step_size_to_bound)
    monkeypatch.setattr(
        trf_module, "intersect_trust_region", lambda *_a, **_k: (0.0, 0.6)
    )

    def _minimize_quadratic_1d(_a, _b, _l, _u, c=None):
        if c is not None:
            return 0.4, 1.0
        return 0.1, 2.0

    monkeypatch.setattr(trf_module, "minimize_quadratic_1d", _minimize_quadratic_1d)

    x = np.array([0.0, 0.0])
    J_h = jnp.eye(2)
    diag_h = jnp.zeros(2)
    g_h = jnp.array([1.0, -1.0])
    p = np.array([2.0, -1.0])
    p_h = jnp.array([2.0, -1.0])
    d = np.array([1.0, 1.0])
    Delta = 1.0
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    theta = 0.8

    step, step_h, predicted = optimizer.select_step(
        x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta
    )

    assert step.shape == p.shape
    assert step_h.shape == p_h.shape
    assert predicted == -1.0
