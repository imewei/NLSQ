"""Fast branch coverage tests for trf helpers without heavy JAX workloads."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.unit
def test_calculate_cost_and_isfinite() -> None:
    trf_module = importlib.import_module("nlsq.core.trf")
    funcs = trf_module.TrustRegionJITFunctions()

    rho = jnp.array([2.0, 4.0])
    mask = jnp.array([True, False])
    cost = funcs.calculate_cost(rho, mask)
    assert float(cost) == pytest.approx(1.0)

    assert bool(funcs.check_isfinite(jnp.array([1.0, 2.0]))) is True
    assert bool(funcs.check_isfinite(jnp.array([1.0, np.nan]))) is False


@pytest.mark.unit
def test_trf_dispatches_to_unbounded_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that trf() dispatches to trf_no_bounds for unbounded problems.

    Both timeit=False and timeit=True now use trf_no_bounds (with different
    profiler settings), so we only need to verify that trf_no_bounds is called.
    """
    trf_module = importlib.import_module("nlsq.core.trf")
    optimizer = trf_module.TrustRegionReflective()

    # Track calls to trf_no_bounds
    calls = {"count": 0, "profiler_provided": []}

    def mock_trf_no_bounds(*_a, **_k):
        calls["count"] += 1
        calls["profiler_provided"].append(_k.get("profiler") is not None)
        return {"mode": "no_bounds"}

    monkeypatch.setattr(optimizer, "trf_no_bounds", mock_trf_no_bounds)

    lb = np.array([-np.inf, -np.inf])
    ub = np.array([np.inf, np.inf])
    x0 = np.array([0.0, 0.0])
    f0 = jnp.array([0.0, 0.0])
    J0 = jnp.eye(2)

    # Test with timeit=False (no explicit profiler)
    result = optimizer.trf(
        fun=lambda *_a, **_k: jnp.array([0.0, 0.0]),
        xdata=jnp.array([0.0, 1.0]),
        ydata=jnp.array([0.0, 1.0]),
        jac=lambda *_a, **_k: jnp.eye(2),
        data_mask=jnp.array([True, True]),
        transform=jnp.array([1.0, 1.0]),
        x0=x0,
        f0=f0,
        J0=J0,
        lb=lb,
        ub=ub,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=10,
        f_scale=1.0,
        x_scale=np.array([1.0, 1.0]),
        loss_function=None,
        tr_options={},
        verbose=0,
        timeit=False,
        solver="exact",
        callback=None,
    )
    assert result["mode"] == "no_bounds"
    assert calls["count"] == 1
    assert calls["profiler_provided"][0] is False

    # Test with timeit=True (TRFProfiler provided)
    result = optimizer.trf(
        fun=lambda *_a, **_k: jnp.array([0.0, 0.0]),
        xdata=jnp.array([0.0, 1.0]),
        ydata=jnp.array([0.0, 1.0]),
        jac=lambda *_a, **_k: jnp.eye(2),
        data_mask=jnp.array([True, True]),
        transform=jnp.array([1.0, 1.0]),
        x0=x0,
        f0=f0,
        J0=J0,
        lb=lb,
        ub=ub,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=10,
        f_scale=1.0,
        x_scale=np.array([1.0, 1.0]),
        loss_function=None,
        tr_options={},
        verbose=0,
        timeit=True,
        solver="exact",
        callback=None,
    )
    assert result["mode"] == "no_bounds"
    assert calls["count"] == 2
    assert calls["profiler_provided"][1] is True
