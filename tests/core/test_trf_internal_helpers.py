"""Fast unit tests for TRF helper branches without heavy JAX usage."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _make_optimizer() -> SimpleNamespace:
    trf_module = importlib.import_module("nlsq.core.trf")

    opt = SimpleNamespace()
    opt.logger = SimpleNamespace(
        debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None
    )
    opt.cJIT = SimpleNamespace(
        evaluate_quadratic=lambda *_a, **_k: -1.0,
        scale_for_robust_loss_function=lambda J, f, rho: (J, f),
        compute_jac_scale=lambda J, scale_inv=None: (
            jnp.ones(J.shape[1]),
            jnp.ones(J.shape[1]),
        ),
    )
    opt.default_loss_func = lambda f: 0.5 * jnp.dot(f, f)
    opt.compute_grad = lambda J, f: f.dot(J)
    opt.compute_grad_hat = lambda g, d: g * d
    opt.solve_tr_subproblem_cg = lambda *_a, **_k: jnp.array([0.1, -0.1])
    opt.check_isfinite = lambda *_a, **_k: True

    # Bind methods from class for direct calls
    cls = trf_module.TrustRegionReflective
    opt._check_convergence_criteria = cls._check_convergence_criteria.__get__(opt, cls)
    opt._solve_trust_region_subproblem = cls._solve_trust_region_subproblem.__get__(
        opt, cls
    )
    opt._evaluate_step_acceptance = cls._evaluate_step_acceptance.__get__(opt, cls)
    return opt


@pytest.mark.unit
def test_check_convergence_criteria_branches() -> None:
    opt = _make_optimizer()

    g_small = jnp.array([1e-8, 1e-9])
    assert opt._check_convergence_criteria(g_small, gtol=1e-6) == 1

    g_large = jnp.array([1e-2, 2e-2])
    assert opt._check_convergence_criteria(g_large, gtol=1e-6) is None


@pytest.mark.unit
def test_solve_trust_region_subproblem_branches() -> None:
    opt = _make_optimizer()
    J = jnp.eye(2)
    f = jnp.array([1.0, -1.0])
    g = jnp.array([1.0, 1.0])
    scale = np.array([1.0, 2.0])

    result = opt._solve_trust_region_subproblem(
        J, f, g, scale, Delta=1.0, alpha=1.0, solver="cg"
    )
    assert result["step_h"] is not None
    assert result["s"] is None

    def _svd_no_bounds(J_in, d_jnp, f_in):
        J_h = J_in * d_jnp
        U = jnp.eye(2)
        s = jnp.array([1.0, 0.5])
        V = jnp.eye(2)
        uf = U.T.dot(f_in)
        return J_h, U, s, V, uf

    opt.svd_no_bounds = _svd_no_bounds
    result = opt._solve_trust_region_subproblem(
        J, f, g, scale, Delta=1.0, alpha=1.0, solver="sparse"
    )
    assert result["step_h"] is None
    assert result["s"] is not None


@pytest.mark.unit
def test_evaluate_step_acceptance_accepts() -> None:
    opt = _make_optimizer()

    x = np.array([0.0, 0.0])
    f = jnp.array([1.0, -1.0])
    J = jnp.eye(2)
    J_h = jnp.eye(2)
    g_h = jnp.array([1.0, 1.0])
    d = np.array([1.0, 1.0])
    d_jnp = jnp.array([1.0, 1.0])

    def fun(x_new, *_args):
        return jnp.array([0.1, -0.1])

    def jac(x_new, *_args):
        return jnp.eye(2)

    result = opt._evaluate_step_acceptance(
        fun,
        jac,
        x,
        f,
        J,
        J_h,
        g_h,
        cost=1.0,
        d=d,
        d_jnp=d_jnp,
        Delta=1.0,
        alpha=1.0,
        step_h=jnp.array([0.1, -0.1]),
        s=None,
        V=None,
        uf=None,
        xdata=np.array([0.0, 1.0]),
        ydata=np.array([0.0, 1.0]),
        data_mask=jnp.array([True, True]),
        transform=None,
        loss_function=None,
        f_scale=1.0,
        scale_inv=np.array([1.0, 1.0]),
        jac_scale=False,
        solver="cg",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=5,
        nfev=0,
    )

    assert bool(result["accepted"]) is True
    assert result["njev"] == 1
    assert result["actual_reduction"] > 0


@pytest.mark.unit
def test_evaluate_step_acceptance_nonfinite_rejects() -> None:
    opt = _make_optimizer()
    opt.check_isfinite = lambda *_a, **_k: False

    x = np.array([0.0, 0.0])
    f = jnp.array([1.0, -1.0])
    J = jnp.eye(2)
    J_h = jnp.eye(2)
    g_h = jnp.array([1.0, 1.0])
    d = np.array([1.0, 1.0])
    d_jnp = jnp.array([1.0, 1.0])

    def fun(x_new, *_args):
        return jnp.array([1.0, 1.0])

    def jac(x_new, *_args):
        return jnp.eye(2)

    result = opt._evaluate_step_acceptance(
        fun,
        jac,
        x,
        f,
        J,
        J_h,
        g_h,
        cost=1.0,
        d=d,
        d_jnp=d_jnp,
        Delta=1.0,
        alpha=1.0,
        step_h=jnp.array([0.1, -0.1]),
        s=None,
        V=None,
        uf=None,
        xdata=np.array([0.0, 1.0]),
        ydata=np.array([0.0, 1.0]),
        data_mask=jnp.array([True, True]),
        transform=None,
        loss_function=None,
        f_scale=1.0,
        scale_inv=np.array([1.0, 1.0]),
        jac_scale=False,
        solver="cg",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=1,
        nfev=0,
    )

    assert result["accepted"] is False
    assert result["actual_reduction"] == 0
    assert result["Delta"] < 1.0


@pytest.mark.unit
def test_evaluate_step_acceptance_with_loss_and_jac_scale() -> None:
    opt = _make_optimizer()

    x = np.array([0.0, 0.0])
    f = jnp.array([1.0, -1.0])
    J = jnp.eye(2)
    J_h = jnp.eye(2)
    g_h = jnp.array([1.0, 1.0])
    d = np.array([1.0, 1.0])
    d_jnp = jnp.array([1.0, 1.0])

    def fun(x_new, *_args):
        return jnp.array([0.05, -0.05])

    def jac(x_new, *_args):
        return jnp.eye(2)

    def loss_fn(f_new, f_scale, *_args, cost_only=False):
        cost = 0.5 * jnp.dot(f_new, f_new)
        return cost if cost_only else f_new

    result = opt._evaluate_step_acceptance(
        fun,
        jac,
        x,
        f,
        J,
        J_h,
        g_h,
        cost=1.0,
        d=d,
        d_jnp=d_jnp,
        Delta=1.0,
        alpha=1.0,
        step_h=jnp.array([0.1, -0.1]),
        s=None,
        V=None,
        uf=None,
        xdata=np.array([0.0, 1.0]),
        ydata=np.array([0.0, 1.0]),
        data_mask=jnp.array([True, True]),
        transform=None,
        loss_function=loss_fn,
        f_scale=1.0,
        scale_inv=np.array([1.0, 1.0]),
        jac_scale=True,
        solver="cg",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=5,
        nfev=0,
    )

    assert bool(result["accepted"]) is True
    assert result["scale"] is not None


@pytest.mark.unit
def test_log_iteration_callback_converts_types(monkeypatch: pytest.MonkeyPatch) -> None:
    trf_module = importlib.import_module("nlsq.core.trf")
    captured: dict[str, tuple] = {}

    def _capture(*args):
        captured["args"] = args

    monkeypatch.setattr(trf_module, "print_iteration_nonlinear", _capture)

    trf_module.TrustRegionReflective._log_iteration_callback(
        jnp.array(1),
        jnp.array(2),
        jnp.array(3.5),
        jnp.array(0.1),
        jnp.array(0.2),
        jnp.array(0.3),
    )

    iteration, nfev, cost, actual_reduction, step_norm, g_norm = captured["args"]
    assert isinstance(iteration, int)
    assert isinstance(nfev, int)
    assert isinstance(cost, float)
    assert isinstance(actual_reduction, float)
    assert isinstance(step_norm, float)
    assert isinstance(g_norm, float)


@pytest.mark.unit
def test_null_profiler_noops() -> None:
    trf_module = importlib.import_module("nlsq.core.trf")
    profiler = trf_module.NullProfiler()
    assert profiler.time_operation("jit", 5) == 5
    assert profiler.get_timing_data()["ftimes"] == []
