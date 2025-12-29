"""Additional fast branch tests for trf helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.trf import TrustRegionReflective


def test_check_convergence_criteria_hits() -> None:
    trf = TrustRegionReflective()
    status = trf._check_convergence_criteria(jnp.array([1e-9]), gtol=1e-6)
    assert status == 1


def test_solve_trust_region_subproblem_sparse_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trf = TrustRegionReflective()

    def _svd_no_bounds(J: jnp.ndarray, d_jnp: jnp.ndarray, f: jnp.ndarray):
        J_h = J * d_jnp
        s = jnp.array([1.0])
        V = jnp.eye(J.shape[1])
        uf = jnp.array([0.0])
        return J_h, None, s, V, uf

    monkeypatch.setattr(trf, "svd_no_bounds", _svd_no_bounds)

    J = jnp.array([[1.0]])
    f = jnp.array([1.0])
    g = jnp.array([1.0])
    result = trf._solve_trust_region_subproblem(
        J=J,
        f=f,
        g=g,
        scale=np.array([1.0]),
        Delta=1.0,
        alpha=0.0,
        solver="sparse",
    )

    assert result["step_h"] is None
    assert result["J_h"].shape == (1, 1)
    assert result["s"] is not None


def test_evaluate_step_acceptance_handles_nonfinite() -> None:
    trf = TrustRegionReflective()

    def fun(_x, _xdata, _ydata, _mask, _transform):
        return jnp.array([jnp.nan])

    def jac(_x, _xdata, _ydata, _mask, _transform):
        return jnp.array([[1.0]])

    result = trf._evaluate_step_acceptance(
        fun=fun,
        jac=jac,
        x=np.array([0.0]),
        f=jnp.array([1.0]),
        J=jnp.array([[1.0]]),
        J_h=jnp.array([[1.0]]),
        g_h_jnp=jnp.array([1.0]),
        cost=0.5,
        d=np.array([1.0]),
        d_jnp=jnp.array([1.0]),
        Delta=1.0,
        alpha=0.0,
        step_h=jnp.array([0.1]),
        s=None,
        V=None,
        uf=None,
        xdata=np.array([1.0]),
        ydata=np.array([1.0]),
        data_mask=jnp.array([True]),
        transform=None,
        loss_function=None,
        f_scale=1.0,
        scale_inv=np.array([1.0]),
        jac_scale=False,
        solver="cg",
        ftol=0.0,
        xtol=0.0,
        max_nfev=1,
        nfev=0,
    )

    assert result["accepted"] is False
    assert result["nfev"] == 1
    assert result["step_norm"] == 0


def test_evaluate_step_acceptance_accepts_step() -> None:
    trf = TrustRegionReflective()

    def fun(_x, _xdata, _ydata, _mask, _transform):
        return jnp.array([0.0])

    def jac(_x, _xdata, _ydata, _mask, _transform):
        return jnp.array([[1.0]])

    result = trf._evaluate_step_acceptance(
        fun=fun,
        jac=jac,
        x=np.array([0.0]),
        f=jnp.array([1.0]),
        J=jnp.array([[1.0]]),
        J_h=jnp.array([[0.0]]),
        g_h_jnp=jnp.array([1.0]),
        cost=0.5,
        d=np.array([1.0]),
        d_jnp=jnp.array([1.0]),
        Delta=1.0,
        alpha=0.0,
        step_h=jnp.array([-1.0]),
        s=None,
        V=None,
        uf=None,
        xdata=np.array([1.0]),
        ydata=np.array([1.0]),
        data_mask=jnp.array([True]),
        transform=None,
        loss_function=None,
        f_scale=1.0,
        scale_inv=np.array([1.0]),
        jac_scale=False,
        solver="cg",
        ftol=1e-12,
        xtol=1e-12,
        max_nfev=5,
        nfev=0,
    )

    assert bool(result["accepted"]) is True
    assert result["njev"] == 1
    assert np.isclose(result["actual_reduction"], 0.5)
