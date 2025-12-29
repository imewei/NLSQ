"""Fast tests for TRF loss function paths."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.trf import TrustRegionReflective


@pytest.mark.numerical
def test_trf_loss_function_cost_only_path() -> None:
    """Loss function should be used for cost_only and scaling branches."""
    trf = TrustRegionReflective()
    calls = {"cost_only": 0, "full": 0}

    def fun(_x_new, _xdata, _ydata, _mask, _transform):
        return jnp.array([0.0])

    def jac(_x_new, _xdata, _ydata, _mask, _transform):
        return jnp.array([[1.0]])

    def loss_function(f, f_scale, data_mask=None, cost_only=False):
        if cost_only:
            calls["cost_only"] += 1
            return jnp.sum(f**2)
        calls["full"] += 1
        rho0 = f**2
        rho1 = jnp.ones_like(f)
        rho2 = jnp.zeros_like(f)
        return jnp.stack([rho0, rho1, rho2])

    x = np.array([0.0])
    xdata = np.array([0.0])
    ydata = np.array([0.0])
    data_mask = jnp.array([True])
    f = jnp.array([0.1])
    J = jnp.array([[1.0]])

    result = trf._evaluate_step_acceptance(
        fun=fun,
        jac=jac,
        x=x,
        f=f,
        J=J,
        J_h=J,
        g_h_jnp=jnp.array([0.0]),
        cost=1.0,
        d=np.array([1.0]),
        d_jnp=jnp.array([1.0]),
        Delta=1.0,
        alpha=1.0,
        step_h=jnp.array([0.1]),
        s=None,
        V=None,
        uf=None,
        xdata=xdata,
        ydata=ydata,
        data_mask=data_mask,
        transform=None,
        loss_function=loss_function,
        f_scale=1.0,
        scale_inv=np.array([1.0]),
        jac_scale=False,
        solver="cg",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=2,
        nfev=0,
    )

    assert calls["cost_only"] == 1
    assert calls["full"] == 1
    assert bool(result["accepted"]) is True
    assert "f_true_new" in result
