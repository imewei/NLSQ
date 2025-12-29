"""Fast tests for TRF step rejection paths."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.trf import TR_REDUCTION_FACTOR, TrustRegionReflective


@pytest.mark.numerical
def test_trf_step_rejection_on_nonfinite_residuals() -> None:
    """Non-finite residuals should reject the step and reduce trust radius."""
    trf = TrustRegionReflective()

    def fun(_x_new, _xdata, _ydata, _mask, _transform):
        return jnp.array([jnp.nan])

    def jac(_x_new, _xdata, _ydata, _mask, _transform):
        return jnp.array([[1.0]])

    x = np.array([1.0])
    xdata = np.array([0.0])
    ydata = np.array([0.0])
    data_mask = jnp.array([True])
    f = jnp.array([0.0])
    J = jnp.array([[1.0]])
    J_h = J
    g_h = jnp.array([0.0])
    d = np.array([1.0])
    d_jnp = jnp.array([1.0])
    step_h = jnp.array([0.1])
    expected_delta = TR_REDUCTION_FACTOR * float(jnp.linalg.norm(step_h))

    result = trf._evaluate_step_acceptance(
        fun=fun,
        jac=jac,
        x=x,
        f=f,
        J=J,
        J_h=J_h,
        g_h_jnp=g_h,
        cost=1.0,
        d=d,
        d_jnp=d_jnp,
        Delta=1.0,
        alpha=1.0,
        step_h=step_h,
        s=None,
        V=None,
        uf=None,
        xdata=xdata,
        ydata=ydata,
        data_mask=data_mask,
        transform=None,
        loss_function=None,
        f_scale=1.0,
        scale_inv=np.array([1.0]),
        jac_scale=False,
        solver="cg",
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=1,
        nfev=0,
    )

    assert result["accepted"] is False
    assert result["nfev"] == 1
    assert result["step_norm"] == 0
    assert result["Delta"] == pytest.approx(expected_delta)
