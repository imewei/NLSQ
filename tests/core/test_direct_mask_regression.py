"""Direct residual least-squares calls must honor excluded observations."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import LeastSquares


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_direct_residual_mask_excludes_outlier(loss, analytic, bounded):
    result = LeastSquares().least_squares(
        lambda p: p[0] - jnp.array([2.0, 2.0, 100.0]),
        [0.0],
        jac=(lambda p: jnp.ones((3, 1))) if analytic else None,
        data_mask=np.array([True, True, False]),
        bounds=([-10.0], [10.0]) if bounded else (-np.inf, np.inf),
        loss=loss,
    )
    np.testing.assert_allclose(result.x, [2.0], atol=1e-6)
    np.testing.assert_allclose(result.fun, [0.0, 0.0, 0.0], atol=1e-6)
    assert result.cost < 1e-12


def test_direct_residual_mask_excludes_nonfinite_observation():
    result = LeastSquares().least_squares(
        lambda p: jnp.array([p[0] - 2.0, jnp.nan]),
        [0.0],
        data_mask=np.array([True, False]),
    )
    np.testing.assert_allclose(result.x, [2.0], atol=1e-6)
    assert np.isfinite(result.cost)


def test_direct_residual_without_mask():
    result = LeastSquares().least_squares(lambda p: p - 2.0, [0.0])
    np.testing.assert_allclose(result.x, [2.0], atol=1e-6)


@pytest.mark.parametrize("mask", [[True], [[True, False]]])
def test_direct_residual_rejects_misaligned_mask(mask):
    with pytest.raises(ValueError, match=r"[Mm]ask"):
        LeastSquares().least_squares(
            lambda p: jnp.array([p[0], p[0]]), [1.0], data_mask=mask
        )
