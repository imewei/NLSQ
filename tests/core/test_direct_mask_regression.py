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
    if analytic:
        # Excluded observations must not enter J (and hence J^T J).
        np.testing.assert_array_equal(np.asarray(result.jac)[2], 0.0)


def test_direct_analytic_jacobian_mask_matches_deleted_row():
    # Two parameters; the excluded row [10, 1] would change J^T J if unmasked.
    x = np.array([0.0, 1.0, 10.0])
    y = np.array([1.0, 3.0, -50.0])

    def run(xs, ys, mask):
        return LeastSquares().least_squares(
            lambda p: p[0] * xs + p[1] - ys,
            [0.0, 0.0],
            jac=lambda p: jnp.stack([xs, jnp.ones_like(xs)], axis=1),
            data_mask=mask,
        )

    masked = run(x, y, np.array([True, True, False]))
    deleted = run(x[:2], y[:2], None)
    jac = np.asarray(masked.jac)
    np.testing.assert_array_equal(jac[2], 0.0)
    np.testing.assert_allclose(jac.T @ jac, np.asarray(deleted.jac).T @ deleted.jac)
    np.testing.assert_allclose(masked.x, deleted.x, atol=1e-6)


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


def test_direct_residual_rejects_misaligned_mask():
    with pytest.raises(ValueError, match="doesn't match residual shape"):
        LeastSquares().least_squares(
            lambda p: jnp.array([p[0], p[0]]), [1.0], data_mask=[True]
        )


# (2, 1) would broadcast in jnp.where instead of mismatching, so only the
# explicit ndim check rejects it.
@pytest.mark.parametrize("mask", [[[True, False]], [[True], [False]]])
def test_direct_residual_rejects_non_1d_mask(mask):
    with pytest.raises(ValueError, match="one-dimensional"):
        LeastSquares().least_squares(
            lambda p: jnp.array([p[0], p[0]]), [1.0], data_mask=mask
        )
