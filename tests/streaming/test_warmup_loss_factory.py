"""The warmup loss is built by one factory shared by WarmupPhase and the hybrid
optimizer (it used to exist twice and had drifted). Checked against numpy."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.phases.warmup import build_warmup_loss


def _model(x, a, b, c, d):
    return a * x[:, 1] + b + 0.0 * (c + d)


def _data():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=float)
    x = np.stack([labels, np.linspace(0, 1, 8)], axis=1)
    return jnp.asarray(x), jnp.asarray(rng.normal(size=8))


P = jnp.array([1.5, 0.5, 2.0, 4.0])
SCALES, OFFSETS = np.array([2.0, 1.0, 3.0, 0.5]), np.array([0.0, 1.0, 0.0, 2.0])


def _identity(p):
    return p


def _physical(p):
    return p * SCALES + OFFSETS


def _numpy_loss(x, y, weights=None, penalty=0.0):
    x, y = np.asarray(x), np.asarray(y)
    r = y - (float(P[0]) * x[:, 1] + float(P[1]))
    if weights is None:
        return np.mean(r**2) + penalty
    w = np.asarray(weights)[x[:, 0].astype(int)]
    return np.sum(w * r**2) / np.sum(w) + penalty


def test_unweighted_mse():
    x, y = _data()
    loss = build_warmup_loss(
        _model,
        denormalize=_identity,
        var_reg_enabled=False,
        var_lambda=0.0,
        var_indices=None,
    )
    assert float(loss(P, x, y)) == pytest.approx(_numpy_loss(x, y))


def test_weighted_mse_uses_column_zero_groups():
    x, y = _data()
    w = jnp.array([1.0, 3.0, 0.5])
    loss = build_warmup_loss(
        _model,
        denormalize=_identity,
        var_reg_enabled=False,
        var_lambda=0.0,
        var_indices=None,
        weights=w,
    )
    assert float(loss(P, x, y)) == pytest.approx(_numpy_loss(x, y, w))


def test_variance_penalty_is_over_physical_parameters():
    x, y = _data()
    groups = [(0, 2), (2, 4)]
    lam = 0.7
    phys = np.asarray(_physical(P))
    expected_penalty = lam * (np.var(phys[0:2]) + np.var(phys[2:4]))
    loss = build_warmup_loss(
        _model,
        denormalize=_physical,
        var_reg_enabled=True,
        var_lambda=lam,
        var_indices=groups,
    )
    assert float(loss(P, x, y)) == pytest.approx(
        _numpy_loss(x, y, penalty=expected_penalty)
    )
    assert float(loss.penalty(P)) == pytest.approx(expected_penalty)


def test_unequal_trailing_group_does_not_borrow_the_previous_group():
    # dynamic_slice would clamp the last window onto the previous group
    groups = [(0, 3), (3, 4)]
    p = jnp.array([1.0, 2.0, 3.0, 10.0])
    loss = build_warmup_loss(
        lambda x, *q: 0 * x[:, 1],
        denormalize=_identity,
        var_reg_enabled=True,
        var_lambda=1.0,
        var_indices=groups,
    )
    assert float(loss.penalty(p)) == pytest.approx(np.var([1.0, 2.0, 3.0]))


def test_regularization_flag_with_no_groups_adds_nothing():
    x, y = _data()
    loss = build_warmup_loss(
        _model,
        denormalize=_identity,
        var_reg_enabled=True,
        var_lambda=5.0,
        var_indices=[],
    )
    assert float(loss(P, x, y)) == pytest.approx(_numpy_loss(x, y))
    assert float(loss.penalty(P)) == 0.0


def test_chunk_parts_reassemble_the_loss():
    x, y = _data()
    w = jnp.array([1.0, 3.0, 0.5])
    loss = build_warmup_loss(
        _model,
        denormalize=_physical,
        var_reg_enabled=True,
        var_lambda=0.3,
        var_indices=[(0, 2), (2, 4)],
        weights=w,
    )
    n1, d1 = loss.base_sums(P, x[:5], y[:5])
    n2, d2 = loss.base_sums(P, x[5:], y[5:])
    assert float((n1 + n2) / (d1 + d2) + loss.penalty(P)) == pytest.approx(
        float(loss(P, x, y))
    )


def test_weighting_with_1d_xdata_raises_a_clear_error():
    loss = build_warmup_loss(
        lambda x, *q: x,
        denormalize=_identity,
        var_reg_enabled=False,
        var_lambda=0.0,
        var_indices=None,
        weights=jnp.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match=r"shaped \(N, k\)"):
        loss(P, jnp.arange(4.0), jnp.arange(4.0))
