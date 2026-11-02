"""NormalEquations: the shared chunked J^T J / J^T r accumulator (moved from exact_lm)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.normal_equations import (
    _SMALL_P,
    NormalEquations,
    _gradient_and_gauss_newton_hessian,
    make_chunk_stats,
    run_pass,
)

RTOL = 1e-8
ATOL = 1e-10


def exp_model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 3.0, n)
    y = 2.0 * np.exp(-1.1 * x) + rng.normal(0.0, 0.02, n)
    return x, y


def test_accumulate_matches_run_pass():
    x, y = _data(30)
    theta = np.array([1.5, 0.8])
    chunk = 10  # 3 full chunks
    ne = NormalEquations(exp_model, chunk=chunk)
    got = ne.accumulate(theta, (x, y, None))
    ref = run_pass(make_chunk_stats(exp_model), theta, (x, y, None), chunk)
    assert got.n_seen == ref.n_seen == 30
    np.testing.assert_allclose(got.cost, ref.cost, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got.g, ref.g, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got.H, ref.H, rtol=RTOL, atol=ATOL)


def test_single_parameter_shapes():
    x, y = _data(12)
    ne = NormalEquations(lambda x, a: a * x, chunk=5)  # ragged tail
    st = ne.accumulate(np.array([0.7]), (x, y, None))
    assert st.H.shape == (1, 1)
    assert st.g.shape == (1,)
    F, g, H = ne.chunk_stats(
        jnp.asarray([0.7]), jnp.asarray(x[:5]), jnp.asarray(y[:5]), jnp.ones(5, bool)
    )
    assert g.shape == (1,) and H.shape == (1, 1) and F.shape == ()


def test_empty_source_raises():
    ne = NormalEquations(exp_model, chunk=4)
    with pytest.raises(ValueError, match="non-empty"):
        ne.accumulate(np.array([1.0, 1.0]), (np.empty(0), np.empty(0), None))


def test_chunk_size_one():
    x, y = _data(5)
    theta = np.array([1.5, 0.8])
    one = NormalEquations(exp_model, chunk=1).accumulate(theta, (x, y, None))
    whole = NormalEquations(exp_model, chunk=5).accumulate(theta, (x, y, None))
    assert one.n_seen == whole.n_seen == 5
    np.testing.assert_allclose(one.cost, whole.cost, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(one.g, whole.g, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(one.H, whole.H, rtol=RTOL, atol=ATOL)


@jax.custom_vjp
def _rev_exp(x, a, b):
    return a * jnp.exp(-b * x)


def _rev_fwd(x, a, b):
    e = jnp.exp(-b * x)
    return a * e, (x, a, b, e)


def _rev_bwd(res, ct):
    x, a, b, e = res
    return (
        ct * (-a * b * e),
        jnp.sum(ct * e),
        jnp.sum(ct * (-a * x * e)),
    )


_rev_exp.defvjp(_rev_fwd, _rev_bwd)


def _rev_jacobian(theta, x):
    e = jnp.exp(-theta[1] * x)
    return jnp.stack([e, -theta[0] * x * e], axis=1)


def test_custom_jacobian_hook_used():
    x, y = _data(23)  # ragged tail with chunk=8
    theta = np.array([1.5, 0.8])

    with pytest.raises(TypeError):  # jacfwd cannot differentiate a custom_vjp function
        NormalEquations(_rev_exp, chunk=8).accumulate(theta, (x, y, None))

    st = NormalEquations(_rev_exp, chunk=8, jacobian=_rev_jacobian).accumulate(
        theta, (x, y, None)
    )
    J = np.asarray(_rev_jacobian(jnp.asarray(theta), jnp.asarray(x)))
    r = y - theta[0] * np.exp(-theta[1] * x)
    np.testing.assert_allclose(st.H, J.T @ J, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(st.g, J.T @ r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(st.cost, 0.5 * np.sum(r * r), rtol=RTOL, atol=ATOL)


def test_custom_jacobian_weighted_matches_default():
    x, y = _data(23)
    s = np.linspace(0.5, 2.0, 23)
    theta = np.array([1.5, 0.8])
    ref = NormalEquations(exp_model, weighted=True, chunk=8).accumulate(
        theta, (x, y, s)
    )
    hook = NormalEquations(
        exp_model, weighted=True, chunk=8, jacobian=_rev_jacobian
    ).accumulate(theta, (x, y, s))
    np.testing.assert_allclose(hook.H, ref.H, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(hook.g, ref.g, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(hook.cost, ref.cost, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("p", [1, 2, 3, _SMALL_P - 1, _SMALL_P, 12])
def test_gradient_and_hessian_match_dense_for_every_width(p):
    """Both branches (fused sum for p < _SMALL_P, dot otherwise) give J^T r and J^T J."""
    rng = np.random.default_rng(p)
    J, r = rng.normal(size=(257, p)), rng.normal(size=257)
    g, H = jax.jit(_gradient_and_gauss_newton_hessian)(jnp.asarray(J), jnp.asarray(r))
    np.testing.assert_allclose(np.asarray(g), J.T @ r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(H), J.T @ J, rtol=RTOL, atol=ATOL)
