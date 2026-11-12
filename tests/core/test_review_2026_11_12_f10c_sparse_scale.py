"""Sparsity detection / extraction must be invariant to the Jacobian's overall scale."""

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.core.sparse_jacobian import (
    SparseJacobianComputer,
    detect_jacobian_sparsity,
    detect_sparsity_at_p0,
)


def _scaled(scale):
    def f(x, a, b):
        return scale * a * jnp.exp(-b * x)

    return f


def _block(scale):
    def f(x, a, b):
        return scale * jnp.where(x < 0.5, a, b)

    return f


X = np.linspace(0.1, 4, 100)  # x=0 gives an exact zero in d/db
P = np.array([1.0, 1.0])


def test_small_scale_dense_jacobian_not_reported_sparse():
    f = _scaled(1e-3)
    _, sp = SparseJacobianComputer().detect_sparsity_pattern(f, P, X)
    assert sp == 0.0
    assert detect_jacobian_sparsity(f, P, X)[0] == 0.0
    ratio, is_sparse = detect_sparsity_at_p0(f, P, X, n_residuals=100)
    assert ratio == 0.0 and not is_sparse


def test_compute_sparse_jacobian_not_all_zero_for_small_scale():
    f = _scaled(1e-3)
    jac = jax.jacfwd(lambda p, x: f(x, *p))

    def jac_func(p, xc, yc, mc, _):
        return np.asarray(jac(p, xc))

    J = SparseJacobianComputer().compute_sparse_jacobian(
        jac_func, P, np.concatenate([X, X]), np.zeros(200)
    )
    dense = np.asarray(jac(P, np.concatenate([X, X])))
    assert J.nnz > 0
    # entries below 1% of the column max may be dropped, nothing larger
    assert np.abs(J.toarray() - dense).max() <= 0.01 * np.abs(dense).max(axis=0).max()


def test_scale_invariance_of_true_sparsity():
    ref = detect_jacobian_sparsity(_block(1.0), P, X)[0]
    assert ref == 0.5
    for s in (1e-6, 1e-3, 1e3, 1e6):
        assert detect_jacobian_sparsity(_block(s), P, X)[0] == ref


def test_zero_column_stays_zero():
    def f(x, a, b):
        return 1e-4 * a * x + 0.0 * b

    _, sp = SparseJacobianComputer().detect_sparsity_pattern(f, P, X)
    assert sp == 0.5


def test_compute_sparse_jacobian_global_column_scale_across_chunks():
    # Column peaks in the 2nd chunk; first-chunk values below 1% of the global
    # max must be dropped consistently (global, not per-chunk, reference).
    n = 40

    def jac_func(p, xc, yc, mc, _):
        v = np.asarray(xc, dtype=float)
        return np.stack([v, np.ones_like(v)], axis=1)

    x = np.concatenate([np.full(20, 1e-3), np.full(20, 1.0)])
    J = SparseJacobianComputer().compute_sparse_jacobian(
        jac_func, P, x, np.zeros(n), chunk_size=20
    )
    D = J.toarray()
    assert np.all(D[:20, 0] == 0) and np.all(D[20:, 0] == 1.0)
    assert np.all(D[:, 1] == 1.0)
