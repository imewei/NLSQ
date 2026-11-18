"""Regression tests for the F2 fixes of the 2026-11-18 round-2 review (core TRF/JAX helpers)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.common_jax import minimize_quadratic_1d_jax, solve_lsq_trust_region_jax
from nlsq.common_scipy import (
    minimize_quadratic_1d,
    reflective_transformation,
    solve_lsq_trust_region,
)


@pytest.mark.parametrize("k", [1.0, 1e60, 1e-60])
def test_secular_solvers_scale_invariant(k):
    """The trust-region step is invariant to a common scaling of s and uf."""
    s = np.array([2.0, 1.0]) * k
    uf = np.array([3.0, 4.0]) * k
    V = np.eye(2)
    ref = solve_lsq_trust_region(
        2, 2, np.array([3.0, 4.0]), np.array([2.0, 1.0]), V, 1.0
    )[0]
    p_np = solve_lsq_trust_region(2, 2, uf, s, V, 1.0)[0]
    p_jx = np.asarray(
        solve_lsq_trust_region_jax(
            2, 2, jnp.asarray(uf), jnp.asarray(s), jnp.asarray(V), 1.0
        )[0]
    )
    np.testing.assert_allclose(p_np, ref, rtol=1e-6)
    np.testing.assert_allclose(p_jx, ref, rtol=1e-6)


def test_minimize_quadratic_1d_large_constant_does_not_change_minimizer():
    t, y = minimize_quadratic_1d(1.0, -2.0, 0.0, 2.0, c=1e20)
    assert t == 1.0 and y == pytest.approx(1e20)
    tj, yj = minimize_quadratic_1d_jax(1.0, -2.0, 0.0, 2.0, 1e20)
    assert float(tj) == 1.0 and float(yj) == pytest.approx(1e20)


def test_reflective_transformation_keeps_feasible_coordinate_exact():
    y = np.array([1.0, 2.0])
    x, g = reflective_transformation(y, np.array([-1e20, 0.0]), np.array([1e20, 1.0]))
    np.testing.assert_array_equal(x, [1.0, 0.0])
    np.testing.assert_array_equal(g, [1.0, 1.0])


def test_cg_float32_system_with_float64_scalars():
    from nlsq.core.trf_jit import _conjugate_gradient_solve, _solve_tr_subproblem_cg

    J = jnp.eye(2, dtype=jnp.float32)
    f = jnp.array([-1.0, -2.0], dtype=jnp.float32)
    d = jnp.ones(2, dtype=jnp.float32)
    p, _, _ = _conjugate_gradient_solve(
        J, f, d, jnp.array(0.1, dtype=jnp.float64), 1e-6
    )
    np.testing.assert_allclose(np.asarray(p), np.array([1.0, 2.0]) / 1.1, rtol=1e-5)
    q = _solve_tr_subproblem_cg(J, f, d, jnp.array(0.5, dtype=jnp.float64))
    assert np.linalg.norm(np.asarray(q)) <= 0.5 * (1 + 1e-5)


def test_exponential_estimators_keep_representable_origin_amplitude():
    from nlsq.core.functions import exponential_decay, exponential_growth

    x = np.linspace(750, 760, 20)
    y = 2 * np.exp(0.1 * (x - 750)) + 10
    a, b, c = exponential_growth.estimate_p0(x, y)
    pred = a * np.exp(b * x) + c
    assert np.max(np.abs(pred - y)) < 10 * np.ptp(y)  # clipped guess was off by ~1e57
    xd = np.linspace(-1000, -990, 20)
    yd = 2 * np.exp(-0.1 * (xd + 1000)) + 10
    a, b, c = exponential_decay.estimate_p0(xd, yd)
    assert np.isfinite([a, b, c]).all()
    assert np.max(np.abs(a * np.exp(-b * xd) + c - yd)) < 10 * np.ptp(yd)


def test_exponential_estimator_overflow_is_explicit():
    from nlsq.core.functions import exponential_growth

    x = np.linspace(-1e4, -1e4 + 10, 20)
    with pytest.raises(ValueError, match="not representable"):
        exponential_growth.estimate_p0(x, 2 * np.exp(0.1 * (x + 1e4)) + 10)


def test_peak_and_step_estimators_do_not_wrap_narrow_integers():
    from nlsq.core.functions import gaussian, sigmoid

    x = np.arange(7.0)
    p_u8 = sigmoid.estimate_p0(
        x, np.array([180, 181, 182, 200, 218, 219, 220], np.uint8)
    )
    p_f = sigmoid.estimate_p0(x, np.array([180, 181, 182, 200, 218, 219, 220], float))
    assert p_u8 == p_f
    y8 = np.array([0, -4, -40, -128, -40, -4, 0], np.int8)
    assert gaussian.estimate_p0(x, y8) == gaussian.estimate_p0(x, y8.astype(float))


def test_profiler_uses_monotonic_clock(monkeypatch):
    import time

    from nlsq.core.profiler import TRFProfiler

    # A backward wall-clock correction must not yield a negative duration.
    wall = [100.0]

    def backwards():
        wall[0] -= 1.0
        return wall[0]

    monkeypatch.setattr(time, "time", backwards)
    p = TRFProfiler()
    p.time_operation("fun", jnp.ones(3))
    p.time_conversion("svd_convert", time.perf_counter())
    assert p.ftimes[0] >= 0.0 and p.svd_ctimes[0] >= 0.0


@pytest.mark.parametrize("loss", ["huber", "soft_l1", "cauchy", "arctan"])
def test_masked_robust_loss_gradient_ignores_excluded_nan_row(loss):
    import jax

    from nlsq.core.loss_functions import LossFunctionsJIT

    fn = LossFunctionsJIT().get_loss_function(loss)
    mask = jnp.array([False, True])
    f = jnp.array([jnp.nan, 1.0])
    g = jax.grad(lambda r: fn(r, 1.0, mask, cost_only=True))(f)
    assert float(g[0]) == 0.0 and np.isfinite(float(g[1]))
    assert np.isfinite(float(fn(f, 1.0, mask, cost_only=True)))


def test_sparse_fd_uses_autodiff_for_large_constant_offset():
    from nlsq.core.sparse_jacobian import SparseJacobianComputer

    f = lambda x, a: 1e10 + a * x
    x = jnp.array([1.0, 2.0, 3.0])
    J = SparseJacobianComputer().compute_sparse_jacobian(
        None, np.array([1.0]), x, np.zeros(3), func=f
    )
    np.testing.assert_allclose(J.toarray().ravel(), [1.0, 2.0, 3.0])


def test_sparse_jacobian_rejects_mismatched_lengths():
    from nlsq.core.sparse_jacobian import SparseJacobianComputer

    with pytest.raises(ValueError, match=r"same points|ydata has"):
        SparseJacobianComputer().compute_sparse_jacobian(
            None,
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.zeros(4),
            chunk_size=2,
            func=lambda x, a: a * x,
        )


def test_sparse_paths_reject_complex():
    from scipy.sparse import csr_matrix

    from nlsq.core.sparse_jacobian import SparseJacobianComputer

    c = SparseJacobianComputer()
    with pytest.raises(ValueError, match="complex"):
        c.sparse_normal_equations(csr_matrix(np.array([[1j], [2j]])), np.ones(2))
    with pytest.raises(ValueError, match="complex"):
        c.compute_sparse_jacobian(
            None,
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.zeros(2),
            func=lambda x, a: 1j * a * np.asarray(x),
        )


def test_detect_sparsity_prefers_residual_count_axis_match():
    from nlsq.core.sparse_jacobian import detect_sparsity_at_p0

    x = np.arange(15.0).reshape(3, 5)  # points-first: 3 residuals, 5 features
    ratio, _ = detect_sparsity_at_p0(lambda x, a: a * x[:, 4], np.array([1.0]), x, 3)
    assert 0.0 <= ratio <= 1.0


@pytest.mark.parametrize("bounded", [False, True])
def test_trf_rejects_nonfinite_trial_cost_from_custom_loss(bounded):
    from nlsq.core.least_squares import LeastSquares

    def loss(z):
        valid = z <= 4.0
        rho0 = jnp.where(valid, z, jnp.nan)
        return jnp.stack([rho0, jnp.ones_like(z), jnp.zeros_like(z)])

    def fun(p):
        return (p - 9.0) ** 2 - 1.0

    kw = {"bounds": ([0.0], [20.0])} if bounded else {}
    res = LeastSquares().least_squares(
        fun, np.array([9.1]), loss=loss, max_nfev=25, **kw
    )
    # the NaN-cost trials are rejected and the radius shrinks into the valid region
    np.testing.assert_allclose(res.x, [10.0], atol=1e-6)
    assert res.nfev < 25


@pytest.mark.parametrize("bounds", [None, ([0.0], [10.0])])
def test_zero_step_fit_still_saves_iteration_history(tmp_path, monkeypatch, bounds):
    from nlsq import curve_fit

    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    x = np.linspace(0, 1, 20)
    kw = {} if bounds is None else {"bounds": bounds}
    res = curve_fit(lambda x, a: a * x, x, 2.0 * x, p0=[2.0], **kw)
    assert res.nfev == 1
    assert len(list(tmp_path.glob("optimization_history_*.npz"))) == 1


def test_nested_fit_keeps_outer_history_drop_count(monkeypatch, tmp_path):
    from nlsq.core.trf import _saves_iteration_history
    from nlsq.utils.logging import _History

    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))

    class _Logger:
        optimization_history = _History(maxlen=3)

        def save_iteration_data(self):
            pass

        def warning(self, *a, **k):
            pass

    class _Fit:
        logger = _Logger()

        @_saves_iteration_history
        def run(self, nested):
            for i in range(5):  # overflows the maxlen-3 buffer: 2 dropped
                self.logger.optimization_history.append(i)
            if nested:
                self.run(False)

    fit = _Fit()
    fit.run(
        True
    )  # the outer fit's saved state is checked right after the nested restore
    h = fit.logger.optimization_history
    assert list(h) == [2, 3, 4] and h.dropped == 2


# --- least_squares.py ------------------------------------------------------


def _ls(**kw):
    from nlsq.core.least_squares import LeastSquares

    return LeastSquares(**kw)


def test_wrapper_reuse_distinguishes_global_namespaces():
    src = "def f(x, a):\n    return scale * a * x\ndef j(x, a):\n    return scale * x\n"
    fns = []
    for scale in (2.0, 4.0):
        ns = {"scale": scale}
        exec(src, ns)
        fns.append((ns["f"], ns["j"]))
    x = jnp.arange(1.0, 6.0)
    y = 6.0 * x
    ls = _ls()
    r1 = ls.least_squares(fns[0][0], np.array([1.0]), jac=fns[0][1], xdata=x, ydata=y)
    r2 = ls.least_squares(fns[1][0], np.array([1.0]), jac=fns[1][1], xdata=x, ydata=y)
    np.testing.assert_allclose(r1.x, [3.0], rtol=1e-6)
    np.testing.assert_allclose(r2.x, [1.5], rtol=1e-6)


def test_stability_wrapper_does_not_accept_out_of_domain_trial():
    def fun(p):
        return jnp.where(p > 0, 1e11 + p, jnp.inf)

    res = _ls(enable_stability=True).least_squares(fun, np.array([1.0]))
    assert float(res.x[0]) > 0
    assert np.isfinite(float(fun(jnp.asarray(res.x))[0]))


def test_data_mask_is_validated_against_observations():
    f = lambda x, a: a * x
    with pytest.raises(ValueError, match="observations"):
        _ls().least_squares(
            f,
            np.array([1.0]),
            xdata=jnp.array([2.0]),
            ydata=jnp.array([6.0]),
            data_mask=np.array([True, True, True]),
        )
    with pytest.raises(ValueError, match="excludes every observation"):
        _ls().least_squares(
            lambda p: p - 2.0, np.array([0.0]), data_mask=np.array([False])
        )


def test_direct_residual_rejects_transform():
    with pytest.raises(ValueError, match="transform"):
        _ls().least_squares(
            lambda p: jnp.array([p[0] - 2.0, p[0] - 4.0]),
            np.array([0.0]),
            transform=jnp.array([100.0, 1.0]),
        )


def test_direct_residual_masked_nan_row_does_not_poison_reverse_mode():
    def fun(p):
        return jnp.sqrt(p[0] - jnp.array([-3.0, 1.0])) - 2.0

    res = _ls().least_squares(
        fun,
        np.array([0.0]),
        data_mask=np.array([True, False]),
        jacobian_mode="rev",
    )
    np.testing.assert_allclose(res.x, [1.0], atol=1e-6)


def test_length_one_analytic_jacobian_is_rejected():
    x = jnp.arange(3.0)
    with pytest.raises(ValueError, match="one derivative per observation"):
        _ls().least_squares(
            lambda x, a: a * x,
            np.array([1.0]),
            jac=lambda x, a: jnp.array([1.0]),
            xdata=x,
            ydata=2.0 * x - 1.0,
        )


def test_unhashable_callable_loss_is_accepted():
    import dataclasses

    @dataclasses.dataclass
    class Quad:
        k: float = 1.0

        def __call__(self, z):
            return jnp.stack([z, jnp.ones_like(z), jnp.zeros_like(z)])

    res = _ls().least_squares(lambda p: p - 2.0, np.array([0.0]), loss=Quad())
    np.testing.assert_allclose(res.x, [2.0], atol=1e-6)


def test_diagnostics_receive_the_optimizers_actual_counters():
    from nlsq.core.least_squares import LeastSquares

    def rosen(p):
        return jnp.array([10 * (p[1] - p[0] ** 2), 1 - p[0]])

    ls = LeastSquares(enable_diagnostics=True)
    res = ls.least_squares(rosen, np.array([-2.0, 1.0]))
    stats = ls.diagnostics.get_summary_statistics()
    assert stats["function_evaluations"] == res.nfev
    assert ls.diagnostics.jacobian_eval_count == res.njev


def test_exponential_decay_origin_amplitude_uses_log_space():
    from nlsq.core.functions import exponential_decay

    x = -750 + np.log(2) * np.arange(3)
    a, _b, _c = exponential_decay.estimate_p0(x, np.array([1e100, 5e99, 0.0]))
    np.testing.assert_allclose(a, 1.90168e-226, rtol=1e-4)
    x = 720 + np.log(2) * np.arange(3)
    a, b, _c = exponential_decay.estimate_p0(x, np.array([1e-100, 5e-101, 0.0]))
    np.testing.assert_allclose(np.log(a), np.log(1e-100) + b * 720, rtol=1e-12)
    assert 4.9e212 < a < 5.0e212


def test_exponential_decay_underflowing_amplitude_raises_but_true_zero_works():
    from nlsq.core.functions import exponential_decay

    x = -850 + np.log(2) * np.arange(3)
    with pytest.raises(ValueError, match="not representable"):
        exponential_decay.estimate_p0(x, np.array([1.0, 0.5, 0.0]))
    a, _b, _c = exponential_decay.estimate_p0(np.arange(3.0), np.zeros(3))
    assert a == 0.0


def test_power_law_underflowing_amplitude_falls_back():
    from nlsq.core.functions import power_law

    x = np.array([1e-100, 2e-100, 4e-100])
    y = 1e-330 * x ** (
        -2.0
    )  # true a ~ 1e-330 (below the denormals): exp(log_a) underflows
    assert power_law.estimate_p0(x, y) == [1.0, 1.0]  # was a = 0.0
