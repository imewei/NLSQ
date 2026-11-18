"""Regression tests for the 2026-11-15 core review (trf / least_squares / callbacks).

Each test failed on HEAD before the corresponding fix.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares

from nlsq.callbacks import StopOptimization
from nlsq.core.least_squares import LeastSquares
from nlsq.core.trf import TrustRegionReflective

_T = jnp.linspace(0.0, 4.0, 50)
_Y = 2.5 * jnp.exp(-1.3 * _T) + 0.01 * jnp.sin(7.0 * _T)


def _res(p):
    return p[0] * jnp.exp(-p[1] * _T) - _Y


def _sres(p):
    t, y = np.asarray(_T), np.asarray(_Y)
    return p[0] * np.exp(-p[1] * t) - y


def _jac(p):
    e = jnp.exp(-p[1] * _T)
    return jnp.stack([e, -p[0] * _T * e], axis=1)


_BOUNDS = [(-np.inf, np.inf), ([0.0, 0.0], [10.0, 10.0])]


# --- non-finite Jacobian: fail fast with status -4 --------------------------


@pytest.mark.parametrize("bounds", _BOUNDS)
def test_nan_jacobian_after_accepted_step_fails_fast(bounds):
    def jac(p):
        return jnp.where(p[0] > 1.5, jnp.nan, _jac(p))

    r = LeastSquares().least_squares(_res, [1.0, 0.3], jac=jac, bounds=bounds)
    assert r.status == -4
    assert not r.success
    assert "non-finite" in r.message
    assert r.nfev < 10  # HEAD burned ~100 residual evaluations, then status -3
    assert np.all(np.isfinite(np.asarray(r.x)))


@pytest.mark.parametrize("bounds", _BOUNDS)
def test_nan_jacobian_at_x0_fails_fast(bounds):
    r = LeastSquares().least_squares(
        _res, [1.0, 0.3], jac=lambda p: jnp.full((50, 2), jnp.nan), bounds=bounds
    )
    assert r.status == -4
    assert r.nfev == 1


def test_convergence_criteria_reports_nonfinite_gradient():
    trf = TrustRegionReflective()
    status, g_norm = trf._check_convergence_criteria(jnp.array([1.0, jnp.nan]), 1e-8)
    assert status == -4
    assert not np.isfinite(float(g_norm))
    status, _ = trf._check_convergence_criteria(jnp.array([jnp.inf, 0.0]), 1e-8)
    assert status == -4


# --- gtol overwrites ftol/xtol as in scipy ----------------------------------


@pytest.mark.parametrize(
    ("bounds", "x0"),
    [
        ((-np.inf, np.inf), [1.0, 0.5]),
        (([0.0, 0.0], [3.0, 1.0]), [1.0, 0.5]),
    ],
)
def test_gtol_status_matches_scipy(bounds, x0):
    # Problem where gtol is met on the iteration that also meets ftol.
    t = np.linspace(0, 4, 120)
    rng = np.random.default_rng(1)
    yn = 2.5 * np.exp(-1.3 * t) + 0.05 * rng.normal(size=120)
    yn[::17] += 1.0
    tj, yj = jnp.asarray(t), jnp.asarray(yn)
    tol = {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8}
    if bounds[0] == -np.inf:  # unbounded: use the smooth module problem
        r = LeastSquares().least_squares(_res, x0, bounds=bounds, **tol)
        s = scipy_least_squares(_sres, x0, bounds=bounds, **tol)
    else:
        r = LeastSquares().least_squares(
            lambda p: p[0] * jnp.exp(-p[1] * tj) - yj, x0, bounds=bounds, **tol
        )
        s = scipy_least_squares(
            lambda p: p[0] * np.exp(-p[1] * t) - yn, x0, bounds=bounds, **tol
        )
    if (
        s.status != 1
    ):  # SciPy's own result is platform-dependent (ftol first on Windows)
        pytest.skip(f"SciPy reports status {s.status}; the gtol premise does not hold")
    assert r.status == s.status
    assert r.nfev == s.nfev


# --- verbose ---------------------------------------------------------------


def test_verbose_1_prints_termination_report(capsys):
    LeastSquares().least_squares(_res, [1.0, 0.5], verbose=1)
    out = capsys.readouterr().out
    assert "termination condition is satisfied" in out
    assert "Function evaluations" in out


def test_verbose_2_prints_iteration_table(capsys):
    LeastSquares().least_squares(_res, [1.0, 0.5], verbose=2)
    out = capsys.readouterr().out
    assert "Iteration" in out
    assert "Optimality" in out
    assert "Function evaluations" in out


def test_verbose_0_is_silent(capsys):
    LeastSquares().least_squares(_res, [1.0, 0.5])
    assert capsys.readouterr().out == ""


# --- enable_stability must not hide Inf residuals at x0 ---------------------


def test_stability_mode_rejects_infinite_initial_residuals():
    def res(p):
        return jnp.where(p[1] > 800, jnp.inf, p[0] * jnp.exp(p[1] * _T) - _Y)

    with pytest.raises(ValueError, match="not finite"):
        LeastSquares(enable_stability=True).least_squares(res, [1.0, 900.0])


# --- callbacks -------------------------------------------------------------


def test_callback_stopiteration_stops_fit():
    calls = []

    def cb(iteration, cost, params, info):
        calls.append(iteration)
        raise StopIteration

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must not be reported as a callback error
        r = LeastSquares().least_squares(_res, [1.0, 0.5], callback=cb)
    assert r.status == -2
    assert calls == [1]


def test_callback_returning_true_stops_fit():
    r = LeastSquares().least_squares(
        _res, [1.0, 0.5], callback=lambda it, cost, params, info: True
    )
    assert r.status == -2
    assert r.nit == 1


def test_callback_returning_other_values_does_not_stop():
    r = LeastSquares().least_squares(
        _res, [1.0, 0.5], callback=lambda it, cost, params, info: 1
    )
    assert r.status > 0


def test_callback_exception_is_still_reported_not_fatal():
    def cb(iteration, cost, params, info):
        raise ZeroDivisionError("boom")

    with pytest.warns(RuntimeWarning, match="boom"):
        r = LeastSquares().least_squares(_res, [1.0, 0.5], callback=cb)
    assert r.success


@pytest.mark.parametrize("bounds", _BOUNDS)
def test_callback_gradient_norm_is_post_step(bounds):
    seen = []

    def cb(iteration, cost, params, info):
        g = info["gradient"]
        seen.append(info["gradient_norm"])
        if bounds[0] == -np.inf:
            assert info["gradient_norm"] == pytest.approx(np.max(np.abs(g)))

    r = LeastSquares().least_squares(_res, [1.0, 0.5], bounds=bounds, callback=cb)
    assert len(seen) >= 2
    if bounds[0] == -np.inf:
        # norms strictly decrease towards the optimum; stale values would lag
        assert seen[-1] == pytest.approx(r.optimality, rel=1e-12)


@pytest.mark.parametrize("bounds", _BOUNDS)
def test_optimality_consistent_with_grad_after_callback_stop(bounds):
    def cb(iteration, cost, params, info):
        raise StopOptimization

    r = LeastSquares().least_squares(_res, [1.0, 0.5], bounds=bounds, callback=cb)
    assert r.status == -2
    if bounds[0] == -np.inf:
        assert r.optimality == pytest.approx(np.max(np.abs(r.grad)))
    else:  # Coleman-Li scaled gradient at the returned x, not the pre-step one
        from nlsq.common_jax import CL_scaling_vector_jax

        v, _ = CL_scaling_vector_jax(
            jnp.asarray(r.x), jnp.asarray(r.grad), jnp.zeros(2), jnp.full(2, 10.0)
        )
        assert r.optimality == pytest.approx(
            float(jnp.max(jnp.abs(r.grad * v))), rel=1e-12
        )


# --- timeit ------------------------------------------------------------------


def test_timeit_records_function_and_jacobian_times():
    # Round 3: the initial x0 evaluations are recorded too, so the counts are
    # exactly nfev and njev (the earlier nfev - 1 / njev - 1 excluded them).
    r = LeastSquares().least_squares(_res, [1.0, 0.5], timeit=True)
    assert len(r.all_times["ftimes"]) == r.nfev
    assert len(r.all_times["jtimes"]) == r.njev
    assert all(t >= 0 for t in r.all_times["ftimes"])
    assert LeastSquares().least_squares(_res, [1.0, 0.5]).all_times == {}


def test_timeit_measures_function_execution_not_just_the_sync():
    import time

    def slow_res(p):
        time.sleep(0.03)
        return _res(p)

    def slow_jac(p):
        time.sleep(0.03)
        return _jac(p)

    r = LeastSquares().least_squares(slow_res, [1.0, 0.5], jac=slow_jac, timeit=True)
    assert len(r.all_times["ftimes"]) == r.nfev
    assert len(r.all_times["jtimes"]) == r.njev
    assert min(r.all_times["ftimes"]) >= 0.025
    assert min(r.all_times["jtimes"]) >= 0.025


# --- method='lm' validation as scipy ------------------------------------------


def test_lm_rejects_robust_loss():
    with pytest.raises(ValueError, match="only 'linear' loss"):
        LeastSquares().least_squares(_res, [1.0, 0.5], method="lm", loss="huber")


def test_lm_rejects_disabled_tolerance():
    with pytest.raises(ValueError, match="machine epsilon"):
        LeastSquares().least_squares(_res, [1.0, 0.5], method="lm", ftol=None)


def test_lm_underdetermined_is_both_typeerror_and_valueerror():
    def under(p):
        return jnp.array([p[0] - 1.0])

    with pytest.raises(ValueError, match="must not exceed"):
        LeastSquares().least_squares(under, [1.0, 0.5], method="lm")
    with pytest.raises(TypeError, match="must not exceed"):
        LeastSquares().least_squares(under, [1.0, 0.5], method="lm")


def test_lm_default_still_works():
    r = LeastSquares().least_squares(_res, [1.0, 0.5], method="lm")
    assert r.success


# --- tr_options / tr_solver='lsmr' honesty -------------------------------------


def test_tr_options_with_lsmr_warns():
    with pytest.warns(UserWarning, match="tr_options are not used"):
        LeastSquares().least_squares(
            _res, [1.0, 0.5], tr_solver="lsmr", tr_options={"regularize": True}
        )


def test_tr_options_exact_and_empty_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LeastSquares().least_squares(_res, [1.0, 0.5], tr_options={"x": 1})
        LeastSquares().least_squares(_res, [1.0, 0.5], tr_solver="lsmr")


# --- result types / f_scale -----------------------------------------------------


@pytest.mark.parametrize("bounds", _BOUNDS)
def test_result_types_match_scipy(bounds):
    r = LeastSquares().least_squares(_res, [1.0, 0.5], bounds=bounds)
    assert isinstance(r.x, np.ndarray)
    assert isinstance(r.active_mask, np.ndarray)
    assert np.issubdtype(r.active_mask.dtype, np.integer)


def test_linear_loss_accepts_zero_f_scale():
    r = LeastSquares().least_squares(_res, [1.0, 0.5], f_scale=0)
    assert r.success
    with pytest.raises(ValueError, match="f_scale"):
        LeastSquares().least_squares(_res, [1.0, 0.5], loss="huber", f_scale=0)


# --- iteration-history save must never mask the fit ------------------------------


def test_history_save_failure_does_not_mask_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    ls = LeastSquares()

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ls.trf.logger, "save_iteration_data", boom)
    r = ls.least_squares(_res, [1.0, 0.5])
    assert r.success


# --- inner CG tolerance must not follow a large outer gtol -------------------------


@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), ([-10.0, -10.0], [10.0, 10.0])])
@pytest.mark.parametrize("solver", ["lsmr", "exact"])
def test_large_gtol_does_not_stall_inner_cg_solve(bounds, solver):
    def res(p):
        return jnp.array([10.0 * (p[0] - 2.0), 3.0 * (p[1] - 1.0)])

    s = scipy_least_squares(
        lambda p: np.array([10.0 * (p[0] - 2.0), 3.0 * (p[1] - 1.0)]),
        [0.0, 0.0],
        bounds=bounds,
        gtol=2.0,
    )
    r = LeastSquares().least_squares(
        res, [0.0, 0.0], bounds=bounds, tr_solver=solver, gtol=2.0
    )
    # gtol=2 is met once the gradient inf-norm drops below 2 (like scipy), not at x0
    assert r.nfev > 1
    assert r.cost < 1.0  # HEAD: stuck at [0, 0], cost 204.5, status 4
    assert r.status == s.status == 1
    np.testing.assert_allclose(np.asarray(r.x), s.x, atol=0.2)


@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), ([-10.0], [10.0])])
def test_callback_stop_reports_current_gradient_norm(bounds):
    seen = {}

    def cb(iteration, cost, params, info):
        seen["g"] = info["gradient_norm"]
        seen["grad"] = info["gradient"]
        raise StopOptimization

    r = LeastSquares().least_squares(
        lambda p: jnp.array([p[0] - 2.0]), [0.0], bounds=bounds, callback=cb
    )
    # HEAD reported the pre-step norm (|g(x0)| = 2) in both the callback and
    # the result; the step is trust-region limited so x is not yet 2.
    x = float(np.asarray(r.x)[0])
    if bounds[0] == -np.inf:
        expected = abs(x - 2.0)
    else:
        from nlsq.common_jax import CL_scaling_vector_jax

        v, _ = CL_scaling_vector_jax(
            jnp.asarray(r.x), jnp.asarray(r.grad), jnp.array([-10.0]), jnp.array([10.0])
        )
        expected = float(jnp.max(jnp.abs(r.grad * v)))
    assert seen["g"] == pytest.approx(expected, rel=1e-12)
    assert r.optimality == pytest.approx(expected, rel=1e-12)
    assert expected != pytest.approx(2.0)


# --- round 2: stability recovery, bounded all_times -------------------------------


def test_stability_clipping_recovers_when_start_becomes_usable():
    # 1/x overflows at x0=0; clipping + zeroing the Inf Jacobian row leaves a
    # nonzero gradient, so the fit proceeds to the true minimiser (HEAD: 1.3803).
    r = LeastSquares(enable_stability=True).least_squares(
        lambda x: jnp.stack([1.0 / x[0], x[0] - 1.0]), [0.0]
    )
    assert r.success
    assert float(np.asarray(r.x)[0]) == pytest.approx(1.380304, abs=1e-5)


def test_stability_nan_residual_at_start_still_raises():
    with pytest.raises(ValueError, match="not finite"):
        LeastSquares(enable_stability=True).least_squares(
            lambda x: jnp.stack([jnp.nan * x[0], x[0] - 1.0]), [0.5]
        )


def test_timeit_with_bounds_returns_empty_all_times():
    r = LeastSquares().least_squares(
        _res, [1.0, 0.5], bounds=([0.0, 0.0], [10.0, 10.0]), timeit=True
    )
    assert r.all_times == {}


# --- round 3: inner CG tolerance with a disabled / negative gtol --------------------


@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), ([-10.0, -10.0], [10.0, 10.0])])
def test_negative_gtol_does_not_stall_inner_cg_solve(bounds):
    def res(p):
        return jnp.array([10.0 * (p[0] - 2.0), 3.0 * (p[1] - 1.0)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = LeastSquares().least_squares(
            res, [0.0, 0.0], bounds=bounds, tr_solver="lsmr", gtol=-2.0
        )
    assert r.cost < 1e-6  # HEAD: stuck at [0, 0], cost 204.5, status 4
    np.testing.assert_allclose(np.asarray(r.x), [2.0, 1.0], atol=1e-3)


# --- round 3: non-finite gradient beats a simultaneous ftol termination --------------


@pytest.mark.parametrize(
    ("bounds", "ftol"), [((-np.inf, np.inf), 0.9), (([-10.0], [10.0]), 1.1)]
)
@pytest.mark.parametrize("with_callback", [False, True])
def test_nonfinite_gradient_overrides_ftol_termination(bounds, ftol, with_callback):
    def jac(p):
        return jnp.where(p[0] == 0, jnp.array([[1.0]]), jnp.array([[jnp.nan]]))

    cb = (lambda *a: True) if with_callback else None  # would stop with success
    r = LeastSquares().least_squares(
        lambda p: jnp.array([p[0] - 2.0]),
        [0.0],
        jac=jac,
        bounds=bounds,
        ftol=ftol,
        callback=cb,
    )
    assert r.status == -4
    assert not r.success
    assert np.all(np.isfinite(np.asarray(r.x)))
