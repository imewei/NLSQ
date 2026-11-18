"""Round-4 regressions for CurveFitResult statistics.

* confidence_band / prediction_interval use the exact model Jacobian
  (jax.jacfwd) instead of a forward difference whose absolute step floor
  swamped tiny parameters.
* AIC/BIC use the sigma-weighted chi-square when sigma is supplied.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from nlsq import curve_fit


def _hand_band_halfwidth(model, x, res):
    J = np.asarray(
        jax.jacfwd(lambda q: model(jnp.asarray(x), *q))(jnp.asarray(res.popt))
    )
    dof = len(res.ydata) - len(res.popt)
    t = stats.t.ppf(0.975, dof)
    return t * np.sqrt(np.einsum("ij,jk,ik->i", J, res.pcov, J))


def _decay(x, a, b):
    return a * jnp.exp(-b * x)


def _rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def test_confidence_band_small_parameter_exact_jacobian():
    xb = np.linspace(0, 1e8, 50)
    yb = 2 * np.exp(-1e-8 * xb) + 0.01 * np.random.default_rng(5).standard_normal(50)
    res = curve_fit(_decay, xb, yb, p0=[1.0, 1.5e-8])
    assert abs(res.popt[1]) < 1e-7  # the regime the old abs-floor step broke
    hw = _hand_band_halfwidth(_decay, xb, res)
    f0 = np.asarray(_decay(xb, *res.popt))
    lo, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-8
    assert _rel(f0 - lo, hw) <= 1e-8


def test_confidence_band_ordinary_scale_exact_jacobian():
    x = np.linspace(0, 5, 60)
    y = 3 * np.exp(-0.7 * x) + 0.02 * np.random.default_rng(1).standard_normal(60)
    res = curve_fit(_decay, x, y, p0=[1.0, 1.0])
    hw = _hand_band_halfwidth(_decay, x, res)
    f0 = np.asarray(_decay(x, *res.popt))
    _, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-10
    # prediction interval shares the Jacobian
    s2 = np.sum((y - f0) ** 2) / (len(x) - 2)
    t = stats.t.ppf(0.975, len(x) - 2)
    hw_pi = np.sqrt((hw / t) ** 2 + s2) * t
    pi = res.prediction_interval()
    assert _rel(pi[:, 1] - f0, hw_pi) <= 1e-10


def test_band_falls_back_for_non_traceable_model():
    """A numpy-only model cannot be traced; central differences take over."""
    x = np.linspace(0, 5, 40)
    y = 3 * np.exp(-0.7 * x) + 0.02 * np.random.default_rng(2).standard_normal(40)
    res = curve_fit(_decay, x, y, p0=[1.0, 1.0])
    hw = _hand_band_halfwidth(_decay, x, res)
    f0 = np.asarray(_decay(x, *res.popt))
    res["model"] = lambda x, a, b: a * np.exp(-b * np.asarray(x))  # untraceable
    _, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-6


def _lin(x, a, b):
    return a * x + b


def _quad(x, a, b, c):
    return a * x**2 + b * x + c


def _het_data(seed):
    n = 200
    x = np.linspace(0, 10, n)
    sig = 0.05 * (1 + x**2)
    y = 1 + 0.5 * x + sig * np.random.default_rng(seed).standard_normal(n)
    return x, y, sig


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_aic_bic_use_weighted_chi2_with_sigma(absolute_sigma):
    x, y, sig = _het_data(0)
    res = curve_fit(_quad, x, y, p0=[0, 1, 1], sigma=sig, absolute_sigma=absolute_sigma)
    n, k = len(y), 3
    chi2 = np.sum(((y - np.asarray(_quad(x, *res.popt))) / sig) ** 2)
    base = chi2 if absolute_sigma else n * np.log(chi2 / n)
    np.testing.assert_allclose(res.aic, base + 2 * k, rtol=1e-10)
    np.testing.assert_allclose(res.bic, base + k * np.log(n), rtol=1e-10)


def test_aic_with_covariance_sigma_uses_whitened_chi2():
    x, y, sig = _het_data(1)
    C = np.diag(sig**2)
    C[0, 1] = C[1, 0] = 0.3 * sig[0] * sig[1]
    res = curve_fit(_lin, x, y, p0=[1, 1], sigma=C, absolute_sigma=True)
    r = y - np.asarray(_lin(x, *res.popt))
    chi2 = r @ np.linalg.solve(C, r)
    np.testing.assert_allclose(res.aic, chi2 + 4, rtol=1e-10)


def test_aic_bic_unweighted_unchanged():
    x, y, _ = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1])
    n, k = len(y), 2
    rss = np.sum((y - np.asarray(_lin(x, *res.popt))) ** 2)
    assert res.aic == 2 * k + n * np.log(rss / n)
    assert res.bic == k * np.log(n) + n * np.log(rss / n)


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_heteroscedastic_aic_prefers_true_linear_model(absolute_sigma):
    # seed 3: unweighted-RSS AIC/BIC picked the quadratic; chi2 picks linear
    x, y, sig = _het_data(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = curve_fit(_lin, x, y, p0=[1, 1], sigma=sig, absolute_sigma=absolute_sigma)
        r2 = curve_fit(
            _quad, x, y, p0=[0, 1, 1], sigma=sig, absolute_sigma=absolute_sigma
        )
    assert r1.aic < r2.aic
    assert r1.bic < r2.bic


# --- three-brain review 2026-11-15: nlsq/result fixes -----------------------


def _weighted_exp_problem():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 200)
    sig = 0.05 + 0.2 * x
    y = 2 * np.exp(-0.7 * x) + rng.normal(0, sig)
    return x, y, sig


def test_cmaes_result_keeps_sigma_weighting_like_trf():
    """Non-TRF CurveFitResult sites must store sigma/absolute_sigma."""
    x, y, sig = _weighted_exp_problem()
    kw = {"sigma": sig, "bounds": ([0, 0], [10, 10])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trf = curve_fit(_decay, x, y, p0=[1, 1], **kw)
        cma = curve_fit(_decay, x, y, p0=[1, 1], method="cmaes", **kw)
    np.testing.assert_array_equal(cma["sigma"], sig)
    assert cma["absolute_sigma"] is False
    np.testing.assert_allclose(cma.aic, trf.aic, rtol=1e-3)
    np.testing.assert_allclose(cma.bic, trf.bic, rtol=1e-3)
    np.testing.assert_allclose(
        cma.prediction_interval(), trf.prediction_interval(), rtol=5e-2, atol=1e-2
    )


def _restored(**kw):
    from nlsq.result import CurveFitResult

    return CurveFitResult(**kw)


def test_fallback_predictions_residuals_undo_sigma_weighting():
    r = _restored(
        x=np.array([1.0]),
        ydata=np.array([3.0, 5.0]),
        fun=np.array([-1.0, -1.0]),
        sigma=np.array([2.0, 2.0]),
    )
    np.testing.assert_allclose(r.predictions, [1.0, 3.0])
    np.testing.assert_allclose(r.residuals, [2.0, 2.0])
    np.testing.assert_allclose(r._chi2(), 2.0)  # (2/2)^2 * 2, not double-weighted


def test_fallback_covariance_sigma_uses_cholesky_factor():
    C = np.array([[4.0, 1.0], [1.0, 9.0]])
    L = np.linalg.cholesky(C)
    raw = np.array([1.0, -2.0])  # f - y
    r = _restored(
        x=np.array([1.0]),
        ydata=np.array([3.0, 5.0]),
        fun=np.linalg.solve(L, raw),
        sigma=C,
    )
    np.testing.assert_allclose(r.residuals, -raw)
    np.testing.assert_allclose(r._chi2(), raw @ np.linalg.solve(C, raw))


def test_fallback_unmatched_sigma_raises():
    r = _restored(
        x=np.array([1.0]),
        ydata=np.array([3.0, 5.0]),
        fun=np.array([-1.0, -1.0]),
        sigma=np.ones(3),
    )
    with pytest.raises(ValueError, match="sigma weighting"):
        _ = r.residuals


def test_band_small_parameter_numpy_model_stays_finite():
    def model(x, a):  # numpy-only: forces the finite-difference fallback
        return np.exp(-a * x)

    r = _restored(
        x=np.array([1e-8]),
        pcov=np.array([[1e-16]]),
        xdata=np.array([1e8, 1e8, 1e8]),
        ydata=np.array([1.0, 1.0, 1.0]),
        model=model,
    )
    lo, hi = r.confidence_band(np.array([1e8]))
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    # exact: |df/da| = x exp(-1) -> se = 3.68e7 * 1e-8; t(dof=2) = 4.3027
    half = 4.302653 * 1e8 * np.exp(-1.0) * 1e-8
    np.testing.assert_allclose(hi - lo, 2 * half, rtol=1e-4)


def test_nonfinite_finite_difference_jacobian_raises():
    def model(x, a):  # float(a) cannot be traced -> finite differences; overflows
        return np.exp(float(a) * x)

    r = _restored(
        x=np.array([1.0]),
        pcov=np.array([[1.0]]),
        xdata=np.array([1000.0, 1000.0, 1000.0]),
        ydata=np.array([1.0, 1.0, 1.0]),
        model=model,
    )
    with pytest.raises(FloatingPointError):
        r.confidence_band()


def test_caches_follow_result_mutation():
    r = _restored(
        x=np.array([1.0]),
        xdata=np.array([1.0, 2.0, 3.0]),
        ydata=np.array([1.0, 2.0, 3.0]),
        model=lambda x, a: a * x,
    )
    assert r.rmse == 0.0
    r["x"] = np.array([2.0])
    np.testing.assert_allclose(r.predictions, [2.0, 4.0, 6.0])
    assert r.rmse > 0
    r["ydata"] = np.array([2.0, 4.0, 6.0])
    assert r.rmse == 0.0


def test_underdetermined_intervals_are_nan_with_warning():
    """n <= p: no residual dof -> all three methods warn and return NaN."""
    r = _restored(
        x=np.array([1.0]),
        pcov=np.array([[0.1]]),
        xdata=np.array([1.0]),
        ydata=np.array([1.0]),
        model=lambda x, a: a * x,
    )
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        ci = r.confidence_intervals()
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        lo, hi = r.confidence_band()
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        pi = r.prediction_interval()
    assert ci.shape == (1, 2) and np.all(np.isnan(ci))
    assert np.all(np.isnan(lo)) and np.all(np.isnan(hi))
    assert pi.shape == (1, 2) and np.all(np.isnan(pi))


def test_underdetermined_known_noise_uses_normal_quantile():
    from scipy import stats

    r = _restored(
        x=np.array([1.0]),
        pcov=np.array([[0.04]]),
        xdata=np.array([1.0]),
        ydata=np.array([1.0]),
        model=lambda x, a: a * x,
        sigma=np.array([0.1]),
        absolute_sigma=True,
    )
    z = stats.norm.ppf(0.975)
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        ci = r.confidence_intervals()
    np.testing.assert_allclose(ci, [[1 - z * 0.2, 1 + z * 0.2]])
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        lo, hi = r.confidence_band()
    np.testing.assert_allclose(hi - lo, [2 * z * 0.2])
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        pi = r.prediction_interval()
    np.testing.assert_allclose(pi[0, 1] - pi[0, 0], 2 * z * np.sqrt(0.04 + 0.01))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 1.5, np.nan, np.inf])
def test_alpha_must_be_open_unit_interval(alpha):
    x, y, _ = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1])
    for call in (
        res.confidence_intervals,
        res.prediction_interval,
        res.confidence_band,
    ):
        with pytest.raises(ValueError, match="alpha"):
            call(alpha=alpha)


def test_scalar_x_for_interval_and_band():
    x, y, _ = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1])
    for scalar in (2.0, np.array(2.0)):
        pi = res.prediction_interval(scalar)
        lo, hi = res.confidence_band(scalar)
        assert pi.shape == (1, 2) and lo.shape == (1,) and hi.shape == (1,)
    np.testing.assert_allclose(
        res.prediction_interval(2.0), res.prediction_interval([2.0])
    )


def test_property_error_is_not_swallowed_by_getattr():
    r = _restored(x=np.array([1.0]))
    with pytest.raises(AttributeError, match="ydata not available"):
        _ = r.r_squared
    assert not hasattr(r, "r_squared")

    def bad_model(x, a):
        raise AttributeError("model typo")

    r2 = _restored(
        x=np.array([1.0]), xdata=np.ones(2), ydata=np.ones(2), model=bad_model
    )
    with pytest.raises(AttributeError, match="model typo"):
        _ = r2.residuals


def test_property_assignment_routes_or_raises():
    r = _restored(x=np.array([1.0]))
    r.popt = np.array([5.0])
    assert r.popt[0] == 5.0 and r["x"][0] == 5.0
    r.diagnostics = "report"
    assert r.diagnostics == "report"
    with pytest.raises(AttributeError, match="read-only"):
        r.residuals = np.zeros(2)


def test_nonfinite_pcov_gives_infinite_not_nan_band():
    r = _restored(
        x=np.array([1.0, 1.0]),
        pcov=np.full((2, 2), np.inf),
        xdata=np.arange(5.0),
        ydata=np.arange(5.0) + 0.1,
        model=lambda x, a, b: a * x - b * x**2,  # mixed-sign Jacobian
    )
    lo, hi = r.confidence_band(np.array([1.0, 2.0]))
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_sigma_mismatch_warns_in_prediction_interval():
    x, y, sig = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1], sigma=sig)
    res["sigma"] = sig[:-1]
    with pytest.warns(RuntimeWarning, match="pooled unweighted variance"):
        res.prediction_interval()


def test_summary_labels_function_evals_and_iterations(capsys):
    x, y, _ = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1])
    res["nit"] = 7
    res.summary()
    out = capsys.readouterr().out
    assert "Function evals" in out and "Iterations        : 7" in out


def test_underdetermined_absolute_sigma_prediction_interval_stays_finite():
    """Known noise (absolute sigma) needs no residual dof: warn, don't NaN."""
    res = curve_fit(
        _decay,
        np.array([0.0, 1.0]),
        np.array([3.0, 2.0]),
        p0=[1.0, 1.0],
        sigma=np.array([0.1, 0.1]),
        absolute_sigma=True,
    )
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        pi = res.prediction_interval([0.5])
    assert pi.shape == (1, 2) and np.all(np.isfinite(pi))
    assert pi[0, 0] < pi[0, 1]


def test_fd_fallback_handles_callback_model_without_jvp():
    """jax.jacfwd raises ValueError (not TypeError) for a pure_callback model."""

    def model(x, a):
        x = jnp.asarray(x)
        return jax.pure_callback(
            lambda xx, aa: np.asarray(xx) * np.asarray(aa),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            x,
            a,
        )

    with pytest.raises((ValueError, NotImplementedError)):
        jax.jacfwd(lambda p: model(jnp.arange(5.0), p[0]))(jnp.array([2.0]))

    r = _restored(
        x=np.array([2.0]),
        pcov=np.array([[0.01]]),
        xdata=np.arange(5.0),
        ydata=2.0 * np.arange(5.0),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(np.arange(5.0), r.popt)
    np.testing.assert_allclose(jac[:, 0], np.arange(5.0), atol=1e-6)


def test_cache_follows_sigma_replacement():
    r = _restored(
        x=np.array([1.0]),
        ydata=np.array([3.0, 5.0]),
        fun=np.array([-1.0, -1.0]),
        sigma=np.array([2.0, 2.0]),
    )
    np.testing.assert_allclose(r.residuals, [2.0, 2.0])
    r["sigma"] = np.array([4.0, 4.0])
    np.testing.assert_allclose(r.residuals, [4.0, 4.0])


def test_negative_sigma_gives_same_noise_model_and_aic():
    x, y, sig = _het_data(0)
    pos = curve_fit(_lin, x, y, p0=[1, 1], sigma=sig)
    neg = curve_fit(_lin, x, y, p0=[1, 1], sigma=-sig)
    np.testing.assert_allclose(neg.aic, pos.aic)
    np.testing.assert_allclose(neg.bic, pos.bic)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no pooled-variance fallback warning
        np.testing.assert_allclose(
            neg.prediction_interval(), pos.prediction_interval(), rtol=1e-8
        )


def test_fd_cancellation_does_not_return_false_zero_derivative():
    def model(x, a):  # float(a) -> untraceable; f ~ 1 hides a*x at tiny steps
        return np.ones_like(x) + float(a) * x

    x = np.array([1e-3, 1e-3])
    r = _restored(
        x=np.array([1e-8]),
        pcov=np.array([[1.0]]),
        xdata=x,
        ydata=model(x, 1e-8),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, r.popt)
    np.testing.assert_allclose(jac[:, 0], 1e-3, rtol=1e-3)


def test_fd_parameter_with_no_effect_gives_exact_zero_column():
    def model(x, a, b):  # `a` has no effect at all
        return np.exp(-float(b) * x) + 0.0 * float(a)

    x = np.linspace(0, 2, 7)
    r = _restored(
        x=np.array([0.3, 0.5]),
        pcov=np.eye(2),
        xdata=x,
        ydata=model(x, 0.3, 0.5),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, r.popt)
    assert np.all(jac[:, 0] == 0.0)
    np.testing.assert_allclose(jac[:, 1], -x * np.exp(-0.5 * x), rtol=1e-6, atol=1e-9)
    lo, hi = r.confidence_band(x)  # must not raise
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_fd_unresolvably_tiny_derivative_is_finite_not_an_error():
    r = _restored(
        x=np.array([1.0]),
        pcov=np.array([[1.0]]),
        xdata=np.array([1e-30, 1e-30]),
        ydata=np.array([1.0, 1.0]),
        model=lambda x, a: np.ones_like(x) + float(a) * x,
    )
    _, jac = r._finite_diff_jacobian(r.xdata, r.popt)
    assert np.all(np.isfinite(jac)) and np.all(np.abs(jac) < 1e-20)


def _head_fd(model, x, popt):
    """Reference: HEAD central difference, step cbrt(eps)*max(|p|, 1)."""
    step = np.finfo(float).eps ** (1.0 / 3.0)
    jac = np.zeros((len(x), len(popt)))
    for i in range(len(popt)):
        h = step * max(abs(popt[i]), 1.0)
        pp, pm = popt.copy(), popt.copy()
        pp[i] += h
        pm[i] -= h
        jac[:, i] = (model(x, *pp) - model(x, *pm)) / (pp[i] - pm[i])
    return jac


@pytest.mark.parametrize(
    "scale,popt",
    [
        (1.0, [2.0, 0.7, 0.3]),
        (1.0, [1e-8, 1e-4, 1e-12]),
        (1.0, [1e-2, 1.0, 1e2]),
        (1.0, [1e8, 1e-6, 5.0]),
        (1e8, [2.0, 1e-8, 1e-8]),  # Codex: tiny rate, huge x
    ],
)
def test_fd_jacobian_at_least_as_accurate_as_head(scale, popt):
    def model(x, a, b, c):  # numpy-only (float(...)) -> finite differences
        return float(a) * np.exp(-float(b) * x) + float(c)

    popt = np.array(popt, dtype=float)
    x = np.linspace(0.0, 3.0, 40) * scale
    a, b, _ = popt
    analytic = np.column_stack(
        [np.exp(-b * x), -a * x * np.exp(-b * x), np.ones_like(x)]
    )
    r = _restored(
        x=popt,
        pcov=np.eye(3),
        xdata=x,
        ydata=model(x, *popt),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, popt)
    head = _head_fd(model, x, popt)

    def col_err(j):
        with np.errstate(all="ignore"):
            e = np.linalg.norm(j - analytic, axis=0) / np.linalg.norm(analytic, axis=0)
        return np.where(np.isfinite(e), e, np.inf)

    assert np.all(col_err(jac) <= col_err(head) * (1 + 1e-9) + 1e-12)
    assert np.all(col_err(jac) < 1e-3)


def test_nonfinite_autodiff_jacobian_is_reported_like_fd():
    r = _restored(
        x=np.array([0.0]),
        pcov=np.array([[1.0]]),
        xdata=np.arange(1.0, 5.0),
        ydata=np.zeros(4),
        model=lambda x, a: jnp.sqrt(a) * x,  # d/da = inf at a = 0
    )
    with pytest.raises(FloatingPointError):
        r.confidence_band()


def test_fd_nan_probe_is_discarded_and_step_escalated_in_domain():
    """sqrt(a) at a=1e-8: the large probe leaves the domain (NaN)."""

    def model(x, a):  # float(a) -> untraceable -> finite differences
        return np.ones_like(x) + 1e-10 * np.sqrt(float(a))

    x = np.array([1.0, 2.0, 3.0])
    r = _restored(
        x=np.array([1e-8]),
        pcov=np.array([[1.0]]),
        xdata=x,
        ydata=model(x, 1e-8),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, r.popt)
    np.testing.assert_allclose(jac[:, 0], 5e-7, rtol=0.2)
    lo, hi = r.confidence_band(x)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi > lo)
    pi = r.prediction_interval(x)
    assert np.all(np.isfinite(pi))


@pytest.mark.parametrize("method", ["confidence_band", "prediction_interval"])
def test_underdetermined_returns_nan_before_differentiating(method):
    """n <= p and unknown noise: warn + NaN, never reach a bad derivative."""
    r = _restored(
        x=np.array([0.0]),
        pcov=np.array([[1.0]]),
        xdata=np.array([1.0]),
        ydata=np.array([0.0]),
        model=lambda x, a: jnp.sqrt(a) * x,  # d/da = inf at a = 0
    )
    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        out = getattr(r, method)()
    out = (
        np.concatenate([np.ravel(o) for o in np.atleast_1d(out)])
        if isinstance(out, tuple)
        else np.ravel(out)
    )
    assert out.size == 2 and np.all(np.isnan(out))


def test_fd_single_ulp_probe_is_rejected_as_noise():
    """sqrt(a) at a=1e-8: the small probe moves f by one ULP, the large probe
    leaves the domain; the derivative (0.002) must still be recovered."""

    def model(x, a):  # float(a) -> untraceable -> finite differences
        return np.ones_like(x) + 4e-7 * np.sqrt(float(a))

    x = np.array([1.0, 2.0, 3.0])
    r = _restored(
        x=np.array([1e-8]),
        pcov=np.array([[1.0]]),
        xdata=x,
        ydata=model(x, 1e-8),
        model=model,
    )
    analytic = 4e-7 * 0.5 / np.sqrt(1e-8)
    _, jac = r._finite_diff_jacobian(x, r.popt)
    np.testing.assert_allclose(jac[:, 0], analytic, rtol=0.03)
    half = 4.302653 * analytic  # t(dof=2) * |J| * sqrt(pcov)
    lo, hi = r.confidence_band(x)
    np.testing.assert_allclose((hi - lo) / 2, half, rtol=0.03)
    pi = r.prediction_interval(x)  # zero residuals: same width as the band
    np.testing.assert_allclose((pi[:, 1] - pi[:, 0]) / 2, half, rtol=0.03)


def test_fd_tiny_parameter_sqrt_derivative_is_recovered():
    """popt = 1e-16: every probe >= 6e-14 is hundreds of times the parameter."""

    def model(x, a):
        return np.sqrt(float(a)) * np.ones_like(x)

    x = np.array([1.0, 2.0, 3.0])
    r = _restored(
        x=np.array([1e-16]),
        pcov=np.array([[1e-32]]),
        xdata=x,
        ydata=model(x, 1e-16),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, r.popt)
    np.testing.assert_allclose(jac[:, 0], 0.5 / np.sqrt(1e-16), rtol=0.03)


def test_fd_tiny_parameter_steep_exponential_is_finite():
    def model(x, a):
        return np.exp(-float(a) / 1e-16) * np.ones_like(x)

    x = np.array([1.0, 2.0, 3.0])
    r = _restored(
        x=np.array([1e-16]),
        pcov=np.array([[1e-32]]),
        xdata=x,
        ydata=model(x, 1e-16),
        model=model,
    )
    _, jac = r._finite_diff_jacobian(x, r.popt)
    np.testing.assert_allclose(jac[:, 0], -np.exp(-1.0) / 1e-16, rtol=0.03)
    lo, hi = r.confidence_band(x)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def _fd_jac(model, p0, x=(1.0, 2.0, 3.0)):
    x = np.array(x)
    r = _restored(
        x=np.array([p0]),
        pcov=np.array([[1e-12]]),
        xdata=x,
        ydata=model(x, p0),
        model=model,
    )
    return r._finite_diff_jacobian(x, r.popt)[1][:, 0]


def test_fd_subnormal_parameter_keeps_standard_estimate():
    jac = _fd_jac(lambda x, a: float(a) * x, 1e-320)
    np.testing.assert_allclose(jac, [1.0, 2.0, 3.0], rtol=1e-6)


def test_fd_domain_error_probe_does_not_abort_the_ladder():
    import math

    jac = _fd_jac(lambda x, a: math.sqrt(float(a)) * np.ones_like(x), 1e-8)
    np.testing.assert_allclose(jac, 0.5 / np.sqrt(1e-8), rtol=0.03)


def test_fd_all_probes_raising_reraises_model_error():
    def model(x, a):
        if abs(float(a) - 1.0) > 0:  # every probe (and only the probes) fails
            raise ValueError("bad probe")
        return x * float(a)

    r = _restored(
        x=np.array([1.0]),
        pcov=np.array([[1.0]]),
        xdata=np.arange(3.0),
        ydata=np.arange(3.0),
        model=model,
    )
    with pytest.raises(ValueError, match="bad probe"):
        r._fd_derivative(np.arange(3.0), r.popt, 0, 6e-6)


def test_fd_near_domain_edge_needs_steps_below_1e_minus_6():
    def model(x, a):
        return np.sqrt(float(a) - (1 - 1e-8)) * np.ones_like(x)

    jac = _fd_jac(model, 1.0)
    np.testing.assert_allclose(jac, 0.5 / np.sqrt(1e-8), rtol=0.03)


def test_fd_tiny_parameter_below_the_old_ladder_cap():
    jac = _fd_jac(lambda x, a: np.sqrt(float(a)) * np.ones_like(x), 1e-30)
    np.testing.assert_allclose(jac, 0.5 / np.sqrt(1e-30), rtol=0.03)


def _outer_domain_model():
    def model(x, a):
        d = abs(float(a) - 1.0)
        if d == 0:
            return np.ones_like(x)
        if d > 1e-6:
            raise ValueError("outer domain")
        return np.full_like(x, np.nan)

    return model


def _outer_domain_result():
    x = np.array([1.0, 2.0, 3.0])
    return _restored(
        x=np.array([1.0]),
        pcov=np.array([[1.0]]),
        xdata=x,
        ydata=np.ones(3),
        model=_outer_domain_model(),
    )


def test_fd_some_probes_raise_rest_nan_gives_floating_point_error():
    r = _outer_domain_result()
    with pytest.raises(FloatingPointError):
        r.confidence_band()
    with pytest.raises(FloatingPointError):
        r.prediction_interval()


def _flat(x, a):
    return a + 0.0 * x


def test_known_sigma_intervals_use_normal_quantile_for_any_n():
    from scipy import stats

    res = curve_fit(
        _flat,
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        p0=[0.5],
        sigma=np.array([1.0, 1.0]),
        absolute_sigma=True,
    )
    z = stats.norm.ppf(0.975)
    se = np.sqrt(0.5)  # known covariance
    np.testing.assert_allclose(
        res.confidence_intervals(), [[1 - z * se, 1 + z * se]], rtol=1e-6
    )
    lo, hi = res.confidence_band()
    np.testing.assert_allclose((hi - lo) / 2, z * se, rtol=1e-6)
    pi = res.prediction_interval()  # noise variance is the known sigma^2 = 1
    np.testing.assert_allclose(
        (pi[:, 1] - pi[:, 0]) / 2, z * np.sqrt(0.5 + 1.0), rtol=1e-6
    )


def test_estimated_noise_keeps_student_t():
    from scipy import stats

    res = curve_fit(
        _flat,
        np.array([0.0, 1.0]),
        np.array([1.0, 1.2]),
        p0=[0.5],
        sigma=np.array([1.0, 1.0]),
        absolute_sigma=False,
    )
    t1 = stats.t.ppf(0.975, 1)
    perr = np.sqrt(res.pcov[0, 0])
    np.testing.assert_allclose(
        np.diff(res.confidence_intervals())[0, 0], 2 * t1 * perr, rtol=1e-6
    )


def test_alpha_near_one_and_zero_stay_valid():
    near_one = np.nextafter(1.0, 0.0)
    r = _restored(
        x=np.array([0.0]),
        pcov=np.array([[1.0]]),
        xdata=np.array([0.0, 1.0]),
        ydata=np.array([1.0, 2.0]),
        model=_flat_np,
    )
    ci = r.confidence_intervals(near_one)
    lo, hi = r.confidence_band(alpha=near_one)
    pi = r.prediction_interval(alpha=near_one)
    for a, b in ((ci[0, 0], ci[0, 1]), (lo[0], hi[0]), (pi[0, 0], pi[0, 1])):
        assert np.isfinite(a) and np.isfinite(b)
        assert b > a
    np.testing.assert_allclose(ci[0, 1], -ci[0, 0])  # symmetric about popt = 0
    np.testing.assert_allclose(ci[0, 1], 5.734e15, rtol=1e-3)
    # alpha -> 0: zero-width but valid
    np.testing.assert_allclose(r.confidence_intervals(1e-300), [[0.0, 0.0]], atol=1e-12)
    lo0, hi0 = r.confidence_band(alpha=1e-300)
    np.testing.assert_allclose(hi0 - lo0, 0.0, atol=1e-12)


def _flat_np(x, a):
    return a + 0.0 * x


def _unit_sigma_fits(x, y, p0):
    kw = {"p0": p0, "absolute_sigma": True}
    omitted = curve_fit(_flat, x, y, **kw)
    explicit = curve_fit(_flat, x, y, sigma=np.ones_like(y), **kw)
    return omitted, explicit


def _assert_same_statistics(a, b):
    np.testing.assert_allclose(a.confidence_intervals(), b.confidence_intervals())
    for u, v in zip(a.confidence_band(), b.confidence_band(), strict=True):
        np.testing.assert_allclose(u, v)
    np.testing.assert_allclose(a.prediction_interval(), b.prediction_interval())
    np.testing.assert_allclose(a.aic, b.aic)
    np.testing.assert_allclose(a.bic, b.bic)


def test_absolute_sigma_omitted_equals_explicit_unit_sigma(capsys):
    x, y = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    omitted, explicit = _unit_sigma_fits(x, y, [1.0])
    _assert_same_statistics(omitted, explicit)
    np.testing.assert_allclose(
        omitted.confidence_intervals(), [[-0.3859, 2.3859]], atol=1e-3
    )
    omitted.summary()
    out_omitted = capsys.readouterr().out
    explicit.summary()
    assert out_omitted == capsys.readouterr().out


def test_absolute_sigma_omitted_underdetermined_equals_explicit_unit_sigma():
    x, y = np.array([0.0]), np.array([1.0])
    omitted, explicit = _unit_sigma_fits(x, y, [1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _assert_same_statistics(omitted, explicit)
        assert np.all(np.isfinite(omitted.prediction_interval()))
        assert np.isfinite(omitted.aic) and np.isfinite(omitted.bic)


def test_absolute_sigma_omitted_handbuilt_result_matches_unit_sigma():
    kw = {
        "x": np.array([1.0]),
        "pcov": np.array([[0.5]]),
        "xdata": np.array([0.0, 1.0]),
        "ydata": np.array([1.0, 1.0]),
        "model": _flat_np,
        "absolute_sigma": True,
    }
    omitted = _restored(**kw)
    explicit = _restored(**kw, sigma=np.ones(2))
    _assert_same_statistics(omitted, explicit)


def test_estimated_noise_results_unchanged_without_sigma():
    x, y = np.linspace(0, 1, 6), np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    res = curve_fit(_flat, x, y, p0=[1.0])  # absolute_sigma False
    n, rss = len(y), np.sum((y - 1.0) ** 2)
    np.testing.assert_allclose(res.aic, 2 + n * np.log(np.sum(res.residuals**2) / n))
    assert rss > 0 and res._chi2() is None
