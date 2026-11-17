"""Regression tests for the streaming exact-path review (2026-11-16, round 1)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from nlsq.result.optimize_warning import OptimizeWarning
from nlsq.streaming.specs import Exact
from nlsq.streaming.telemetry import DefenseLayerTelemetry


def _exp_data(amp, n=5000, noise=0.0, seed=0):
    x = np.linspace(0.0, 1.0, n)
    y = amp * np.exp(-5.0 * x)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, n)
    return x, y


def _exp(x, a, k):
    return a * jnp.exp(-k * x)


# --- A-W1: the TSQR gate uses the Jacobi-scaled condition number ----------------------


@pytest.mark.serial
def test_scaled_condition_gate_keeps_normal_route_and_pcov():
    x, y = _exp_data(1e6, noise=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # no ill-conditioning warning on a unit-mismatch fit
        out = lm.exact_lm_fit(_exp, x, y, [9e5, 4.0], chunk_size=1024)
    assert out["covariance_method"] == "normal"
    # the QR route on the same fit gives the same covariance
    old = lm.TSQR_KAPPA
    lm.TSQR_KAPPA = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = lm.exact_lm_fit(_exp, x, y, [9e5, 4.0], chunk_size=1024)
    finally:
        lm.TSQR_KAPPA = old
    assert ref["covariance_method"] == "tsqr"
    np.testing.assert_allclose(out["pcov"], ref["pcov"], rtol=1e-8)


def test_scaled_condition_gate_still_routes_collinear_to_tsqr():
    x = np.linspace(0.0, 1.0, 4000)
    y = 2.0 * x + 3.0 * (x + 1e-5 * x**2) + 0.01 * np.sin(40 * x)

    def model(x, a, b):
        return a * x + b * (x + 1e-5 * x**2)

    with pytest.warns(UserWarning, match="ill-conditioned"):
        out = lm.exact_lm_fit(model, x, y, [1.0, 1.0], chunk_size=1024)
    assert out["covariance_method"] == "tsqr"


# --- A-W2: scale-aware gtol and damping floor ------------------------------------------


@pytest.mark.parametrize("amp", [1e-9, 1e-6, 1.0, 1e6])
def test_small_amplitude_fit_is_scale_invariant(amp):
    x, y = _exp_data(amp)
    out = lm.exact_lm_fit(_exp, x, y, [2.0 * amp, 3.0], chunk_size=1024)
    assert out["success"]
    np.testing.assert_allclose(out["x"], [amp, 5.0], rtol=1e-4)


@pytest.mark.parametrize("amp", [1e-9, 1.0, 1e6])
def test_noise_free_fit_at_the_truth_is_an_exact_fit(amp):
    x, y = _exp_data(amp)  # residual at the rounding level of y, whatever its units
    out = lm.exact_lm_fit(_exp, x, y, [amp, 5.0], chunk_size=1024)
    assert out["success"] and out["termination"] == "gtol" and out["nfev"] == 1


def test_damping_has_no_relative_floor():
    D = lm._damping(np.zeros(2), np.diag([1e3, 1e-16]))
    assert D[1] == 1e-16  # the old 1e-10 * max floor froze the small-scale parameter
    assert (
        lm._damping(np.zeros(2), np.diag([1.0, 0.0])).min() > 0
    )  # zero column stays solvable


# --- A-W3: a scalar-returning model is pointwise --------------------------------------


def test_scalar_model_is_accepted_like_curve_fit():
    x = np.linspace(0.0, 1.0, 2000)
    y = 3.0 + np.random.default_rng(1).normal(0.0, 0.01, x.size)
    out = lm.exact_lm_fit(lambda x, a: a, x, y, [1.0], chunk_size=512)
    assert out["success"]
    np.testing.assert_allclose(out["x"], [y.mean()], rtol=1e-6)


def test_wrong_shape_model_still_rejected():
    x = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="not pointwise"):
        lm.exact_lm_fit(lambda x, a: jnp.stack([a * x, a * x]), x, x, [1.0])


# --- A-S: skipped dead passes, spec validation, chunk vs memory warning -----------------


def test_nonpositive_predicted_decrease_costs_no_data_pass(monkeypatch):
    x, y = _exp_data(1.0, n=500, noise=0.01)
    # an ascent step: pred = s.g - s'Hs/2 < 0 for every damping
    monkeypatch.setattr(lm, "_solve", lambda Hf, lam, Df, gf: -gf)
    out = lm.exact_lm_fit(_exp, x, y, [2.0, 3.0], chunk_size=256)
    assert out["termination"] == "stalled"
    assert out["nfev"] == 1  # only the initial pass touched the data


@pytest.mark.parametrize(
    "kw",
    [
        {"ftol": float("nan")},
        {"xtol": -1.0},
        {"gtol": float("inf")},
        {"ftol": 0.0, "xtol": 0.0, "gtol": 0.0},
    ],
)
def test_exact_spec_validates_at_construction(kw):
    with pytest.raises(ValueError):
        Exact(**kw)


def test_exact_spec_keeps_tolerance_identity():
    tol = np.float32(1e-8)
    assert Exact(gtol=tol).gtol is tol


def test_explicit_chunk_over_memory_limit_warns():
    with pytest.warns(UserWarning, match="memory_limit_gb"):
        assert lm.derive_chunk_size(10**7, 8, 0.01, 1 << 22) == 1 << 22
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lm.derive_chunk_size(10**7, 8, 1.0, 1 << 10)
        lm.derive_chunk_size(10**7, 8, None, 1 << 22)


# --- Codex-W1: non-finite or malformed p0 -------------------------------------------------


@pytest.mark.parametrize("p0", [[np.nan], [np.inf], [[1.0]]])
def test_nonfinite_or_malformed_p0_rejected(p0):
    x = np.arange(3.0)
    with pytest.raises(ValueError, match="p0"):
        lm.exact_lm_fit(
            lambda x, a: jnp.ones_like(x), x, np.ones(3), p0, compute_covariance=False
        )


# --- Codex-S1: warmups are told apart by token -----------------------------------------


def test_concurrent_warmups_each_count_their_failed_step():
    t = DefenseLayerTelemetry()
    a, b = t.record_warmup_start(), t.record_warmup_start()
    t.record_lbfgs_iteration()
    t.record_lbfgs_iteration()
    t.record_lbfgs_line_search_failure(0, "x", warmup=a)
    t.record_lbfgs_line_search_failure(0, "x", warmup=b)
    t.record_lbfgs_line_search_failure(0, "dup", warmup=a)  # same step reported twice
    assert t.lbfgs_line_search_failed_steps == 2
    assert t.get_trigger_rates()["lbfgs_line_search_failure_rate"] == 100.0


# --- the chunked route honours the global memory limit ---------------------------------


def test_chunked_route_uses_the_global_memory_limit(monkeypatch):
    import nlsq.streaming.strategy as strat
    from nlsq import curve_fit_large
    from nlsq.config import JAXConfig, set_memory_limits
    from nlsq.streaming import Chunked

    monkeypatch.delenv("NLSQ_MEMORY_LIMIT_GB", raising=False)
    inst = JAXConfig()
    monkeypatch.setattr(inst, "_memory_config", None)  # restored on teardown
    seen = []
    real = strat.build_large_strategy

    def spy(*a, **kw):
        seen.append(kw.get("memory_limit_gb"))
        return real(*a, **kw)

    monkeypatch.setattr(strat, "build_large_strategy", spy)
    x, y = _exp_data(1.0, n=2000, noise=0.01)
    set_memory_limits(memory_limit_gb=0.5)
    curve_fit_large(_exp, x, y, p0=[1.5, 4.0], strategy=Chunked())
    assert seen == [0.5]


def test_exact_route_uses_the_global_memory_limit(monkeypatch):
    from nlsq import curve_fit_large
    from nlsq.config import JAXConfig, set_memory_limits
    from nlsq.streaming import Exact

    monkeypatch.delenv("NLSQ_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.setattr(JAXConfig(), "_memory_config", None)  # restored on teardown
    seen = []
    real = lm.exact_lm_fit

    def spy(*a, **kw):
        seen.append(kw.get("memory_limit_gb"))
        return real(*a, **kw)

    monkeypatch.setattr(lm, "exact_lm_fit", spy)
    x, y = _exp_data(1.0, n=2000, noise=0.01)
    set_memory_limits(memory_limit_gb=0.5)
    curve_fit_large(_exp, x, y, p0=[1.5, 4.0], strategy=Exact())
    curve_fit_large(_exp, x, y, p0=[1.5, 4.0], strategy=Exact(), memory_limit_gb=2.0)
    assert seen == [0.5, 2.0]


def test_large_offset_is_not_an_exact_fit():
    # |r| ~ 1e-7 per point is 1e3 eps |y| for y ~ 1e6, yet 30% of the signal
    x = np.linspace(0.0, 1.0, 5000)
    y = 1e6 + 1e-6 * np.exp(-5.0 * x)

    def model(x, c, b, k):
        return c + b * jnp.exp(-k * x)

    out = lm.exact_lm_fit(model, x, y, [1e6, 1.3e-6, 4.0], chunk_size=1024)
    # not stopped at p0 as an "exact fit" (it used to be: nfev=1, termination gtol); the
    # xtol swamp is gated by the gradient cosine, so it keeps going to the truth
    assert out["nfev"] > 1
    np.testing.assert_allclose(out["x"][1:], [1e-6, 5.0], rtol=2e-2)


def test_stall_at_rounding_level_is_labelled_and_a_larger_one_is_a_stall(monkeypatch):
    # a residual of 1e-14 relative, orthogonal to the model (nothing left to fit): above
    # the immediate bound (4 eps), below the stall bound (1e3 eps); with every step
    # rejected the fit stalls and is judged at the loose bound
    x = np.linspace(0.0, 1.0, 3000)
    monkeypatch.setattr(lm, "_solve", lambda Hf, lam, Df, gf: -gf)
    model = lambda x, a, b: a * jnp.exp(-b * x)
    y = 3.0 * np.exp(-2.0 * x) * (1 + 1e-14 * np.sin(1e3 * x))
    # nothing is left to fit at this offset-free scale: a residual at the rounding level of
    # data that vary is an exact fit (gtol), not a rounding-limited offset: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        out = lm.exact_lm_fit(model, x, y, [3.0, 2.0])
    assert out["success"] and out["termination"] == "gtol" and out["nfev"] == 1
    # a residual above the loose bound is a real stall
    out = lm.exact_lm_fit(
        model, x, 3.0 * np.exp(-2.0 * x) * (1 + 1e-9 * np.sin(1e3 * x)), [3.0, 2.0]
    )
    assert not out["success"] and out["termination"] == "stalled"


# --- exact-fit rule: per observation, overflow-safe (Codex round 2) -------------------------


def test_a_large_observation_does_not_hide_a_wrong_small_one():
    x = np.array([0.0, 1.0, 1.0])
    y = np.array([1e16, 1.0, 1.0])
    out = lm.exact_lm_fit(
        lambda x, a: jnp.where(x == 0, 1e16, a),
        x,
        y,
        [0.0],
        chunk_size=2,
        compute_covariance=False,
    )
    np.testing.assert_allclose(out["x"], [1.0], rtol=1e-8)
    assert out["success"] and out["nfev"] > 1


def test_data_scale_overflow_does_not_declare_convergence():
    y = np.full(3, 1e155)
    p0 = [1e155 - 1e145]  # 1e-10 relative error, cost 1.5e290: a real misfit
    out = lm.exact_lm_fit(
        lambda x, a: a + 0.0 * x, np.arange(3.0), y, p0, compute_covariance=False
    )
    assert abs(out["x"][0] / 1e155 - 1) < 1e-12  # was 1e-10: stopped at p0
    assert out["nfev"] > 1


def test_nonfinite_relative_residual_never_certifies_exact_fit():
    nan = lm._stationary(
        1.0,
        np.ones(1),
        np.eye(1),
        np.ones(1, bool),
        np.zeros(1),
        -np.inf,
        np.inf,
        0.0,
        0.0,
        bool(np.float64("nan") <= lm.EXACT_FIT_EPS),
    )
    assert nan is None


def test_offset_fit_is_not_certified_at_15_ulp():
    x = np.linspace(0.0, 5.0, 20000)
    y = 1e6 + 1e-6 * np.exp(-5.0 * x)
    out = lm.exact_lm_fit(
        lambda x, a, k: 1e6 + a * jnp.exp(-k * x),
        x,
        y,
        [2e-6, 3.0],
        chunk_size=4096,
        compute_covariance=False,
    )
    np.testing.assert_allclose(out["x"], [1e-6, 5.0], rtol=1e-3)


def test_exact_fit_check_costs_no_extra_data_pass():
    calls = []

    def factory(chunk):
        calls.append(1)
        yield np.array([0.0, 1.0, 2.0]), np.ones(3)

    out = lm.exact_lm_fit(
        lambda x, a: a + 0.0 * x,
        factory,
        None,
        [np.nextafter(1.0, 0.0)],  # one ulp off: an exact fit at the rounding level
        n_points=3,
        chunk_size=3,
        max_nfev=1,
        compute_covariance=False,
    )
    assert out["success"] and out["termination"] == "gtol"
    assert out["nfev"] == len(calls) == 1


@pytest.mark.parametrize(
    ("y", "f", "mask", "want"),
    [
        ([np.nan], [1.0], [True], "nan"),
        ([1.0], [np.nan], [True], "nan"),
        ([1.0], [np.inf], [True], "nan"),
        ([np.inf], [np.inf], [True], "nan"),
        ([0.0], [0.0], [True], 0.0),
        ([np.nan, 2.0], [np.nan, 2.0], [False, True], 0.0),  # padded row ignored
        ([2.0, 1.0], [1.0, 1.0], [True, True], 0.5),
    ],
)
def test_max_rel_propagates_nonfinite_real_rows(y, f, mask, want):
    from nlsq.streaming.normal_equations import _max_rel

    got = float(_max_rel(jnp.array(y), jnp.array(f), jnp.array(mask)))
    assert np.isnan(got) if want == "nan" else got == want


# --- swamped xtol, ftol polish, resolved-signal floor (Claude round 4) --------------------


def test_swamped_xtol_does_not_accept_a_wrong_small_parameter():
    out = lm.exact_lm_fit(
        lambda x, a, b: jnp.where(x == 0, a, b),
        np.array([0.0, 1.0, 2.0]),
        np.array([1e16, 1.0, 1.0]),
        [1e16, 2.0],
        compute_covariance=False,
    )
    assert out["success"]
    np.testing.assert_allclose(out["x"], [1e16, 1.0], rtol=1e-12)


def _tight_scipy(x, y, p0, sigma=None, bounds=(-np.inf, np.inf)):
    from scipy.optimize import curve_fit

    def f(x, a, b, c):
        return a * np.exp(-b * x) + c

    bounded = np.isfinite(bounds[0]).any()
    kw = (
        {"method": "trf", "max_nfev": 2000}
        if bounded
        else {"method": "lm", "maxfev": 20000}
    )
    return (
        curve_fit(
            f,
            x,
            y,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            **kw,
        )[0]
        if bounded
        else curve_fit(
            f, x, y, p0=p0, sigma=sigma, ftol=1e-15, xtol=1e-15, gtol=1e-15, **kw
        )[0]
    )


@pytest.mark.parametrize("case", ["noisy", "weighted", "bounded"])
def test_default_tolerance_popt_matches_the_tight_optimum(case):
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 4.0, 1000)
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.01 * rng.standard_normal(x.size)
    sigma = 0.01 * (1.0 + rng.random(x.size)) if case == "weighted" else None
    bounds = (np.array([0.0, 0.0, 0.0]), np.array([10.0, 1.2, 10.0]))
    kw = {"bounds": bounds} if case == "bounded" else {}
    p0 = [1.0, 1.0, 0.0]
    out = lm.exact_lm_fit(
        lambda x, a, b, c: a * jnp.exp(-b * x) + c,
        x,
        y,
        p0,
        sigma=sigma,
        chunk_size=256,
        compute_covariance=False,
        **kw,
    )
    ref = _tight_scipy(
        x, y, p0, sigma, bounds if case == "bounded" else (-np.inf, np.inf)
    )
    np.testing.assert_allclose(out["x"], ref, rtol=1e-8)


def test_resolved_signal_floor_is_documented():
    assert "4 ulp" in (lm.exact_lm_fit.__doc__ or "")


# --- xtol bias with a dominant offset parameter; repeated polish (Claude round 5) ----------


# n=1e6 is a documented limit: the remaining 0.27 sigma is below the rounding of the cost
@pytest.mark.parametrize("n", [20_000, 200_000, 400_000, 4_000_000])
def test_dominant_offset_xtol_bias_is_far_below_one_sigma(n):
    x = np.linspace(0.0, 10.0, n)
    y = 1e6 + 1e-3 * np.sin(x) + 1e-6 * np.random.default_rng(0).standard_normal(n)
    s = np.sin(x)
    ref_b = np.linalg.lstsq(np.stack([np.ones(n), s], 1), y - 1e6, rcond=None)[0][1]
    sigma_b = 1e-6 / np.sqrt(np.sum((s - s.mean()) ** 2))
    out = lm.exact_lm_fit(
        lambda x, a, b: a + b * jnp.sin(x),
        x,
        y,
        [1e6, 0.0],
        compute_covariance=False,
    )
    assert out["success"]
    assert abs(out["x"][1] - ref_b) < 0.1 * sigma_b


@pytest.mark.parametrize("rel_noise", [1e-10, 1e-12, 1e-13, 3e-15])
def test_near_exact_fits_are_not_vetoed_forever(rel_noise):
    x = np.linspace(0.0, 1.0, 20_000)
    f = 2.0 * np.exp(-3.0 * x) + 0.5
    y = f * (1 + rel_noise * np.random.default_rng(1).standard_normal(x.size))
    out = lm.exact_lm_fit(
        lambda x, a, k, c: a * jnp.exp(-k * x) + c,
        x,
        y,
        [2.2, 2.7, 0.4],
        compute_covariance=False,
    )
    assert out["success"], (out["termination"], out["nfev"])
    np.testing.assert_allclose(
        out["x"], [2.0, 3.0, 0.5], rtol=max(1e-3, 1e3 * rel_noise)
    )


def test_bounded_fit_with_a_held_variable_matches_trf():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 4.0, 1000)
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.01 * rng.standard_normal(x.size)
    bounds = (np.array([0.0, 0.0, 0.6]), np.array([10.0, 10.0, 10.0]))
    out = lm.exact_lm_fit(
        lambda x, a, b, c: a * jnp.exp(-b * x) + c,
        x,
        y,
        [1.0, 1.0, 1.0],
        bounds=bounds,
        chunk_size=256,
        compute_covariance=False,
    )
    ref = _tight_scipy(x, y, [1.0, 1.0, 1.0], None, bounds)
    np.testing.assert_allclose(out["x"], ref, rtol=1e-8)


def test_polish_skip_at_max_nfev_is_documented():
    assert "max_nfev" in lm.exact_lm_fit.__doc__ and "polish" in lm.exact_lm_fit.__doc__


@pytest.mark.serial
@pytest.mark.parametrize("n", [20_000, 1_000_000])
@pytest.mark.parametrize("rel_noise", [1e-10, 1e-12, 3e-15, 0.0])
@pytest.mark.parametrize("p0", [[2.2, 2.7, 0.4], [2.0, 3.0, 0.5]])
def test_healthy_near_exact_fits_do_not_warn(n, rel_noise, p0):
    # noise within a few dozen ulp of |y| is the float64 limit of data that vary, not a
    # dominant offset: no OptimizeWarning (users running with -W error) and a normal label
    x = np.linspace(0.0, 1.0, n)
    f = 2.0 * np.exp(-3.0 * x) + 0.5
    y = f * (1 + rel_noise * np.random.default_rng(1).standard_normal(n))
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        out = lm.exact_lm_fit(
            lambda x, a, k, c: a * jnp.exp(-k * x) + c,
            x,
            y,
            p0,
            compute_covariance=False,
        )
    assert out["success"] and out["termination"] in ("gtol", "ftol", "xtol")
    assert "rounding" not in out["message"]


def test_rounding_limited_offset_fit_is_labelled_honestly():
    from nlsq.result.optimize_warning import OptimizeWarning

    n = 1_000_000  # offset/noise = 1e15: beyond what the cost can resolve
    x = np.linspace(0.0, 5.0, n)
    y = 1e9 + 1e-3 * np.sin(x) + 1e-6 * np.random.default_rng(0).standard_normal(n)
    with pytest.warns(OptimizeWarning, match="floating-point rounding"):
        out = lm.exact_lm_fit(
            lambda x, a, b: a + b * jnp.sin(x),
            x,
            y,
            [1e9, 0.0],
            compute_covariance=False,
        )
    assert out["success"] and out["termination"] != "gtol"
    assert "gradient below gtol" not in out["message"]
    assert "floating-point rounding" in out["message"]
