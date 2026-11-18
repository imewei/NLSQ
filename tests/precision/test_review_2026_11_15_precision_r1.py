"""Regression tests for the 2026-11-15 nlsq/precision review (round 1)."""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.precision.algorithm_selector import AlgorithmSelector
from nlsq.precision.bound_inference import (
    analyze_bounds_quality,
    infer_bounds,
    infer_bounds_for_multistart,
    merge_bounds,
)
from nlsq.precision.parameter_estimation import (
    detect_function_pattern,
    estimate_initial_parameters,
    estimate_p0_for_pattern,
)
from nlsq.precision.parameter_normalizer import ParameterNormalizer


def _lin(x, a):
    return a * x


# ---- bound_inference ------------------------------------------------------


class TestScalarP0:
    def test_curve_fit_auto_bounds_scalar_p0(self):
        x = np.linspace(0, 1, 20)
        popt, _ = curve_fit(_lin, x, 3 * x, p0=2.0, auto_bounds=True)
        np.testing.assert_allclose(popt, [3.0], rtol=1e-6)

    def test_infer_bounds_scalar_p0(self):
        x = np.linspace(0, 1, 20)
        lo, hi = infer_bounds(x, 3 * x, 2.0)
        assert lo.shape == hi.shape == (1,)
        assert lo[0] < 2.0 < hi[0]

    def test_multistart_scalar_p0(self):
        x = np.linspace(0, 1, 20)
        lo, hi = infer_bounds_for_multistart(x, 3 * x, 2.0)
        assert lo.shape == hi.shape == (1,)
        assert np.isfinite(lo).all() and np.isfinite(hi).all()


class TestMergeBoundsValidation:
    @pytest.mark.parametrize(
        "user",
        [
            ([np.nan], [8.0]),
            ([0.0], [np.nan]),
            ([np.inf], [8.0]),
            ([0.0], [-np.inf]),
        ],
    )
    def test_invalid_user_bounds_rejected(self, user):
        inferred = (np.array([0.0]), np.array([10.0]))
        with pytest.raises(ValueError, match="bounds"):
            merge_bounds(inferred, user)

    def test_curve_fit_rejects_nan_bound_with_auto_bounds(self):
        x = np.linspace(0, 1, 20)
        with pytest.raises(ValueError):
            curve_fit(
                _lin, x, 3 * x, p0=[1.0], bounds=([np.nan], [10.0]), auto_bounds=True
            )

    def test_valid_partial_inf_still_merges(self):
        inferred = (np.array([0.0, 0.0]), np.array([10.0, 5.0]))
        lo, hi = merge_bounds(inferred, ([1.0, -np.inf], [np.inf, 4.0]))
        np.testing.assert_array_equal(lo, [1.0, 0.0])
        np.testing.assert_array_equal(hi, [10.0, 4.0])


class TestIntegerOverflow:
    def test_infer_bounds_int16_matches_float(self):
        xi = np.array([-30000, 30000], dtype=np.int16)
        lo_i, hi_i = infer_bounds(xi, xi, [1.0, 1.0])
        lo_f, hi_f = infer_bounds(xi.astype(float), xi.astype(float), [1.0, 1.0])
        np.testing.assert_array_equal(lo_i, lo_f)
        np.testing.assert_array_equal(hi_i, hi_f)

    def test_estimate_initial_parameters_int16_matches_float(self):
        xi = np.array([-30000, 30000], dtype=np.int16)

        def f(x, a, b):
            return a * x + b

        p_i = estimate_initial_parameters(f, xi, xi)
        p_f = estimate_initial_parameters(f, xi.astype(float), xi.astype(float))
        np.testing.assert_allclose(p_i, p_f)

    def test_selector_int16_ranges(self):
        xi = np.array([-30000, 30000] * 10, dtype=np.int16)
        an = AlgorithmSelector().analyze_problem(_lin, xi, xi, p0=np.array([1.0]))
        assert an["x_range"] == 60000.0
        assert an["y_range"] == 60000.0


class TestMultistartFiniteness:
    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_p0_rejected(self, bad):
        with pytest.raises(ValueError, match="p0"):
            infer_bounds_for_multistart([0, 1], [0, 1], [bad])

    def test_huge_finite_one_sided_bound_ordered_or_error(self):
        try:
            lo, hi = infer_bounds_for_multistart(
                [0.0, 1.0], [0.0, 1.0], [1.0], user_bounds=([1e308], [np.inf])
            )
        except ValueError:
            return
        assert np.all(lo >= 1e308)
        assert np.all(lo < hi) and np.isfinite(hi).all()

    def test_crossed_finite_result_rejected(self):
        with pytest.raises(ValueError, match="lower"):
            infer_bounds_for_multistart(
                [0.0, 1.0], [0.0, 1.0], [1.0], user_bounds=([5.0], [2.0])
            )


class TestBoundsQuality:
    def test_symmetric_wide_bounds_not_tight(self):
        a = analyze_bounds_quality((np.array([-1000.0]), np.array([1000.0])), [1.0])
        assert a["tight_parameters"] == []
        assert a["loose_parameters"] == [0]

    def test_narrow_negative_bounds_are_tight(self):
        a = analyze_bounds_quality((np.array([-1.1]), np.array([-0.9])), [-1.0])
        assert a["tight_parameters"] == [0]


# ---- parameter_normalizer -------------------------------------------------


class TestNormalizerScales:
    def test_tiny_p0_is_scaled(self):
        n = ParameterNormalizer([1e-18], strategy="p0")
        np.testing.assert_allclose(n.normalize(jnp.array([1e-18])), [1.0])

    def test_tiny_bounds_range_is_unit_width(self):
        n = ParameterNormalizer([2e-20], bounds=([1e-20], [3e-20]), strategy="bounds")
        lb, ub = n.transform_bounds()
        np.testing.assert_allclose(lb, [0.0], atol=1e-12)
        np.testing.assert_allclose(ub, [1.0])

    def test_zero_p0_still_gets_unit_scale(self):
        n = ParameterNormalizer([0.0, 5.0], strategy="p0")
        np.testing.assert_allclose(n.scales, [1.0, 5.0])

    def test_degenerate_bounds_keep_unit_scale(self):
        n = ParameterNormalizer([1.0], bounds=([1.0], [1.0]), strategy="bounds")
        np.testing.assert_allclose(n.scales, [1.0])


class TestNormalizerShapes:
    def test_mismatched_bounds_rejected(self):
        with pytest.raises(ValueError, match="bounds"):
            ParameterNormalizer([1.0], bounds=([0.0, 0.0], [2.0, 4.0]))

    def test_nan_bounds_rejected(self):
        with pytest.raises(ValueError, match="bounds"):
            ParameterNormalizer([1.0], bounds=([np.nan], [2.0]), strategy="bounds")

    def test_2d_p0_rejected(self):
        with pytest.raises(ValueError, match="p0"):
            ParameterNormalizer([[1.0, 2.0]], strategy="p0")

    def test_scalar_bounds_broadcast_offsets(self):
        n = ParameterNormalizer([1.0, 2.0], bounds=(0.0, 10.0), strategy="bounds")
        assert n.offsets.shape == n.scales.shape == (2,)
        assert n.normalize(jnp.array([1.0, 2.0])).shape == (2,)


# ---- algorithm_selector ---------------------------------------------------


class TestSelectorAnalysis:
    @staticmethod
    def _analyze(p0):
        x = np.linspace(0, 1, 50)
        return AlgorithmSelector().analyze_problem(
            _lin, x, x, p0=np.asarray(p0, dtype=float)
        )

    def test_zero_p0_does_not_inflate_scale_range(self):
        assert self._analyze([1.0, 1.0, 0.0])["param_scale_range"] == 0.0

    def test_tiny_p0_spread_is_seen(self):
        assert self._analyze([1e-11, 1e-16])["param_scale_range"] == pytest.approx(5.0)

    def test_all_zero_p0(self):
        assert self._analyze([0.0, 0.0])["param_scale_range"] == 0.0

    def test_descending_nonuniform_x_not_uniform(self):
        x = np.array([10.0, 4.0, 3.0, 2.0, 0.0])
        an = AlgorithmSelector().analyze_problem(_lin, x, x, p0=np.array([1.0]))
        assert not an["x_uniform"]

    def test_descending_uniform_x_is_uniform(self):
        x = np.linspace(10, 0, 50)
        an = AlgorithmSelector().analyze_problem(_lin, x, x, p0=np.array([1.0]))
        assert an["x_uniform"]

    def test_lsmr_for_many_params_mid_size(self):
        rec: dict = {}
        AlgorithmSelector()._select_trust_region_solver(rec, 500_000, 200)
        assert rec["tr_solver"] == "lsmr"
        rec = {}
        AlgorithmSelector()._select_trust_region_solver(rec, 500_000, 5)
        assert "tr_solver" not in rec


# ---- parameter_estimation -------------------------------------------------


class TestPatternEstimation:
    def test_sigmoid_inv_is_estimated(self):
        x = np.linspace(-5, 5, 101)
        y = 1 / (1 + np.exp(2 * x))
        assert detect_function_pattern(y, x) == "sigmoid_inv"
        L, x0, k, b = estimate_p0_for_pattern("sigmoid_inv", x, y, 4)
        assert pytest.approx(1.0, abs=1e-2) == L
        assert x0 == pytest.approx(0.0, abs=0.2)
        assert k < 0  # model L/(1+exp(-k(x-x0)))+b decays for k<0
        assert b == pytest.approx(0.0, abs=1e-2)

    def test_gaussian_sigma_independent_of_x_order(self):
        x = np.linspace(-5, 5, 101)
        y = np.exp(-(x**2) / 2)
        asc = estimate_p0_for_pattern("gaussian", x, y, 3)
        desc = estimate_p0_for_pattern("gaussian", x[::-1], y[::-1], 3)
        assert desc[2] > 0
        np.testing.assert_allclose(desc, asc)

    def test_detection_independent_of_x_order(self):
        x = np.linspace(-5, 5, 101)
        y = 1 / (1 + np.exp(-2 * x))
        assert detect_function_pattern(y[::-1], x[::-1]) == detect_function_pattern(
            y, x
        )

    def test_exponential_decay_docstring_values(self):
        x = np.linspace(0, 5, 50)
        y = 3 * np.exp(-0.5 * x) + 1
        p0 = estimate_p0_for_pattern("exponential_decay", x, y, 3)
        # Documented in the docstring: [amplitude, rate, offset]
        np.testing.assert_allclose(p0, [2.7537, 0.5225, 1.2463], atol=1e-3)


class TestUserEstimatorFailure:
    def test_failing_estimator_is_logged(self, caplog):
        class Model:
            def __call__(self, x, a, b):
                return a * x + b

            def estimate_p0(self, x, y):
                raise RuntimeError("boom")

        x = np.linspace(0, 1, 10)
        with caplog.at_level(logging.WARNING):
            p0 = estimate_initial_parameters(Model(), x, x)
        assert len(p0) == 2
        assert any("boom" in r.getMessage() for r in caplog.records)


# ---- round 2: zero-p0 hint must keep its 10*eps cutoff --------------------


def _exp_decay(t, a, b, c):
    return a * jnp.exp(-b * t) + c


@pytest.mark.parametrize("c0", [1e-16, 5.5e-17, 0.0])
def test_hybrid_streaming_tiny_offset_p0_still_fits(c0):
    t = np.linspace(0, 5, 400)
    y = 3.0 * np.exp(-0.5 * t) + 1.0
    popt, _ = curve_fit(_exp_decay, t, y, p0=[3.0, 0.5, c0], method="hybrid_streaming")
    np.testing.assert_allclose(popt, [3.0, 0.5, 1.0], rtol=5e-3)


def test_hybrid_streaming_tiny_offset_p0_bounded():
    t = np.linspace(0, 5, 400)
    y = 3.0 * np.exp(-0.5 * t) + 1.0
    popt, _ = curve_fit(
        _exp_decay,
        t,
        y,
        p0=[3.0, 0.5, 1e-16],
        bounds=([0, 0, -5], [10, 5, 5]),
        method="hybrid_streaming",
    )
    np.testing.assert_allclose(popt, [3.0, 0.5, 1.0], rtol=5e-3)


def test_normalizer_hint_applies_below_10eps_only_with_hint():
    hint = jnp.array([np.nan, 7.0])
    n = ParameterNormalizer([2.0, 1e-16], strategy="p0", zero_p0_scales=hint)
    np.testing.assert_allclose(n.scales, [2.0, 7.0])
    # no hint: 1e-18 still normalises
    n = ParameterNormalizer([2.0, 1e-18], strategy="p0")
    np.testing.assert_allclose(n.scales, [2.0, 1e-18])


# ---- round 3: hybrid no-hint paths must not freeze sub-10*eps parameters ---


def _exp3(t, a, b, c):
    return a * jnp.exp(-b * t) + c


@pytest.mark.parametrize("p0", [[1e-16, 1e-16, 1e-16], [1e-16, 0.5, 1.0]])
def test_hybrid_streaming_all_tiny_or_mixed_tiny_p0(p0):
    t = np.linspace(0, 5, 400)
    y = 3.0 * np.exp(-0.5 * t) + 1.0
    popt, _ = curve_fit(_exp3, t, y, p0=p0, method="hybrid_streaming")
    np.testing.assert_allclose(popt, [3.0, 0.5, 1.0], rtol=5e-2)


def test_hybrid_normalization_without_jacobian_probe_keeps_unit_scale():
    from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    opt = AdaptiveHybridStreamingOptimizer(config=HybridStreamingConfig())
    opt._setup_normalization(_exp3, jnp.array([1e-16, 1e-16, 1e-16]), None, None)
    np.testing.assert_allclose(opt.normalizer.scales, [1.0, 1.0, 1.0])


def test_hybrid_streaming_all_tiny_p0_linear_model():
    x = np.linspace(0, 1, 400)
    y = 3.0 * x + 1.0

    def lin(x, a, b):
        return a * x + b

    popt, _ = curve_fit(lin, x, y, p0=[1e-16, 1e-16], method="hybrid_streaming")
    np.testing.assert_allclose(popt, [3.0, 1.0], rtol=1e-2)


@pytest.mark.parametrize(
    "user_bounds",
    [([np.finfo(float).max], [np.inf]), ([-np.inf], [-np.finfo(float).max])],
)
def test_multistart_no_representable_interval_raises(user_bounds):
    with pytest.raises(ValueError, match="lower"):
        infer_bounds_for_multistart(
            [0.0, 1.0], [0.0, 1.0], [1.0], user_bounds=user_bounds
        )
