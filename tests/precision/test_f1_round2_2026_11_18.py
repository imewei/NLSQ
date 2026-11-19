"""Regression tests for fixer F1, three-brain round 2026-11-18 #2 (nlsq.precision)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.precision.algorithm_selector import auto_select_algorithm
from nlsq.precision.bound_inference import merge_bounds
from nlsq.precision.parameter_estimation import (
    detect_function_pattern,
    estimate_initial_parameters,
)
from nlsq.precision.parameter_normalizer import ParameterNormalizer


@pytest.mark.parametrize("strategy", ["none", "p0", "bounds", "auto"])
def test_inverted_bounds_rejected_for_every_normalization_strategy(strategy):
    """codex P06 HIGH: strategy != 'bounds' accepted an empty box."""
    with pytest.raises(ValueError, match="lower bounds must not exceed"):
        ParameterNormalizer(
            jnp.array([1.0]), (jnp.array([5.0]), jnp.array([1.0])), strategy
        )


def test_normalizer_rejects_wrong_parameter_width():
    """codex P06: a short/long vector broadcast against the scales."""
    n = ParameterNormalizer(jnp.array([1.0, 2.0, 3.0]), None, "p0")
    for bad in (jnp.array([1.0]), jnp.array([1.0, 2.0]), jnp.ones(4)):
        with pytest.raises(ValueError, match="expected 3 parameters"):
            n.normalize(bad)
        with pytest.raises(ValueError, match="expected 3 parameters"):
            n.denormalize(bad)
    np.testing.assert_allclose(
        n.denormalize(n.normalize(jnp.array([1.0, 2.0, 3.0]))), [1.0, 2.0, 3.0]
    )


@pytest.mark.parametrize(
    "bad", [np.zeros((3, 1)), np.zeros((2, 2)), [0.0, 0.0], np.zeros(4)]
)
def test_merge_bounds_rejects_wrong_shape_user_bounds(bad):
    """codex P06: a column/matrix broadcast into a multidimensional box."""
    inferred = (np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="scalar or have 3 elements"):
        merge_bounds(inferred, (bad, np.full(3, 5.0)))


def test_merge_bounds_still_accepts_scalar_and_full_vector():
    lo, hi = merge_bounds((np.zeros(3), np.ones(3)), (1.0, np.array([5.0, 6, 7])))
    np.testing.assert_array_equal(lo, [1, 1, 1])
    np.testing.assert_array_equal(hi, [5, 6, 7])


def test_custom_estimator_receives_only_finite_observations():
    """codex P06: NaN rows bypassed the finite filter for custom estimators."""
    seen = {}

    def model(x, a):
        return a * x

    def est(x, y):
        seen["x"], seen["y"] = np.asarray(x), np.asarray(y)
        return np.array([1.0])

    model.estimate_p0 = est  # type: ignore[attr-defined]
    x = np.array([1.0, 2, np.nan, 4])
    y = np.array([1.0, np.nan, 3, 4])
    estimate_initial_parameters(model, x, y)
    np.testing.assert_array_equal(seen["x"], [1.0, 4.0])
    np.testing.assert_array_equal(seen["y"], [1.0, 4.0])


def test_finite_filter_respects_points_first_multifeature_x():
    """codex P06: the filter assumed points on the last axis."""
    seen = {}

    def model(x, a):
        return a * x[:, 0]

    def est(x, y):
        seen["x"] = np.asarray(x)
        return np.array([1.0])

    model.estimate_p0 = est  # type: ignore[attr-defined]
    x = np.array([[1.0, 2], [np.nan, 3], [4, 5], [6, 7], [8, 9]])  # (5, 2)
    y = np.arange(5.0)
    estimate_initial_parameters(model, x, y)
    assert seen["x"].shape == (4, 2)


def test_pattern_detector_ignores_multifeature_x():
    """codex P06: np.corrcoef on a 2-D x read rows as variables."""
    x = np.random.default_rng(0).normal(size=(2, 50))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_function_pattern(np.linspace(0, 1, 50), x) == "unknown"


def test_estimation_with_multifeature_x_returns_finite_p0():
    def model(x, a, b, c):
        return a * x[0] + b * x[1] + c

    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 60))
    y = 2 * x[0] + x[1] + 3
    p0 = estimate_initial_parameters(model, x, y)
    assert p0.shape == (3,) and np.all(np.isfinite(p0))


def test_algorithm_selection_is_unit_invariant():
    """codex P06: absolute 1e-10 floors made the SNR depend on the units."""
    x = np.linspace(0, 1, 200)
    y = np.sin(6 * x)

    def f(x, a, b):
        return a * x + b

    base = auto_select_algorithm(f, x, y, p0=np.array([1.0, 1.0]))
    for sx, sy in ((1e-12, 1.0), (1.0, 1e-12), (1e6, 1e6)):
        got = auto_select_algorithm(f, x * sx, y * sy, p0=np.array([1.0, 1.0]))
        assert got == base, (sx, sy)


def test_data_analysis_snr_and_zeros_are_unit_invariant():
    from nlsq.precision.algorithm_selector import AlgorithmSelector

    x = np.linspace(0, 1, 200)
    y = 1.0 + np.sin(6 * x)  # no exact zeros
    sel = AlgorithmSelector()
    base = sel._analyze_data(x, y)
    for sy in (1e-12, 1e6):
        got = sel._analyze_data(x, y * sy)
        assert got["snr_estimate"] == pytest.approx(base["snr_estimate"], rel=1e-9)
        assert got["has_zeros"] == base["has_zeros"] is False
