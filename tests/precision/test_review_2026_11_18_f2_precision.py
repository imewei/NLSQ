"""Regression tests for the 2026-11-18 three-brain review (fixer F2: precision/)."""

import numpy as np
import pytest

from nlsq.precision.parameter_estimation import (
    estimate_initial_parameters,
    estimate_p0_for_pattern,
)


def test_p06_all_nonfinite_observations_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        estimate_initial_parameters(
            lambda x, a, b, c: a * x + b + c,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.full(4, np.nan),
        )


def test_p06_gaussian_width_positive_for_single_observation():
    p0 = estimate_p0_for_pattern("gaussian", [1.0], [2.0], 3)
    assert np.all(np.isfinite(p0)) and p0[2] > 0


@pytest.mark.parametrize("shape", [(2, 50), (50, 2)])
def test_agy_p06_linear_pattern_multidimensional_x_falls_back(shape):
    rng = np.random.default_rng(0)
    p0 = estimate_p0_for_pattern(
        "linear", rng.standard_normal(shape), rng.standard_normal(50), 3
    )
    assert p0.shape == (3,) and np.all(np.isfinite(p0))


def test_wave3_gaussian_width_positive_when_half_max_points_share_x():
    p0 = estimate_p0_for_pattern("gaussian", [0, 1, 1, 2], [0, 1, 1, 0], 3)
    assert p0[2] > 0
