"""Tests for nlsq.utils.bounds_sampling.sample_in_bounds."""

import numpy as np

from nlsq.utils.bounds_sampling import sample_in_bounds


def test_finite_box_matches_linear_map():
    u = np.array([[0.0, 1.0], [0.5, 0.25]])
    out = sample_in_bounds(u, [0, -2], [10, 2], [5, 0], [1, 1])
    np.testing.assert_allclose(out, [[0, 2], [5, -1]])


def test_half_infinite_and_unbounded_dims_stay_finite_and_in_bounds():
    u = np.linspace(0, 1, 21)[:, None] * np.ones((1, 3))
    lb = np.array([0.0, -np.inf, -np.inf])
    ub = np.array([np.inf, 5.0, np.inf])
    out = sample_in_bounds(u, lb, ub, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert np.all(np.isfinite(out))
    assert np.all(out >= lb) and np.all(out <= ub)
