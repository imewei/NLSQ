"""Malformed scalar inputs should produce validation errors, not crashes."""

import numpy as np

from nlsq.utils.validators import InputValidator


def test_scalar_initial_point_reports_validation_error():
    errors, _, _ = InputValidator().validate_least_squares_inputs(lambda p: p, 1.0)
    assert any("1-dimensional" in error for error in errors)


def test_scalar_observation_reports_validation_error():
    errors, _, _, _ = InputValidator().validate_curve_fit_inputs(
        lambda x, a: a * x, np.arange(3.0), 1.0, p0=[1.0]
    )
    assert any("1-dimensional" in error for error in errors)
