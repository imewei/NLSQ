"""Regression tests for round-1 review findings in nlsq/utils."""

import numpy as np

from nlsq.utils.diagnostics import ConvergenceMonitor
from nlsq.utils.validators import InputValidator


def test_oscillation_threshold_decreases_with_sensitivity():
    lo = ConvergenceMonitor(sensitivity=0.5).oscillation_threshold
    mid = ConvergenceMonitor(sensitivity=1.0).oscillation_threshold
    hi = ConvergenceMonitor(sensitivity=2.0).oscillation_threshold
    assert hi < mid < lo <= 1.0
    assert mid == 0.7


def test_scalar_p0_accepted_for_single_parameter_model():
    x = np.linspace(0, 1, 10)
    y = 2.0 * x
    errors, *_ = InputValidator().validate_curve_fit_inputs(
        lambda x, a: a * x, x, y, p0=1.0
    )
    assert not errors, errors
