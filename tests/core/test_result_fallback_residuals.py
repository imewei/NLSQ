"""Regression coverage for results restored without their model callable."""

import numpy as np

from nlsq.result import CurveFitResult


def test_predictions_restore_optimizer_residual_sign():
    """Optimizer fun is predicted minus observed, so predictions add it back."""
    result = CurveFitResult(
        x=np.array([2.0]),
        ydata=np.array([3.0, 5.0, 7.0]),
        fun=np.array([-1.0, -1.0, -1.0]),
    )

    np.testing.assert_array_equal(result.predictions, [2.0, 4.0, 6.0])


def test_residuals_restore_observed_minus_predicted_sign():
    """Reported raw residuals use the opposite sign to the optimizer vector."""
    result = CurveFitResult(
        x=np.array([2.0]),
        ydata=np.array([3.0, 5.0, 7.0]),
        fun=np.array([-1.0, -1.0, -1.0]),
    )

    np.testing.assert_array_equal(result.residuals, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result.predictions + result.residuals, result.ydata)
