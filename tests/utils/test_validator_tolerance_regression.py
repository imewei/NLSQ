"""NaN cannot satisfy the documented positive tolerance requirement."""

import numpy as np
import pytest

from nlsq.utils.validators import InputValidator


@pytest.mark.parametrize("name", ["ftol", "xtol", "gtol"])
def test_nan_tolerance_returns_validation_error(name):
    errors, _, _ = InputValidator().validate_least_squares_inputs(
        lambda p: p, [1.0], **{name: np.nan}
    )
    assert any(name in error for error in errors)


@pytest.mark.parametrize("value", [1e-8, np.inf])
def test_positive_tolerances_still_accepted(value):
    errors, _, _ = InputValidator().validate_least_squares_inputs(
        lambda p: p, [1.0], ftol=value, xtol=value, gtol=value
    )
    assert not errors
