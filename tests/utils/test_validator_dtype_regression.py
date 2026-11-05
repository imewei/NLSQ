"""Invalid array dtypes belong in validation errors, not NumPy exceptions."""

import numpy as np
import pytest

from nlsq.utils.validators import InputValidator


@pytest.mark.parametrize(
    "values", [["bad", "data"], [None, None], np.array([object(), object()])]
)
def test_invalid_guess_dtype_returns_errors(values):
    errors, _, _ = InputValidator().validate_least_squares_inputs(lambda p: p, values)
    assert errors


@pytest.mark.parametrize("field", ["x", "y", "coordinate"])
@pytest.mark.parametrize("values", [["bad", "data"], [None, None]])
def test_invalid_curve_data_dtype_returns_errors(field, values):
    x, y = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    if field == "x":
        x = values
    elif field == "y":
        y = values
    else:
        x = (np.array([1.0, 2.0]), values)
    errors, _, _, _ = InputValidator().validate_curve_fit_inputs(
        lambda x, a: a * x, x, y, p0=[1.0]
    )
    assert errors


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int64, np.float32, np.float64, np.complex128]
)
def test_supported_guess_dtype_is_preserved(dtype):
    values = np.array([1, 2], dtype=dtype)
    errors, _, clean = InputValidator().validate_least_squares_inputs(
        lambda p: p, values
    )
    assert not errors
    assert clean.dtype == values.dtype
