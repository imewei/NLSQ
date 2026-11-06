"""Invalid data shapes must stop dependent validation reductions."""

import numpy as np
import pytest

from nlsq.utils.validators import InputValidator


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([], []),
        ([1.0, 2.0], []),
        ([], [1.0, 2.0]),
        ((), []),
        (np.empty((2, 0)), []),
        ([1.0, 2.0], np.empty((2, 0))),
        ([1.0, 2.0], [[1.0], [2.0]]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
    ],
)
@pytest.mark.parametrize("fast_mode", [True, False])
def test_invalid_shapes_return_errors(x, y, fast_mode):
    errors, _, _, _ = InputValidator(fast_mode=fast_mode).validate_curve_fit_inputs(
        lambda x, a: a * x, x, y, p0=[1.0]
    )
    assert errors
