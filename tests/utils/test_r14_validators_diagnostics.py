import numpy as np

from nlsq.utils.diagnostics import ConvergenceMonitor
from nlsq.utils.validators import InputValidator


def test_stagnation_is_scale_free():
    for scale in (1.0, 1e-12, 1e9):
        m = ConvergenceMonitor(window_size=5)
        for i in range(5):
            m.update(scale * 0.5**i, np.array([float(i)]))
        assert m.detect_stagnation()[0] is False
    m = ConvergenceMonitor(window_size=5)
    for i in range(5):
        m.update(1e6 + 1e-3 * i, np.array([float(i)]))
    assert m.detect_stagnation()[0] is True


def test_bool_ydata_and_none_tol():
    v = InputValidator()
    x = np.arange(6.0)
    y = np.array([True, False] * 3)
    v.validate_curve_fit_inputs(lambda x, a: a * x, x, y, p0=[1.0])
    errors, _, x0 = v.validate_least_squares_inputs(
        lambda p: p, np.array([2.0]), bounds=(-np.inf, np.inf), ftol=None
    )
    assert errors == []
    assert x0.shape == (1,)
