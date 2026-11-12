import numpy as np

from nlsq.gui_qt.adapters.fit_adapter import _resolve_workflow
from nlsq.utils.logging import NLSQLogger, _get_operation_id


def test_nested_operation_restores_parent_id():
    logger = NLSQLogger("r19_nested")
    with logger.operation("outer") as outer:
        with logger.operation("inner"):
            pass
        assert _get_operation_id() == outer
    assert _get_operation_id() is None


def test_auto_global_needs_all_bounds_finite():
    inf = np.inf
    partial = (np.array([0.0, -inf]), np.array([1.0, inf]))
    notes: list[str] = []
    assert _resolve_workflow("auto_global", partial, notes) == "auto"
    assert notes
    full = (np.array([0.0, -1.0]), np.array([1.0, 1.0]))
    assert _resolve_workflow("auto_global", full, []) == "auto_global"


def test_interleaved_operations_do_not_leave_a_dead_id():
    logger = NLSQLogger("r20_interleave")

    def gen(name):
        with logger.operation(name):
            yield

    a, b = gen("A"), gen("B")
    next(a)
    next(b)
    list(a)
    list(b)
    assert _get_operation_id() is None


def test_polynomial_validates_degree():
    import pytest

    from nlsq.core.functions import polynomial

    for bad in (-1, 2.5, True):
        with pytest.raises(ValueError):
            polynomial(bad)
    assert polynomial(2.0) is not None


def test_gaussian_and_lorentzian_p0_handle_dips():
    from nlsq.core import functions as F

    x = np.linspace(0, 10, 200)
    g = -4 * np.exp(-((x - 5) ** 2) / (2 * 1.2**2))
    p0 = F.estimate_p0_gaussian(x, g)
    assert p0[0] < 0 and abs(p0[1] - 5) < 0.2
    lz = -3 * 0.8**2 / ((x - 4) ** 2 + 0.8**2)
    p0 = F.estimate_p0_lorentzian(x, lz)
    assert p0[0] < 0 and abs(p0[1] - 4) < 0.2


def test_xdata_layout_error_names_expected_shape():
    import pytest

    from nlsq import curve_fit

    X = np.random.default_rng(0).uniform(size=(30, 2))
    y = X[:, 0] - X[:, 1]
    with pytest.raises(ValueError, match=r"n_dims, n_points"):
        curve_fit(lambda X, a, b: a * X[0] + b * X[1], X, y, p0=[0.0, 0.0])


def test_safe_dumps_rejects_lossy_longdouble_scalar():
    import pytest

    from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps

    if np.finfo(np.longdouble).bits <= 64:
        pytest.skip("longdouble is float64 on this platform")
    lossy = np.longdouble(1) + np.finfo(np.longdouble).eps
    with pytest.raises(SafeSerializationError):
        safe_dumps({"x": lossy})
    assert safe_dumps({"x": np.longdouble(1.5)})  # exactly representable


def test_safe_dumps_rejects_lossy_longdouble_dict_key():
    import pytest

    from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps, safe_loads

    if np.finfo(np.longdouble).bits <= 64:
        pytest.skip("longdouble is float64 on this platform")
    lossy = np.longdouble(1) + np.finfo(np.longdouble).eps
    with pytest.raises(SafeSerializationError, match="losing precision"):
        safe_dumps({lossy: "value"})
    assert safe_loads(safe_dumps({np.longdouble(1.5): "value"})) == {1.5: "value"}


def test_dip_fit_keeps_negative_amplitude_with_default_bounds():
    import warnings

    from nlsq import curve_fit
    from nlsq.core import functions as F

    x = np.linspace(0, 10, 200)
    rng = np.random.default_rng(0)
    y = -4 * np.exp(-((x - 5) ** 2) / (2 * 1.2**2)) + 0.02 * rng.standard_normal(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(
            F.gaussian,
            x,
            y,
            p0=F.estimate_p0_gaussian(x, y),
            bounds=F.gaussian.bounds(),
        )
    assert abs(popt[0] + 4) < 0.2


def test_polynomial_rejects_inf_and_longdouble_nan_roundtrips():
    import pytest

    from nlsq.core.functions import polynomial
    from nlsq.utils.safe_serialize import safe_dumps

    with pytest.raises(ValueError):
        polynomial(np.inf)
    assert safe_dumps({"x": np.longdouble("nan")})
