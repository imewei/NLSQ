"""Round-27 regression tests: the ten open items."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.utils.diagnostics import ConvergenceMonitor


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _lin(x, a, b):
    return a * x + b


# 1 -----------------------------------------------------------------------
def test_chunked_all_fail_p0_none_has_model_width():
    from nlsq.streaming.large_dataset import _unfitted_params

    def three(x, a, b, c):
        return a + b * x + c * x**2

    p = _unfitted_params(three, None)
    assert p.shape == (3,) and np.all(np.isnan(p))

    def kwonly(x, a, *, scale=1):
        return a * x * scale

    assert _unfitted_params(kwonly, None).shape == (1,)
    kept = np.array([1.0, 2.0, 3.0])
    assert _unfitted_params(three, kept) is kept


# 2 -----------------------------------------------------------------------
def test_prediction_interval_uses_sigma():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    sigma = np.where(x < 5, 0.05, 1.0)
    y = 2 * x + 1 + sigma * rng.standard_normal(x.size)
    r = curve_fit(_lin, x, y, p0=[1.0, 0.0], sigma=sigma)
    pi = np.asarray(r.prediction_interval(np.array([1.0, 9.0])))  # (n, 2)
    w = pi[:, 1] - pi[:, 0]
    assert w[1] > 5 * w[0]  # noisy region has a much wider interval


# 3 -----------------------------------------------------------------------
def test_square_jacobian_warns_about_orientation():
    x = np.array([0.5, 1.0])  # m == n_params == 2
    y = 2 * np.exp(-0.5 * x)

    def jac(x, a, b):  # native (n_params, m)
        return jnp.stack([jnp.exp(-b * x), -a * x * jnp.exp(-b * x)])

    with pytest.warns(UserWarning, match="orientation is ambiguous"):
        curve_fit(_exp, x, y, p0=[1.5, 0.7], jac=jac)


# 4 -----------------------------------------------------------------------
@pytest.mark.parametrize("j", ["2-point", "3-point", "cs"])
def test_scipy_jac_strings_fall_back_to_autodiff(j):
    x = np.linspace(0, 4, 50)
    y = 2 * np.exp(-0.5 * x)
    with pytest.warns(UserWarning, match="automatic differentiation"):
        popt, _ = curve_fit(_exp, x, y, p0=[1.0, 1.0], jac=j)
    np.testing.assert_allclose(popt, [2.0, 0.5], rtol=1e-6)


def test_garbage_jac_string_still_rejected():
    x = np.linspace(0, 4, 50)
    with pytest.raises(ValueError, match="callable"):
        curve_fit(_exp, x, x, p0=[1.0, 1.0], jac="oops")


# 5 -----------------------------------------------------------------------
def test_plot_2d_xdata():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2, 200))
    y = 3 * X[0] + 2 * X[1] ** 2 + 0.01 * rng.standard_normal(200)

    def f(X, a, b):
        return a * X[0] + b * X[1] ** 2

    r = curve_fit(f, X, y, p0=[1.0, 1.0])
    fig, _ = r.plot()
    plt.close(fig)


# 10 ----------------------------------------------------------------------
def _score(costs):
    m = ConvergenceMonitor(window_size=len(costs))
    for c in costs:
        m.update(c, np.array([c], dtype=float))
    return m.detect_oscillation()


@pytest.mark.parametrize(
    "costs",
    [
        [10, 10, 11, 11, 10, 10, 11, 11, 10, 10],
        [1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    ],
)
def test_plateau_interleaved_alternation_is_oscillation(costs):
    osc, score = _score(costs)
    assert osc and score > 0.7


def test_monotone_descent_is_not_oscillation():
    assert not _score([10, 8, 8, 6, 6, 4, 4, 2, 2, 1])[0]


# GUI items 6-9 -------------------------------------------------------------
def test_decimal_comma_semicolon_csv():
    from nlsq.gui_qt.adapters.data_adapter import parse_table_text

    rows, header = parse_table_text("x;y\n1,5;2,5\n2,5;3,5\n3,5;4,5\n")
    assert header == ["x", "y"]
    assert rows == [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]
    rows, _ = parse_table_text("1,5,2\n3,4,5\n")  # plain comma CSV unchanged
    assert rows == [[1.0, 5.0, 2.0], [3.0, 4.0, 5.0]]


def test_robust_preset_runs_multistart_with_semi_infinite_bounds(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from nlsq.gui_qt.session_state import SessionState

    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(fa, "fit", spy)
    x = np.linspace(0, 5, 100)
    s = SessionState(
        xdata=x,
        ydata=3 * np.exp(-0.7 * x),
        model_func=lambda x, a, b: a * jnp.exp(-b * x),
    )
    s.p0 = [1.0, 1.0]
    s.bounds = ([0.0, 0.0], [None, None])
    s.workflow = "auto_global"
    s.n_starts = 4
    with pytest.raises(RuntimeError):
        fa.run_fit(s)
    assert seen["workflow"] == "auto"
    assert seen["multistart"] is True and seen["n_starts"] == 4


@pytest.mark.serial
def test_stale_stats_cleared_and_midfit_set_data_discards_result():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    _ = QApplication.instance() or QApplication([])
    state = AppState()

    page = DataLoadingPage(state)
    page._xdata = np.arange(5.0)
    page._ydata = np.arange(5.0)
    page._validate_and_update_stats()
    assert "5" in page._stats_points.text()
    page._ydata = np.array([0.0, np.nan, 1.0, 2.0, 3.0])
    import nlsq.gui_qt.pages.data_loading as dl

    orig = dl.QMessageBox.warning
    dl.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        page._validate_and_update_stats()
    finally:
        dl.QMessageBox.warning = orig
    assert page._stats_points.text() == "Points: -"

    fit_page = FittingOptionsPage(state)
    fit_page._fit_epoch = state.input_epoch
    state.set_data(np.arange(3.0), np.arange(3.0))  # data replaced mid-fit
    fit_page._on_fit_finished(object())
    assert state.state.fit_result is None


def test_covariance_sigma_and_reset_epoch():
    from nlsq.result.curve_fit_result import CurveFitResult

    r = CurveFitResult({"x": np.array([1.0])})
    r["sigma"] = np.diag([1.0, 4.0, 9.0])
    r["residuals"] = np.zeros(3)
    r["xdata"] = np.arange(3.0)
    r["absolute_sigma"] = True
    v = np.asarray(r._noise_variance_at(np.arange(3.0), 3, 1))
    np.testing.assert_allclose(v, [1.0, 4.0, 9.0])

    pytest.importorskip("PySide6")
    from nlsq.gui_qt.app_state import AppState

    st = AppState()
    e = st.input_epoch
    st.reset()
    assert st.input_epoch == e + 1
