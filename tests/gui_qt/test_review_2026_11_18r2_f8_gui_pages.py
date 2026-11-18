"""Regression tests for the 2026-11-18 round-2 GUI findings (pages/widgets/plots)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.serial


# --- plots ------------------------------------------------------------------


def test_histogram_bins_cover_all_data_at_large_offset():
    from nlsq.gui_qt.plots.histogram_plot import _histogram

    data = 1e16 + 2 * np.arange(20)
    counts, edges = _histogram(data, 30)
    assert counts.sum() == data.size
    assert edges[0] <= data.min() and edges[-1] >= data.max()
    # every value lies inside the edges that count it
    assert np.array_equal(np.histogram(data, bins=edges)[0], counts)


def test_histogram_single_bin_fallback_contains_extrema():
    from nlsq.gui_qt.plots.histogram_plot import _histogram

    data = np.array([1e16, 1e16 + 38.0])
    counts, edges = _histogram(data, 1)
    assert counts.sum() == 2 and edges[0] <= data.min() and edges[-1] >= data.max()


def test_population_std_large_offset_and_tiny_spread():
    from nlsq.gui_qt.plots import population_std

    big = np.array([1e16, np.nextafter(1e16, np.inf)])
    assert population_std(big) == pytest.approx(np.std(big - big[0]), rel=1e-12)
    tiny = np.array([1e-170, 2e-170])
    assert population_std(tiny) == pytest.approx(5e-171, rel=1e-12)
    assert population_std(np.array([3.0, 3.0])) == 0.0


def test_residuals_standardization_translates_before_scaling(qtbot):
    from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

    w = ResidualsPlotWidget()
    qtbot.addWidget(w)
    r = np.array([1e16, np.nextafter(1e16, np.inf)])
    w.set_residuals(np.arange(2.0), r)
    # true population std is exactly 1 (spread of 2 ulps = 2.0 -> std 1.0)
    np.testing.assert_allclose(w._std_residuals, r / 1.0, rtol=1e-15)


def test_residuals_fitted_label_without_fitted_values(qtbot):
    from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

    w = ResidualsPlotWidget()
    qtbot.addWidget(w)
    w.set_residuals(np.arange(3.0), np.array([1.0, -1.0, 0.5]))
    assert "unavailable" in w._fitted_label()
    w.set_residuals(np.arange(3.0), np.array([1.0, -1.0, 0.5]), np.arange(3.0))
    assert w._fitted_label() == "Fitted Values"


def test_histogram_statistics_survive_variance_underflow(qtbot):
    from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

    w = HistogramPlotWidget()
    qtbot.addWidget(w)
    w.set_data(np.array([1e-170, 2e-170, 3e-170]))
    stats = w.get_statistics()
    assert stats["std"] > 0
    assert np.isfinite(stats["kurtosis"])
    assert w._normal_curve is not None or not w._show_normal_cb.isChecked()


@pytest.mark.parametrize("which", ["fit", "residuals", "histogram"])
def test_plot_export_raises_when_nothing_written(qtbot, tmp_path, which):
    from nlsq.gui_qt.plots.fit_plot import FitPlotWidget
    from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget
    from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

    w = {
        "fit": FitPlotWidget,
        "residuals": ResidualsPlotWidget,
        "histogram": HistogramPlotWidget,
    }[which]()
    qtbot.addWidget(w)
    with pytest.raises(OSError):
        w.export_image(str(tmp_path / "missing_dir" / "p.png"))


def test_plot_export_svg_never_returns_without_a_file(qtbot, tmp_path):
    from nlsq.gui_qt.plots.fit_plot import FitPlotWidget

    w = FitPlotWidget()
    qtbot.addWidget(w)
    w.set_data(np.arange(5.0), np.arange(5.0))
    out = tmp_path / "p.svg"
    try:
        w.export_image(str(out))
    except (
        Exception
    ):  # pyqtgraph's SVG writer may fail on this Qt build; it must be loud
        return
    assert out.stat().st_size > 0


# --- widgets ----------------------------------------------------------------


def test_pasted_displayed_scale_factor_retires_hidden_exact_value(qtbot):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    w.set_options({"scale_factor": 1.23456789})
    assert w.get_options()["scale_factor"] == 1.23456789
    edit = w._scale_spin.lineEdit()
    edit.selectAll()
    edit.insert("1.234568")  # same displayed value: valueChanged stays silent
    edit.textEdited.emit("1.234568")
    w._scale_spin.interpretText()
    assert w.get_options()["scale_factor"] == 1.234568


def test_absolute_sigma_help_mentions_unit_sigma(qtbot):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    tip = w._abs_sigma_check.toolTip()
    assert "unit sigma" in tip and "Only matters" not in tip


# --- pages ------------------------------------------------------------------


def _page(app_state, qtbot):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)
    return page


def test_explicit_nan_sigma_column_is_kept(app_state, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    page = _page(app_state, qtbot)
    page._clipboard_text.setPlainText("0,1,nan\n1,3,nan\n2,5,nan\n")
    page._on_parse_clipboard()
    assert page._raw_data.shape == (3, 3)


def test_failed_yaml_mapping_does_not_leave_apply_enabled(
    app_state, qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    shown: list[str] = []
    monkeypatch.setattr(
        DataLoadingPage, "_show_error", lambda self, t, m: shown.append(m)
    )
    (tmp_path / "d.csv").write_text(
        "x,y,temperature,sigma\n0,1,5,0.1\n1,3,5,0.1\n2,5,5,0.1\n"
    )
    cfg = tmp_path / "w.yaml"
    cfg.write_text(
        "data:\n  input_file: d.csv\n  columns: {x: x, y: missing, sigma: sigma}\n"
    )
    page = _page(app_state, qtbot)
    page.load_yaml_config(cfg)
    assert shown
    assert not page._apply_btn.isEnabled()


def test_export_figure_falls_back_for_data_length_model(
    app_state, qtbot, tmp_path, monkeypatch
):
    import jax.numpy as jnp
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    from nlsq.gui_qt.pages.results import ResultsPage

    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("publication", True))
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "fig.png"), "")),
    )
    msgs: list[str] = []
    for name in ("information", "critical", "warning"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, _n=name, **k: msgs.append(_n)),
        )

    def model(x, a, b):
        return a * x + b * jnp.arange(4.0)

    x = np.array([0.0, 1.0, 3.0, 4.0])
    popt = np.array([2.0, 3.0])
    y = model(x, *popt)
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    st = app_state.state
    st.xdata, st.ydata, st.sigma, st.model_func = x, np.asarray(y), None, model
    page._last_result = SimpleNamespace(
        popt=popt, x=popt, pcov=np.eye(2) * 1e-4, fun=np.zeros(4)
    )
    page._last_statistics = {}
    from nlsq.cli.visualization import FitVisualizer

    figs = []
    orig = FitVisualizer._create_combined_figure

    def capture(self, *a, **k):
        fig = orig(self, *a, **k)
        figs.append(fig)
        return fig

    monkeypatch.setattr(FitVisualizer, "_create_combined_figure", capture)
    page._on_export_clicked()
    assert msgs == ["information"], msgs
    assert (tmp_path / "fig_combined.png").exists()
    # the exported curve is the on-screen fallback: the model at the observed x
    curves = [
        ln
        for ax in figs[0].axes
        for ln in ax.lines
        if len(ln.get_xdata()) == len(x) and np.allclose(ln.get_xdata(), x)
    ]
    assert any(np.allclose(ln.get_ydata(), [0.0, 5.0, 12.0, 17.0]) for ln in curves)


def test_population_std_does_not_overflow_for_extreme_spread():
    from nlsq.gui_qt.plots import population_std

    assert population_std(np.array([-1e308, 1e308])) == pytest.approx(1e308)


def test_fallback_band_is_centred_on_the_plotted_curve_for_unsorted_x(app_state, qtbot):
    import jax.numpy as jnp

    from nlsq.gui_qt.pages.results import ResultsPage

    def model(x, a, b):
        return a * x + b * jnp.arange(4.0)

    popt = np.array([2.0, 3.0])

    class Result:
        pcov = np.eye(2)

        def confidence_band(self, x):  # delta method stand-in: model(x) +- 1
            y = np.asarray(model(np.asarray(x), *popt))
            return y - 1.0, y + 1.0

    x = np.array([4.0, 0.0, 3.0, 1.0])
    y_fit = np.asarray(model(x, *popt))
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    app_state.state.model_func = model
    page._update_plots(x, y_fit, popt, Result(), y_fit)
    plot = page._fit_plot
    np.testing.assert_array_equal(plot._fit_x, [0.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(plot._fit_y, [3.0, 11.0, 12.0, 8.0])
    np.testing.assert_allclose((plot._conf_lower + plot._conf_upper) / 2, plot._fit_y)


@pytest.mark.parametrize("columns", ["", "[x, y]", "3", "text"])
def test_non_mapping_yaml_columns_leave_state_untouched(
    app_state, qtbot, tmp_path, monkeypatch, columns
):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    shown: list[str] = []
    monkeypatch.setattr(
        DataLoadingPage, "_show_error", lambda self, t, m: shown.append(m)
    )
    (tmp_path / "d.csv").write_text("x,y,temp,sigma\n0,1,5,0.1\n1,3,2,0.1\n")
    cfg = tmp_path / "w.yaml"
    cfg.write_text(f"data:\n  input_file: d.csv\n  columns: {columns}\n")
    page = _page(app_state, qtbot)
    page._clipboard_text.setPlainText("7,8\n9,10\n")
    page._on_parse_clipboard()
    before = page._raw_data.copy()
    apply_before = page._apply_btn.isEnabled()
    page.load_yaml_config(cfg)
    assert shown and "mapping" in shown[-1]
    np.testing.assert_array_equal(page._raw_data, before)
    assert page._apply_btn.isEnabled() == apply_before


def test_residuals_band_is_rowwise_with_duplicate_x_in_fallback(app_state, qtbot):
    import jax.numpy as jnp

    from nlsq.gui_qt.pages.results import ResultsPage

    def model(x, a, b):
        return a * x + b * jnp.arange(4.0)

    popt = np.array([2.0, 3.0])
    half = np.array([0.64, 0.73, 1.88, 1.80])  # distinct per row

    class Result:
        pcov = np.eye(2)

        def confidence_band(self, x):
            y = np.asarray(model(np.asarray(x), *popt))
            return y - half, y + half

    x = np.array([1.0, 0.0, 4.0, 1.0])  # duplicate x, model tied to row order
    y_fit = np.asarray(model(x, *popt))
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    app_state.state.model_func = model
    page._update_plots(x, y_fit, popt, Result(), y_fit)
    np.testing.assert_allclose(page._residuals_plot._confidence_interval, half)


def test_residuals_band_interpolated_for_normal_dense_grid(app_state, qtbot):
    from nlsq.gui_qt.pages.results import ResultsPage

    def model(x, a, b):
        return a * x + b

    popt = np.array([2.0, 1.0])

    class Result:
        pcov = np.eye(2)

        def confidence_band(self, x):
            x = np.asarray(x, dtype=float)
            return a_line(x) - 0.5, a_line(x) + 0.5

    def a_line(x):
        return 2.0 * x + 1.0

    x = np.linspace(0.0, 4.0, 9)
    y_fit = model(x, *popt)
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    app_state.state.model_func = model
    page._update_plots(x, y_fit, popt, Result(), y_fit)
    np.testing.assert_allclose(page._residuals_plot._confidence_interval, 0.5)
