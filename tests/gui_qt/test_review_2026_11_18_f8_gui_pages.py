"""Regression tests for the 2026-11-18 three-brain GUI findings (pages/widgets/plots)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.serial


def _quiet_dialogs(monkeypatch):
    """Record error dialogs instead of blocking on them."""
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        DataLoadingPage, "_show_error", lambda self, t, m: shown.append((t, m))
    )
    return shown


def _page(app_state, qtbot):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)
    return page


# --- data_loading.py -------------------------------------------------------


def test_interior_empty_column_keeps_positional_indices(app_state, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    # the default selection lands on the empty column and warns about its NaNs
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    page = _page(app_state, qtbot)
    page._clipboard_text.setPlainText("1,,10,2,3\n2,,20,3,4\n")
    page._on_parse_clipboard()
    page._column_selector.set_selection({"x": 0, "y": 2, "sigma": 3})
    np.testing.assert_array_equal(page._ydata, [10, 20])
    np.testing.assert_array_equal(page._sigma, [2, 3])


def test_trailing_delimiter_column_is_still_dropped(app_state, qtbot):
    page = _page(app_state, qtbot)
    page._clipboard_text.setPlainText("1,2,\n2,4,\n")
    page._on_parse_clipboard()
    assert page._raw_data.shape == (2, 2)


def test_yaml_npz_sigma_key_loaded(app_state, qtbot, tmp_path):
    np.savez(
        tmp_path / "d.npz",
        x=[1.0, 2.0, 3.0],
        y=[10.0, 20.0, 30.0],
        uncertainty=[2.0, 3.0, 4.0],
    )
    cfg = tmp_path / "w.yaml"
    cfg.write_text(
        "data:\n  input_file: d.npz\n  format: npz\n  npz: {sigma_key: uncertainty}\n"
    )
    page = _page(app_state, qtbot)
    page.load_yaml_config(cfg)
    np.testing.assert_array_equal(page._ydata, [10, 20, 30])
    np.testing.assert_array_equal(page._sigma, [2, 3, 4])


def test_yaml_npz_z_key_loads_surface(app_state, qtbot, tmp_path):
    np.savez(
        tmp_path / "s.npz",
        x=[1.0, 2.0, 3.0],
        y=[4.0, 5.0, 6.0],
        z=[7.0, 8.0, 9.0],
        err=[0.1, 0.2, 0.3],
    )
    cfg = tmp_path / "w.yaml"
    cfg.write_text(
        "data:\n  input_file: s.npz\n  format: npz\n  npz: {z_key: z, sigma_key: err}\n"
    )
    page = _page(app_state, qtbot)
    page.load_yaml_config(cfg)
    np.testing.assert_array_equal(page._xdata, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(page._ydata, [7, 8, 9])
    np.testing.assert_array_equal(page._sigma, [0.1, 0.2, 0.3])


def test_failed_yaml_load_does_not_reselect_previous_dataset(
    app_state, qtbot, tmp_path, monkeypatch
):
    shown = _quiet_dialogs(monkeypatch)
    (tmp_path / "old.csv").write_text("x,y,s\n1,10,2\n2,20,3\n")
    page = _page(app_state, qtbot)
    page._load_file(tmp_path / "old.csv")
    cfg = tmp_path / "w.yaml"
    cfg.write_text("data:\n  input_file: missing.csv\n  columns: {x: 0, y: 2}\n")
    page.load_yaml_config(cfg)
    assert shown  # the load error was reported
    np.testing.assert_array_equal(page._ydata, [10, 20])  # not the sigma column
    np.testing.assert_array_equal(page._sigma, [2, 3])


def test_page_follows_sigma_change_with_same_xy(app_state, qtbot):
    page = _page(app_state, qtbot)
    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    app_state.set_data(x, y, np.array([1.0, 2.0]))
    new_sigma = np.array([3.0, 4.0])
    app_state.set_data(x, y, new_sigma)
    page._on_apply = lambda: None  # no modal dialog; inspect staged state only
    np.testing.assert_array_equal(page._sigma, new_sigma)
    np.testing.assert_array_equal(page._raw_data[:, 2], new_sigma)


def test_page_clears_staged_data_when_state_reset(app_state, qtbot):
    page = _page(app_state, qtbot)
    app_state.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert page._apply_btn.isEnabled()
    app_state.reset()
    assert page._xdata is None and page._raw_data is None
    assert not page._apply_btn.isEnabled()


# --- results.py / fit_statistics.py ---------------------------------------


def _fitted_results_page(x2d: bool = False):
    from nlsq.gui_qt.adapters.fit_adapter import run_fit
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.results import ResultsPage
    from nlsq.gui_qt.session_state import SessionState

    rng = np.random.default_rng(0)
    if x2d:
        x = rng.uniform(0, 1, (2, 50))
        y = 2 * x[0] + 3 * x[1] + 0.01 * rng.normal(size=50)
        st = SessionState(
            xdata=x,
            ydata=y,
            model_func=lambda x, a, b: a * x[0] + b * x[1],
            data_mode="2d",
        )
    else:
        x = np.linspace(0, 5, 50)
        y = 2 * x + 1 + 0.05 * rng.normal(size=50)
        st = SessionState(xdata=x, ydata=y, model_func=lambda x, a, b: a * x + b)
        st.absolute_sigma = True
        st.sigma = np.full(50, 0.05)
    aps = AppState()
    aps._state = st
    page = ResultsPage(aps)
    page.update_results(run_fit(st))
    return page


def _export_spy(monkeypatch, page, out: Path, answer):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    from nlsq.cli.visualization import FitVisualizer

    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("publication", True))
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: answer))
    for name in ("information", "critical", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    calls: list[dict] = []

    def fake_generate(self, result, data, model, config):
        calls.append({"result": result, "config": config})
        return []

    monkeypatch.setattr(FitVisualizer, "generate", fake_generate)
    return calls


def test_figure_export_passes_absolute_sigma(qapp, tmp_path, monkeypatch):
    page = _fitted_results_page()
    calls = _export_spy(monkeypatch, page, tmp_path / "fig.png", None)
    page._on_export_clicked()
    assert calls[0]["result"]["absolute_sigma"] is True


def test_figure_export_confirms_actual_destinations(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    page = _fitted_results_page()
    existing = tmp_path / "summary_combined.png"
    existing.write_text("keep me")
    calls = _export_spy(
        monkeypatch, page, tmp_path / "summary.png", QMessageBox.StandardButton.No
    )
    page._on_export_clicked()
    assert calls == []  # declined: nothing written
    assert existing.read_text() == "keep me"

    calls = _export_spy(
        monkeypatch, page, tmp_path / "summary.png", QMessageBox.StandardButton.Yes
    )
    page._on_export_clicked()
    assert len(calls) == 1


def test_figure_export_disabled_for_surface_fit(qapp):
    page = _fitted_results_page(x2d=True)
    assert page.has_results()
    assert not page._export_btn.isEnabled()


def test_statistics_card_is_labelled_evaluations(qapp):
    from nlsq.gui_qt.widgets.fit_statistics import FitStatisticsWidget

    w = FitStatisticsWidget()
    assert "valuation" in w._iter_card._title_label.text()


# --- advanced_options.py ---------------------------------------------------


def test_sampler_menu_only_offers_supported_samplers(qapp):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    items = [w._sampler_combo.itemText(i) for i in range(w._sampler_combo.count())]
    assert items == ["lhs", "sobol", "halton"]


def test_sampler_restore_is_case_insensitive(qapp):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    w.set_options({"sampler": "SOBOL"})
    assert w.get_options()["sampler"] == "sobol"


def test_unknown_sampler_restore_is_logged(qapp, caplog):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    with caplog.at_level(logging.WARNING):
        w.set_options({"sampler": "random"})
    assert "random" in caplog.text


def test_scale_factor_is_not_rounded_by_the_widget(qapp):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    w.set_options({"scale_factor": 1.23456789})
    assert w.get_options()["scale_factor"] == 1.23456789
    w._scale_spin.setValue(2.5)  # a user edit wins over the imported value
    assert w.get_options()["scale_factor"] == 2.5


# --- plots ------------------------------------------------------------------


def test_histogram_narrow_range_has_nonzero_width_bin(qapp):
    from nlsq.gui_qt.plots.histogram_plot import _histogram

    lo = 1e16
    counts, edges = _histogram(np.array([lo, np.nextafter(lo, np.inf)]), 30)
    assert counts.sum() == 2
    assert np.all(np.diff(edges) > 0)


def test_histogram_normality_subsampling_is_logged(qapp, caplog):
    from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

    w = HistogramPlotWidget()
    w.set_data(np.random.default_rng(0).normal(size=6000))
    with caplog.at_level(logging.INFO):
        stats = w.get_statistics()
    assert stats["normality_p"] is not None
    assert "5000" in caplog.text


@pytest.mark.parametrize(
    ("residuals", "expected"),
    [([1e-200, 2e-200], [2, 4]), ([-1e160, 1e160], [-1, 1])],
)
def test_standardized_residuals_survive_extreme_scales(qapp, residuals, expected):
    from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

    w = ResidualsPlotWidget()
    r = np.array(residuals)
    w.set_residuals(np.arange(2.0), r)
    np.testing.assert_allclose(w._std_residuals, expected)


# --- pages/__init__.py, fitting_options.py ---------------------------------


def test_fitting_options_not_reachable_before_a_model_is_applied():
    from nlsq.gui_qt.pages import PageState
    from nlsq.gui_qt.session_state import SessionState

    st = SessionState(xdata=np.arange(3.0), ydata=np.arange(3.0))
    assert not PageState.from_session_state(st).can_access("fitting_options")
    st.model_func = lambda x, a: a * x
    assert PageState.from_session_state(st).can_access("fitting_options")


def test_reset_restores_manual_p0_and_bounds(app_state, qtbot):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    pc = page._param_config
    pc.set_param_names(["a", "b"])
    pc._auto_p0_check.setChecked(False)
    pc.set_values([7.0, 8.0], ([6.0, 7.0], [8.0, 9.0]))
    page.reset()
    p0, bounds = pc.get_values()
    assert pc.is_auto_p0() and p0 == [] and bounds is None
    assert app_state.state.p0 is None and app_state.state.bounds is None


def test_loaded_2d_npz_keys_keep_surface_layout(
    app_state, qtbot, tmp_path, monkeypatch
):
    shown = _quiet_dialogs(monkeypatch)
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=20), rng.normal(size=20)
    z = 2.0 * x0 + 3.0 * x1
    np.savez(tmp_path / "surface.npz", x=x0, y=x1, z=z)
    page = _page(app_state, qtbot)
    page._load_file(tmp_path / "surface.npz", {"npz": {"z_key": "z"}})
    assert not shown, shown
    assert page._xdata.shape == (2, 20)
    np.testing.assert_array_equal(page._ydata, z)
    assert page._sigma is None


def test_browse_loaded_1d_npz_with_sigma_selects_it(
    app_state, qtbot, tmp_path, monkeypatch
):
    shown = _quiet_dialogs(monkeypatch)
    np.savez(
        tmp_path / "d.npz", x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], sigma=[1.0, 2.0, 3.0]
    )
    page = _page(app_state, qtbot)
    page._load_file(tmp_path / "d.npz", {"npz": {"sigma_key": "sigma"}})
    assert not shown, shown
    np.testing.assert_array_equal(page._sigma, [1, 2, 3])


def test_scale_factor_user_edit_retires_imported_exact_value(qapp):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    w.set_options({"scale_factor": 1.23456789})
    w._scale_spin.setValue(2.0)
    w._scale_spin.setValue(1.234568)  # re-entering the displayed digits
    assert w.get_options()["scale_factor"] == 1.234568
