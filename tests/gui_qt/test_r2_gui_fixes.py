"""Regression tests for the round-2 GUI review findings (G2-01..G2-08)."""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _manager(tmp_path, monkeypatch):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.autosave import AutosaveManager

    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", tmp_path / "autosave.json")
    app_state = MagicMock()
    st = app_state.state
    st.xdata = st.ydata = st.sigma = st.model_config = st.p0 = st.bounds = None
    st.auto_p0 = True
    mgr = AutosaveManager.__new__(AutosaveManager)
    mgr._app_state = app_state
    mgr.recovery_available = MagicMock()
    return mgr, st, autosave.AUTOSAVE_FILE


# --- G2-01: corrupt autosave must never raise ---------------------------------


@pytest.mark.serial
@pytest.mark.parametrize("payload", ["[1, 2, 3]", "null", '"str"', "42"])
def test_non_dict_autosave_is_ignored(tmp_path, monkeypatch, payload):
    mgr, _, f = _manager(tmp_path, monkeypatch)
    f.write_text(payload, encoding="utf-8")
    assert mgr.check_recovery() is False
    assert mgr.recover() is False


@pytest.mark.serial
def test_non_dict_model_config_does_not_raise(tmp_path, monkeypatch):
    mgr, st, f = _manager(tmp_path, monkeypatch)
    f.write_text(
        json.dumps({"timestamp": "t", "model_type": "builtin", "model_config": [1]}),
        encoding="utf-8",
    )
    assert mgr.recover() in (True, False)  # must not raise
    assert not isinstance(st.model_config, list)


# --- G2-02: partial data is dropped as a unit, and recovery says so -----------


@pytest.mark.serial
def test_nan_ydata_drops_all_data_and_warns(tmp_path, monkeypatch, caplog):
    mgr, st, f = _manager(tmp_path, monkeypatch)
    f.write_text(
        json.dumps(
            {"timestamp": "t", "xdata": [1.0, 2.0, 3.0], "ydata": [1.0, None, 3.0]}
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        mgr.recover()
    assert st.xdata is None and st.ydata is None
    assert any("data" in r.message.lower() for r in caplog.records)


@pytest.mark.serial
def test_length_mismatch_drops_all_data(tmp_path, monkeypatch, caplog):
    mgr, st, f = _manager(tmp_path, monkeypatch)
    f.write_text(
        json.dumps({"timestamp": "t", "xdata": [1.0, 2.0], "ydata": [1.0, 2.0, 3.0]}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        mgr.recover()
    assert st.xdata is None and st.ydata is None
    assert any("data" in r.message.lower() for r in caplog.records)


@pytest.mark.serial
def test_data_skipped_recorded_and_logged_when_cap_applies(
    tmp_path, monkeypatch, caplog
):
    from nlsq.gui_qt.autosave import AutosaveManager

    mgr, st, _ = _manager(tmp_path, monkeypatch)
    st.xdata = np.zeros(100_001)
    st.ydata = np.zeros(100_001)
    st.sigma = None
    st.model_config = None
    st.p0 = None
    st.bounds = None
    with caplog.at_level(logging.WARNING):
        data = AutosaveManager._serialize_state(mgr)
    assert "xdata" not in data
    assert data["data_skipped"] == 100_001
    assert any("100" in r.message for r in caplog.records)


# --- G2-03: p0 / auto_p0 type validation --------------------------------------


@pytest.mark.serial
def test_bad_p0_and_auto_p0_ignored(tmp_path, monkeypatch):
    mgr, st, _ = _manager(tmp_path, monkeypatch)
    mgr._restore_state({"p0": ["a", "b"], "auto_p0": "yes"})
    assert st.p0 is None
    assert st.auto_p0 is True


@pytest.mark.serial
def test_good_p0_restored(tmp_path, monkeypatch):
    mgr, st, _ = _manager(tmp_path, monkeypatch)
    mgr._restore_state({"p0": [1, 2.5], "auto_p0": False})
    assert st.p0 == [1.0, 2.5]
    assert st.auto_p0 is False


# --- G2-04: downsampling is logged --------------------------------------------


@pytest.mark.serial  # >100K-element arrays
def test_downsample_logs(caplog):
    from nlsq.gui_qt.plots import _downsample

    x = np.arange(120_000, dtype=float)
    with caplog.at_level(logging.INFO, logger="nlsq.gui_qt.plots"):
        xs, _ys = _downsample(x, x, 50_000)
    assert len(xs) < len(x)
    assert any("120000" in r.getMessage() for r in caplog.records)


def test_no_log_without_downsampling(caplog):
    from nlsq.gui_qt.plots import _downsample

    x = np.arange(10, dtype=float)
    with caplog.at_level(logging.INFO, logger="nlsq.gui_qt.plots"):
        _downsample(x, x, 50_000)
    assert not caplog.records


def test_plot_decimation_note():
    from nlsq.gui_qt.plots import decimation_note

    assert decimation_note(10, 50_000) == ""
    note = decimation_note(120_000, 50_000)
    assert "120000" in note and "60000" in note


# --- G2-05 / G2-06: AIC / BIC ---------------------------------------------------


def _stats_page():
    from nlsq.gui_qt.pages.results import ResultsPage

    page = ResultsPage.__new__(ResultsPage)
    page._fit_statistics = MagicMock()
    return page


def _shown(page):
    return page._fit_statistics.set_statistics.call_args.kwargs


def test_aic_bic_nan_when_residuals_nan():
    page = _stats_page()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    page._update_statistics(
        y, np.array([1.0, 1.0]), SimpleNamespace(), np.array([1.1, np.nan, 2.9, 4.0])
    )
    assert np.isnan(_shown(page)["aic"]) and np.isnan(_shown(page)["bic"])


def test_aic_bic_prefer_result_values():
    page = _stats_page()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    res = SimpleNamespace(aic=-12.5, bic=-11.0)
    page._update_statistics(y, np.array([1.0, 1.0]), res, y + 0.1)
    assert _shown(page)["aic"] == -12.5 and _shown(page)["bic"] == -11.0


def test_aic_bic_fallback_matches_library_convention():
    from nlsq.result import CurveFitResult

    page = _stats_page()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    fit = y + np.array([0.1, -0.1, 0.1, -0.1])
    page._update_statistics(y, np.array([1.0, 1.0]), SimpleNamespace(), fit)
    rss, n, k = 0.04, 4, 2
    assert _shown(page)["aic"] == pytest.approx(2 * k + n * np.log(rss / n))
    assert _shown(page)["bic"] == pytest.approx(k * np.log(n) + n * np.log(rss / n))


# --- G2-08: no extrapolation outside the data domain --------------------------


def test_dense_grid_stays_inside_data_domain():
    from nlsq.gui_qt.pages.results import _dense_grid

    g = _dense_grid(np.array([0.1, 5.0, 10.0]))
    assert g[0] == 0.1 and g[-1] == 10.0 and len(g) == 500


# --- G2-07: residual band is built on sorted x ----------------------------------


def test_band_curves_sorted_by_x():
    from nlsq.gui_qt.plots.residuals_plot import _sorted_band

    x = np.array([3.0, 1.0, 2.0])
    ci = np.array([30.0, 10.0, 20.0])
    xs, lo, hi = _sorted_band(x, ci, 50_000)
    assert xs.tolist() == [1.0, 2.0, 3.0]
    assert hi.tolist() == [10.0, 20.0, 30.0]
    assert lo.tolist() == [-10.0, -20.0, -30.0]
