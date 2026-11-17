"""Regression tests for the 2026-11-16 GUI review (round 2)."""

from __future__ import annotations

import os
import threading
import warnings

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


# --- W1: capture never touches process-global warnings state ------------------


def test_worker_capture_interleaved_with_main_catch_warnings_leaves_state():
    from nlsq.gui_qt.adapters.fit_adapter import (
        capture_fit_diagnostics,
        install_warning_hook,
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        install_warning_hook()
        before = (warnings.showwarning, list(warnings.filters))

        worker_in, main_in, worker_out = (threading.Event() for _ in range(3))

        def worker():
            with capture_fit_diagnostics():
                worker_in.set()
                assert main_in.wait(10)
            worker_out.set()

        t = threading.Thread(target=worker)
        t.start()
        assert worker_in.wait(10)
        with warnings.catch_warnings():  # e.g. pyqtgraph on a live-plot update
            main_in.set()
            assert worker_out.wait(10)
        t.join()

        assert warnings.showwarning is before[0]
        assert warnings.filters == before[1]

        # a warning from any later thread is still reported, not swallowed
        def later():
            warnings.warn("late warning", RuntimeWarning, stacklevel=1)

        t2 = threading.Thread(target=later)
        t2.start()
        t2.join()
    assert any("late warning" in str(w.message) for w in rec)


def test_repeated_optimize_warning_is_captured_every_fit():
    from nlsq.gui_qt.adapters.fit_adapter import capture_fit_diagnostics
    from nlsq.result import OptimizeWarning

    for _ in range(2):  # the default filter would show it only once
        with capture_fit_diagnostics() as msgs:
            warnings.warn("hybrid ignores ftol", OptimizeWarning, stacklevel=1)
        assert any("hybrid ignores ftol" in m for m in msgs)


# --- S1: one note per diagnostic ----------------------------------------------


def test_warning_and_log_twin_give_one_note():
    from nlsq.gui_qt.adapters.fit_adapter import _dedupe_notes

    notes = _dedupe_notes(
        [
            "Covariance could not be estimated reason=insufficient_data",
            "OptimizeWarning: Covariance could not be estimated",
            "Covariance could not be estimated",
        ]
    )
    assert len(notes) == 1


def test_underdetermined_fit_has_one_covariance_note():
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _exp3, _fit_state

    r = fa.run_fit(_fit_state([0.0, 1.0], [2.0, 1.0], _exp3, [1.0, 1.0, 0.0]))
    assert sum("ovariance" in n for n in r["gui_notes"]) == 1


# --- S2: the note reflects the actual interval --------------------------------


@pytest.mark.serial
def test_known_sigma_n_equals_p_is_not_called_undefined(app_state, qtbot):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from nlsq.gui_qt.pages.results import ResultsPage
    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _fit_state, _installed

    s = _fit_state(
        [1.0], [2.0], lambda x, a: a * x, [1.0], sigma=[1.0], absolute_sigma=True
    )
    r = fa.run_fit(s)
    app_state.set_data(s.xdata, s.ydata, sigma=s.sigma)
    app_state.set_model("custom", model_func=s.model_func)
    app_state.state.absolute_sigma = True
    _installed(app_state, app_state.state, r)
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(r)
    ci = page._param_results._table.item(0, 3).text()
    assert "nan" not in ci.lower()
    assert "undefined" not in page._status_label.text()


# --- S3: sigma alone never attaches to the old x/y ----------------------------


def test_sigma_only_recovery_is_dropped(app_state):
    from nlsq.gui_qt.autosave import AutosaveManager

    app_state.set_data(np.arange(3.0), np.arange(3.0))
    AutosaveManager(app_state)._restore_state({"sigma": [1.0, 2.0, 3.0]})
    assert app_state.state.sigma is None


# --- S0: unknown OS scheme decided from the palette ---------------------------


def _unknown_app(lightness):
    from PySide6.QtCore import Qt

    from tests.gui_qt.test_review_2026_11_16_gui_r1 import _FakeApp

    return _FakeApp(Qt.ColorScheme.Unknown, lightness)


def test_unknown_scheme_follows_a_light_palette():
    from nlsq.gui_qt.theme import ThemeManager

    assert ThemeManager(_unknown_app(239)).current_theme.name == "light"


def test_unknown_scheme_follows_a_dark_palette():
    from nlsq.gui_qt.theme import ThemeManager

    assert ThemeManager(_unknown_app(40)).current_theme.name == "dark"


# --- S5: status text stays readable on the light theme ------------------------


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    c = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@pytest.mark.serial
def test_light_theme_status_amber_has_text_contrast(app_state, qtbot):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from nlsq.gui_qt.pages.results import ResultsPage
    from nlsq.gui_qt.theme import LIGHT_THEME
    from tests.gui_qt.test_review_2026_11_16_gui_r1 import (
        _exp3,
        _fit_state,
        _installed,
    )

    s = _fit_state([0.0, 1.0], [2.0, 1.0], _exp3, [1.0, 1.0, 0.0])
    r = fa.run_fit(s)
    app_state.set_data(s.xdata, s.ydata)
    app_state.set_model("custom", model_func=_exp3)
    _installed(app_state, app_state.state, r)
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(r)
    page.set_theme(LIGHT_THEME)  # restyles the status that is already shown

    sheet = page._status_label.styleSheet()
    colour = sheet.split("color:")[1].strip(" ;")
    assert colour.startswith("#") and colour.upper() != "#FF9800"
    bg = "#ebebeb"  # the darkest light-theme surface
    hi, lo = sorted((_luminance(colour), _luminance(bg)), reverse=True)
    assert (hi + 0.05) / (lo + 0.05) >= 4.5


# --- r3: the hook's filter guard is idempotent --------------------------------


def test_hook_filter_guard_holds_after_install():
    import nlsq.gui_qt.adapters.fit_adapter as fa

    with warnings.catch_warnings():
        fa.install_warning_hook()
        assert warnings.filters[:2] == fa._hook_filters  # else re-inserted per fit
        before = list(warnings.filters)
        for _ in range(3):
            with fa.capture_fit_diagnostics():
                pass
        assert warnings.filters == before


# --- Codex r2: recovery needs a complete x/y pair ------------------------------


def _old_data(app_state):
    app_state.set_data(
        np.array([0.0, 1.0, 2.0]), np.array([4.0, 5.0, 6.0]), np.array([9.0, 9.0, 9.0])
    )


def _restore(app_state, record):
    from nlsq.gui_qt.autosave import AutosaveManager

    AutosaveManager(app_state)._restore_state(record)
    return app_state.state


@pytest.mark.parametrize(
    "record", [{"xdata": [10.0, 11.0, 12.0]}, {"ydata": [1.0, 2.0, 3.0]}]
)
def test_half_a_recovered_pair_leaves_existing_data_alone(app_state, record):
    _old_data(app_state)
    s = _restore(app_state, record)
    np.testing.assert_array_equal(s.xdata, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(s.ydata, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(s.sigma, [9.0, 9.0, 9.0])


def test_mismatched_recovered_pair_is_not_mixed_with_old_data(app_state):
    _old_data(app_state)
    s = _restore(app_state, {"xdata": [1.0, 2.0, 3.0], "ydata": [1.0, 2.0]})
    assert s.xdata is None and s.ydata is None and s.sigma is None


def test_complete_recovered_pair_replaces_everything(app_state):
    _old_data(app_state)
    s = _restore(
        app_state, {"xdata": [10.0, 11.0], "ydata": [1.0, 2.0], "sigma": [0.1, 0.2]}
    )
    np.testing.assert_array_equal(s.xdata, [10.0, 11.0])
    np.testing.assert_array_equal(s.ydata, [1.0, 2.0])
    np.testing.assert_array_equal(s.sigma, [0.1, 0.2])


def test_sigma_without_a_pair_is_not_applied(app_state):
    _old_data(app_state)
    s = _restore(app_state, {"xdata": [10.0, 11.0, 12.0], "sigma": [1.0, 2.0, 3.0]})
    np.testing.assert_array_equal(s.sigma, [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(s.xdata, [0.0, 1.0, 2.0])
