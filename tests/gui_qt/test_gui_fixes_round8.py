"""Round-8 GUI regressions: stale custom-model error, OpenGL fallback, launcher
args, absolute_sigma control, preset reset, and export parameter names/AIC/BIC."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.serial


def _model(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


# --------------------------------------------------------------------------- 1
class TestModelSelectionErrors:
    @pytest.fixture
    def page(self, app_state, qtbot, monkeypatch):
        from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

        page = ModelSelectionPage(app_state)
        qtbot.addWidget(page)
        page.errors = []
        monkeypatch.setattr(
            page, "_show_error", lambda title, msg: page.errors.append((title, msg))
        )
        monkeypatch.setattr(
            "nlsq.gui_qt.pages.model_selection.QMessageBox.information",
            lambda *a, **k: None,
        )
        page._custom_radio.click()
        return page

    @staticmethod
    def _apply(page, code):
        page.errors.clear()
        page._code_editor.set_code(code)
        page._on_apply()
        return page.errors[-1] if page.errors else None

    def test_stale_error_not_shown_for_new_code(self, page):
        err = self._apply(page, "import _ctypes\ndef model(x, a):\n    return a*x\n")
        assert "_ctypes" in err[1]

        err = self._apply(page, "def model(x, a, b)\n    return a*x+b\n")
        assert err[0] == "Syntax Error"
        assert "Line 1" in err[1] and "column" in err[1]

        err = self._apply(page, "model = 3\n")
        assert "_ctypes" not in err[1]
        assert "no model function" in err[1].lower()

    def test_fresh_page_syntax_error_is_reported(self, page):
        err = self._apply(page, "def model(x, a, b)\n    return a*x+b\n")
        assert err[0] == "Syntax Error"


# --------------------------------------------------------------------------- 2
class TestOpenGLDefault:
    @pytest.fixture(autouse=True)
    def _restore_pg(self):
        import pyqtgraph as pg

        from nlsq.gui_qt import plots

        saved = pg.getConfigOption("useOpenGL"), plots._pg_state["configured"]
        yield
        pg.setConfigOptions(useOpenGL=saved[0])
        plots._pg_state["configured"] = saved[1]

    @pytest.mark.parametrize("platform", ["offscreen", "minimal", "offscreen:foo"])
    def test_headless_platforms_default_off(self, monkeypatch, platform):
        from nlsq.gui_qt import plots

        monkeypatch.delenv("NLSQ_GUI_USE_OPENGL", raising=False)
        monkeypatch.setenv("QT_QPA_PLATFORM", platform)
        monkeypatch.setattr(sys, "platform", "linux")
        assert plots._opengl_default() == (False, True)

    def test_no_gl_context_defaults_off(self, monkeypatch, qtbot):
        from PySide6.QtGui import QOpenGLContext

        from nlsq.gui_qt import plots

        monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(plots, "_NO_GL_PLATFORMS", frozenset({"xcb-never"}))
        monkeypatch.setattr(QOpenGLContext, "create", lambda self: False)
        assert plots._opengl_default() == (False, True)

    @pytest.mark.parametrize(("env", "expected"), [("1", True), ("0", False)])
    def test_env_override_wins_both_ways(self, monkeypatch, env, expected):
        import pyqtgraph as pg

        from nlsq.gui_qt import plots

        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        monkeypatch.setenv("NLSQ_GUI_USE_OPENGL", env)
        plots._pg_state["configured"] = False
        plots._configure_pyqtgraph()
        assert pg.getConfigOption("useOpenGL") is expected


# --------------------------------------------------------------------------- 3
@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_launcher_flags_exit_before_qt(flag):
    proc = subprocess.run(
        [sys.executable, "-m", "nlsq.gui_qt", flag],
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"},
        check=False,
    )
    assert proc.returncode == 0
    assert ("usage: nlsq-gui" in proc.stdout) or ("nlsq-gui" in proc.stdout)
    assert "Launching" not in proc.stdout


def test_run_desktop_signature_stays_compatible():
    import inspect

    from nlsq.gui_qt import run_desktop

    assert inspect.signature(run_desktop).parameters["argv"].default is None


# --------------------------------------------------------------------------- 4
class TestAbsoluteSigma:
    def test_widget_roundtrip_defaults_false(self, qtbot):
        from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

        w = AdvancedOptionsWidget()
        qtbot.addWidget(w)
        assert w.get_options()["absolute_sigma"] is False
        w.set_options({"absolute_sigma": True})
        assert w.get_options()["absolute_sigma"] is True

    def test_state_to_fitconfig(self, app_state):
        from nlsq.gui_qt.adapters.fit_adapter import create_fit_config_from_state

        assert create_fit_config_from_state(app_state.state).absolute_sigma is False
        app_state.state.absolute_sigma = True
        assert create_fit_config_from_state(app_state.state).absolute_sigma is True

    def test_changes_pcov_with_sigma(self):
        from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit

        rng = np.random.default_rng(0)
        x = np.linspace(0, 3, 60)
        y = 2 * np.exp(-1.2 * x) + rng.normal(0, 0.05, x.size)
        sigma = np.full_like(x, 0.01)  # deliberately wrong -> rescaling matters
        base = {"p0": [1.0, 1.0], "workflow": "auto"}
        rel = execute_fit(x, y, sigma, _model, FitConfig(**base), None)
        ab = execute_fit(
            x, y, sigma, _model, FitConfig(**base, absolute_sigma=True), None
        )
        assert np.sqrt(ab.pcov[0, 0]) < np.sqrt(rel.pcov[0, 0]) / 2

    def test_yaml_and_autosave_roundtrip(self, app_state, qtbot):
        from nlsq.gui_qt.adapters.config_adapter import (
            export_yaml_config,
            load_yaml_config,
            validate_yaml_config,
        )
        from nlsq.gui_qt.app_state import AppState
        from nlsq.gui_qt.autosave import AutosaveManager

        app_state.state.absolute_sigma = True
        text = export_yaml_config(app_state.state)
        assert validate_yaml_config(text)[0]
        assert load_yaml_config(text).absolute_sigma is True

        data = AutosaveManager(app_state)._serialize_state()
        assert data["settings"]["absolute_sigma"] is True
        fresh = AppState()
        AutosaveManager(fresh)._restore_state(data)
        assert fresh.state.absolute_sigma is True

    def test_script_and_results_note(self, app_state, qtbot):
        from nlsq import curve_fit
        from nlsq.gui_qt.pages.export import ExportPage
        from nlsq.gui_qt.pages.results import ResultsPage

        x = np.linspace(0, 3, 30)
        y = 2 * np.exp(-1.2 * x)
        sigma = np.full_like(x, 0.05)
        app_state.set_data(x, y, sigma)
        app_state.set_model("custom", {"code": "", "function": "_model"}, _model)
        export = ExportPage(app_state)
        qtbot.addWidget(export)
        assert "absolute_sigma" not in export.generate_python_code()
        app_state.state.absolute_sigma = True
        assert "absolute_sigma=True," in export.generate_python_code()

        results = ResultsPage(app_state)
        qtbot.addWidget(results)
        results.update_results(curve_fit(_model, x, y, p0=[1.0, 1.0], sigma=sigma))
        assert "absolute" in results._status_label.text()
        app_state.state.absolute_sigma = False
        results.update_results(curve_fit(_model, x, y, p0=[1.0, 1.0], sigma=sigma))
        assert "relative" in results._status_label.text()


# --------------------------------------------------------------------------- 5
def test_preset_switch_resets_keys_the_new_preset_lacks(app_state, qtbot):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    page._advanced_options.set_options({"absolute_sigma": True})
    page._preset_combo.setCurrentText("Quality")
    assert page._advanced_options.get_options()["n_starts"] == 10
    page._preset_combo.setCurrentText("Fast")
    opts = page._advanced_options.get_options()
    assert opts["n_starts"] == page._default_options["n_starts"]
    assert opts["workflow"] == "auto"
    assert opts["absolute_sigma"] is True  # not a preset property


# --------------------------------------------------------------------------- 6
def test_export_script_names_and_json_aic_bic(app_state, qtbot):
    from nlsq import curve_fit
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 3, 40)
    y = 2 * np.exp(-1.2 * x) + 0.01 * np.sin(7 * x)
    app_state.set_data(x, y)
    app_state.set_model("custom", {"code": "", "function": "_model"}, _model)
    result = curve_fit(_model, x, y, p0=[1.0, 1.0])
    app_state.set_fit_result(result)
    page = ExportPage(app_state)
    qtbot.addWidget(page)

    script = page.generate_python_code()
    assert "param_names = ['a', 'b']" in script
    assert "{name} = " in script and "p{i} = " not in script

    stats = json.loads(page._generate_json())["statistics"]
    assert stats["aic"] == pytest.approx(result.aic)
    assert stats["bic"] == pytest.approx(result.bic)
