"""Tests for ExportPage._generate_json() -- NaN/Infinity JSON compliance."""

from __future__ import annotations

import json

import numpy as np
import pytest

# GUI tests crash xdist workers on headless CI (see test_gui_startup.py) --
# run on a single dedicated worker to avoid taking down the shared pool.
pytestmark = pytest.mark.serial


class FakeFitResult:
    def __init__(self, x):
        self.x = np.asarray(x)
        self.pcov = None
        self.success = True
        self.nfev = 1


def _nan_model(x, a):
    return np.full_like(x, np.nan)


class TestGenerateJsonNonFiniteStatistics:
    """_generate_json() must normalize non-finite r_squared/rmse to `null`
    instead of emitting non-standard `NaN`/`Infinity` JSON tokens, which a
    diverged fit (model producing NaN residuals) would otherwise trigger."""

    def test_diverged_fit_emits_null_not_nan_token(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        app_state.set_data(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        app_state.set_model("custom", model_func=_nan_model)
        app_state.set_fit_result(FakeFitResult([1.0]))

        page = ExportPage(app_state)
        qtbot.addWidget(page)

        json_str = page._generate_json()

        assert "NaN" not in json_str
        assert "Infinity" not in json_str
        data = json.loads(json_str)
        assert data["statistics"]["r_squared"] is None
        assert data["statistics"]["rmse"] is None

    def test_normal_fit_emits_finite_statistics(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        xdata = np.array([1.0, 2.0, 3.0, 4.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0])
        app_state.set_data(xdata, ydata)
        app_state.set_model("custom", model_func=lambda x, a: a * x)
        app_state.set_fit_result(FakeFitResult([2.0]))

        page = ExportPage(app_state)
        qtbot.addWidget(page)

        data = json.loads(page._generate_json())

        assert data["statistics"]["r_squared"] == 1.0
        assert data["statistics"]["rmse"] == 0.0


class TestGenerateJsonNonFiniteParameters:
    """_generate_json()'s parameters block must normalize non-finite
    value/uncertainty/ci_lower/ci_upper to `null` too -- NLSQ deliberately
    reports +inf uncertainty for a singular covariance, which the
    statistics block a few lines below already sanitized but the
    parameters block did not."""

    def test_singular_covariance_emits_null_not_infinity_token(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        xdata = np.array([1.0, 2.0, 3.0, 4.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0])
        app_state.set_data(xdata, ydata)
        app_state.set_model("custom", model_func=lambda x, a: a * x)
        result = FakeFitResult([2.0])
        result.pcov = np.array([[np.inf]])
        app_state.set_fit_result(result)

        page = ExportPage(app_state)
        qtbot.addWidget(page)

        json_str = page._generate_json()

        assert "Infinity" not in json_str
        assert "NaN" not in json_str
        data = json.loads(json_str)
        param = next(iter(data["parameters"].values()))
        assert param["uncertainty"] is None
        assert param["ci_lower"] is None
        assert param["ci_upper"] is None
        assert param["value"] == 2.0


class TestGeneratedScriptMatchesGui:
    """GUI2/GUI3: the exported script must reproduce the GUI fit."""

    @staticmethod
    def _page(app_state, qtbot, workflow, bounds=None):
        from nlsq.gui_qt.adapters.fit_adapter import run_fit
        from nlsq.gui_qt.adapters.model_adapter import get_model
        from nlsq.gui_qt.pages.export import ExportPage

        x = np.linspace(-2, 2, 15)
        y = 3 * x**2 - x + 0.5 + 0.01 * np.sin(7 * x)
        app_state.set_data(x, y)
        app_state.set_model(
            "polynomial", {"degree": 2}, get_model("polynomial", {"degree": 2})
        )
        s = app_state.state
        s.workflow = workflow
        s.n_starts = 5
        if bounds:
            s.p0 = [1.0, 1.0, 1.0]
            s.bounds = bounds
            s.auto_p0 = False
        result = run_fit(s.snapshot_for_fit())
        app_state.set_fit_result(result)
        page = ExportPage(app_state)
        qtbot.addWidget(page)
        return page, result

    def test_polynomial_script_matches_gui_coefficient_order(self, app_state, qtbot):
        page, result = self._page(
            app_state, qtbot, "auto", ([0.0, None, None], [None, None, None])
        )
        ns: dict = {}
        exec(compile(page.generate_python_code(), "script", "exec"), ns)
        np.testing.assert_allclose(ns["popt"], np.asarray(result.popt), rtol=1e-6)

    def test_default_global_workflow_without_bounds_runs(self, app_state, qtbot):
        page, result = self._page(app_state, qtbot, "auto_global")
        code = page.generate_python_code()
        assert 'workflow = "auto_global"' not in code
        ns: dict = {}
        exec(compile(code, "script", "exec"), ns)  # no 'requires bounds' ValueError
        np.testing.assert_allclose(ns["popt"], np.asarray(result.popt), rtol=1e-6)


class TestStandaloneScriptData:
    """GUI-3: a saved/pasted script must not depend on a sibling data.csv."""

    @staticmethod
    def _page(app_state, qtbot, n):
        from nlsq.gui_qt.adapters.model_adapter import get_model
        from nlsq.gui_qt.pages.export import ExportPage

        x = np.linspace(-2, 2, n)
        y = 3 * x**2 - x + 0.5
        app_state.set_data(x, y)
        app_state.set_model(
            "polynomial", {"degree": 2}, get_model("polynomial", {"degree": 2})
        )
        app_state.set_fit_result(FakeFitResult([3.0, -1.0, 0.5]))
        page = ExportPage(app_state)
        qtbot.addWidget(page)
        return page

    def test_modest_data_is_inlined_and_runs_without_file(
        self, app_state, qtbot, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)  # empty dir: no data.csv anywhere
        code = self._page(app_state, qtbot, 50).generate_python_code()
        assert "data.csv" not in code
        ns: dict = {}  # no __file__, like pasted code
        exec(compile(code, "script", "exec"), ns)
        np.testing.assert_allclose(ns["popt"], [3.0, -1.0, 0.5], rtol=1e-5)

    def test_bundle_script_still_loads_data_csv(self, app_state, qtbot, tmp_path):
        page = self._page(app_state, qtbot, 50)
        bundle = tmp_path / "b.zip"
        page.export_session_bundle(str(bundle))
        import zipfile

        with zipfile.ZipFile(bundle) as zf:
            script = zf.read("fit_script.py").decode()
            zf.extractall(tmp_path)
        assert "data.csv" in script
        ns: dict = {"__file__": str(tmp_path / "fit_script.py")}
        exec(compile(script, "script", "exec"), ns)
        np.testing.assert_allclose(ns["popt"], [3.0, -1.0, 0.5], rtol=1e-5)

    def test_large_data_keeps_file_loading_with_clear_note(self, app_state, qtbot):
        code = self._page(app_state, qtbot, 6000).generate_python_code()
        assert "np.loadtxt" in code
        assert "data.csv" in code.split("import numpy")[0]  # header note
        assert "next to this script" in code

    def test_export_data_default_name_matches_script(
        self, app_state, qtbot, monkeypatch
    ):
        from PySide6.QtWidgets import QFileDialog

        seen = []
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (seen.append(a[2]), ("", ""))[1]),
        )
        self._page(app_state, qtbot, 6000)._on_export_data_csv()
        assert seen == ["data.csv"]
