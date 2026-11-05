"""An exported session bundle must rerun with its original measurements."""

import runpy
import zipfile

import numpy as np
import pytest

from nlsq.gui_qt.pages.export import ExportPage
from nlsq.result import CurveFitResult

pytestmark = pytest.mark.serial


def test_large_bundle_script_loads_bundled_data(app_state, qtbot, tmp_path):
    x = np.linspace(1, 5, 21)
    y = 2 * x + 0.5 + 0.001 * np.sin(x)
    sigma = np.linspace(0.1, 0.3, 21)
    app_state.set_data(x, y, sigma=sigma)
    app_state.set_model("builtin", {"name": "linear"})
    app_state.state.p0 = [1.0, 0.0]
    app_state.set_fit_result(
        CurveFitResult(popt=np.array([2.0, 0.5]), pcov=np.eye(2), success=True)
    )
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    bundle = tmp_path / "session.zip"
    page.export_session_bundle(str(bundle))
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(tmp_path / "unpacked")

    result = runpy.run_path(str(tmp_path / "unpacked" / "fit_script.py"))

    np.testing.assert_array_equal(result["xdata"], x)
    np.testing.assert_array_equal(result["ydata"], y)
    np.testing.assert_array_equal(result["sigma"], sigma)
    np.testing.assert_allclose(result["popt"], [2.0, 0.5], atol=0.005)
