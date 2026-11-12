"""Round-6 regressions: Results page "Export Figure" must pass sigma + statistics."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.serial


def test_export_figure_passes_sigma_and_statistics(
    app_state, qtbot, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    from nlsq import curve_fit
    from nlsq.cli.visualization import FitVisualizer
    from nlsq.gui_qt.pages.results import ResultsPage

    def model(x, a, b):
        import jax.numpy as jnp

        return a * jnp.exp(-b * x)

    x = np.linspace(0.0, 3.0, 40)
    y = 2.0 * np.exp(-1.2 * x) + 0.01 * np.sin(7 * x)
    sigma = np.full_like(x, 0.05)
    app_state.set_data(x, y, sigma)
    app_state.state.model_func = model

    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(curve_fit(model, x, y, p0=[1.0, 1.0], sigma=sigma))

    captured = {}

    def spy(self, result, data, model, config):
        captured.update(result=result, data=data)
        return []

    monkeypatch.setattr(FitVisualizer, "generate", spy)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("publication", True))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "f.png"), "")
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail(a[2]))

    page._on_export_clicked()

    np.testing.assert_array_equal(captured["data"]["sigma"], sigma)
    stats = captured["result"]["statistics"]
    assert 0.9 < stats["r_squared"] <= 1.0
    assert stats["rmse"] > 0
    assert captured["result"]["popt"] is not None
