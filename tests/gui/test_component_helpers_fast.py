"""Fast unit tests for GUI helper components without Streamlit."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.mark.gui
@pytest.mark.unit
def test_param_config_helpers() -> None:
    module = importlib.import_module("nlsq.gui.components.param_config")

    assert module.validate_bounds(None, 1.0) == (True, "")
    assert module.validate_bounds(2.0, 1.0) == (
        False,
        "Lower bound must be <= upper bound",
    )
    assert module.get_default_p0_value() == 1.0

    config = module.create_param_config_dict(["a", "b"])
    assert set(config.keys()) == {"a", "b"}
    assert config["a"]["transform"] == "none"

    class ModelNoP0:
        pass

    assert module.estimate_p0_for_model(ModelNoP0(), [1], [2]) is None

    class ModelP0:
        def estimate_p0(self, *_a, **_k):
            return (1.0, 2.0)

    assert module.estimate_p0_for_model(ModelP0(), [1], [2]) == [1.0, 2.0]

    class ModelBad:
        def estimate_p0(self, *_a, **_k):
            raise RuntimeError("boom")

    assert module.estimate_p0_for_model(ModelBad(), [1], [2]) is None

    assert module.format_p0_display(None) == "Auto"
    assert module.format_p0_display(1e-4).endswith("e-04")
    assert module.format_p0_display(1.2345) == "1.2345"

    assert module.validate_p0_input(" ") == (True, None, "")
    assert module.validate_p0_input("1.5") == (True, 1.5, "")
    ok, value, msg = module.validate_p0_input("bad")
    assert ok is False
    assert value is None
    assert "Invalid number" in msg


@pytest.mark.gui
@pytest.mark.unit
def test_param_results_formatting_and_ci() -> None:
    module = importlib.import_module("nlsq.gui.components.param_results")

    popt = np.array([1.0, 2.0])
    pcov = np.diag([0.04, 0.01])
    df = module.format_parameter_table(popt, pcov, ["a"])
    assert list(df.columns) == ["Parameter", "Value", "Std Error", "Rel Error (%)"]
    assert df["Parameter"].tolist() == ["a", "p1"]

    df_no_cov = module.format_parameter_table(popt, None)
    assert (df_no_cov["Std Error"] == "N/A").any()

    ci_missing = module.compute_confidence_intervals(popt, None)
    assert ci_missing == [(-np.inf, np.inf), (-np.inf, np.inf)]

    pcov_bad = np.array([[np.inf, 0.0], [0.0, np.inf]])
    ci_bad = module.compute_confidence_intervals(popt, pcov_bad)
    assert ci_bad[0][0] == -np.inf

    pcov_neg = np.array([[-1.0, 0.0], [0.0, 0.01]])
    df_neg = module.format_parameter_table(popt, pcov_neg)
    assert "N/A" in df_neg["Std Error"].tolist()


@pytest.mark.gui
@pytest.mark.unit
def test_iteration_history_helpers() -> None:
    module = importlib.import_module("nlsq.gui.components.iteration_table")

    history = module.create_iteration_history(["a", "b"])
    history = module.update_iteration_history(
        history, 1, np.array([1.0, 2.0]), cost=3.0
    )
    history = module.update_iteration_history(
        history, 2, np.array([1.5, 2.5]), cost=2.0
    )
    assert history["iterations"] == [1, 2]
    assert history["costs"] == [3.0, 2.0]

    limited = module.limit_history_size(history, max_entries=1)
    assert limited["iterations"] == [2]

    df = module.format_iteration_table(history)
    assert list(df.columns) == ["Iteration", "a", "b", "Cost"]

    empty_df = module.format_iteration_table(module.create_iteration_history())
    assert empty_df.empty


@pytest.mark.gui
@pytest.mark.unit
def test_live_cost_plot_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.gui.components.live_cost_plot")

    history = module.create_cost_history()
    history = module.update_cost_history(history, 1, 10.0)
    history = module.update_cost_history(history, 2, 5.0)
    assert history["iterations"] == [1, 2]

    module.clear_cost_history(history)
    assert history["costs"] == []

    monkeypatch.setattr(module, "get_current_theme", lambda: "light")
    monkeypatch.setattr(
        module,
        "get_plotly_colors",
        lambda *_a, **_k: {
            "cost_line": "#000",
            "grid": "#ccc",
            "background": "#fff",
            "text": "#000",
        },
    )
    monkeypatch.setattr(
        module,
        "get_annotation_style",
        lambda *_a, **_k: {"bgcolor": "#fff", "bordercolor": "#000"},
    )
    monkeypatch.setattr(module, "get_plotly_template", lambda *_a, **_k: "plotly")

    fig = module.create_cost_plot_figure({"iterations": [1], "costs": [1.0]})
    assert fig.data

    empty_fig = module.create_empty_cost_plot("light")
    assert empty_fig.layout is not None

    config = module.get_plot_config()
    assert config["displayModeBar"] is False
