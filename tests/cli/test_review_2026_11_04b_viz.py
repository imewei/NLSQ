"""Regression tests for the 2026-11-04 visualization/batch review findings."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from scipy import stats

import nlsq
from nlsq.cli import visualization as V
from nlsq.cli.commands.batch import _warn_output_collisions, run_batch
from nlsq.cli.commands.fit import run_fit
from nlsq.cli.visualization import FitVisualizer


def _lin(x, a):
    return a * x


X = np.linspace(0, 1, 10)


def test_residuals_are_unweighted_data_minus_model():
    # "fun" holds the sigma-weighted (model - y)/sigma; it must not be used.
    y = 2 * X + 0.1
    result = {"popt": [2.0], "fun": ((2 * X - y) / 0.01).tolist()}
    r = FitVisualizer()._get_residuals(result, {"xdata": X, "ydata": y}, _lin)
    np.testing.assert_allclose(r, y - 2 * X)


def test_curve_fit_result_popt_is_drawn(tmp_path):
    import jax.numpy as jnp

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.7 * x) + np.random.default_rng(0).normal(0, 0.01, 50)
    res = nlsq.curve_fit(model, x, y, p0=[1, 1])
    fig = FitVisualizer()._create_combined_figure(
        res, {"xdata": x, "ydata": y}, model, {"visualization": {}}
    )
    try:
        assert len(fig.axes[0].lines) >= 1  # fit curve present
    finally:
        plt.close(fig)


def test_cli_annotation_has_statistics(tmp_path):
    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.7 * x) + np.random.default_rng(0).normal(0, 0.01, 50)
    np.savetxt(
        tmp_path / "d.csv", np.c_[x, y], delimiter=",", header="x,y", comments=""
    )
    cfg = {
        "data": {"input_file": str(tmp_path / "d.csv"), "columns": {"x": 0, "y": 1}},
        "model": {"type": "builtin", "name": "exponential_decay"},
        "export": {"results_file": str(tmp_path / "o.json")},
        "visualization": {
            "enabled": True,
            "output_dir": str(tmp_path / "f"),
            "formats": ["png"],
            "main_plot": {"annotation": {"enabled": True}},
        },
    }
    (tmp_path / "w.yaml").write_text(yaml.safe_dump(cfg))
    seen = {}
    orig = FitVisualizer._add_statistics_annotation

    def spy(self, ax, result, c):
        seen["stats"] = dict(result.get("statistics", {}))
        return orig(self, ax, result, c)

    FitVisualizer._add_statistics_annotation = spy
    try:
        run_fit(str(tmp_path / "w.yaml"))
    finally:
        FitVisualizer._add_statistics_annotation = orig
    assert "r_squared" in seen["stats"]


def test_confidence_band_large_offset_and_t_quantile():
    def m(x, a, b, c):
        return a * np.exp(-b * x) + c

    x = np.linspace(0, 5, 5)
    p = np.array([2.0, 0.5, 1e9])
    pcov = np.diag([1e-2, 1e-4, 1.0])
    lo, hi = FitVisualizer()._calculate_confidence_band(m, x, p, pcov, ndata=6)
    J = np.c_[np.exp(-p[1] * x), -p[0] * x * np.exp(-p[1] * x), np.ones_like(x)]
    sd = np.sqrt(np.einsum("ij,jk,ik->i", J, pcov, J))
    expected = stats.t.ppf(0.975, 3) * sd
    np.testing.assert_allclose((hi - lo) / 2, expected, rtol=1e-3)


def test_failed_save_does_not_leak_figures(tmp_path):
    plt.close("all")
    cfg = {"visualization": {"output_dir": str(tmp_path), "formats": ["bogusfmt"]}}
    for _ in range(3):
        with pytest.raises(ValueError):
            FitVisualizer().generate(
                {"popt": [1.0]}, {"xdata": X, "ydata": X}, _lin, cfg
            )
    assert plt.get_fignums() == []


def test_batch_defaults_figure_prefix_to_workflow_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0, 5, 30)
    paths = []
    for name, amp in (("a", 3.0), ("b", 5.0)):
        np.savetxt(
            f"{name}.csv",
            np.c_[x, amp * np.exp(-0.7 * x)],
            delimiter=",",
            header="x,y",
            comments="",
        )
        cfg = {
            "data": {"input_file": f"{name}.csv", "columns": {"x": 0, "y": 1}},
            "model": {"type": "builtin", "name": "exponential_decay"},
            "export": {"results_file": f"{name}.json"},
            "visualization": {"enabled": True, "formats": ["png"]},
        }
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
        paths.append(f"{name}.yaml")
    run_batch(paths, max_workers=1)
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {
        "a_combined.png",
        "b_combined.png",
    }


def test_output_collision_warns(tmp_path, caplog):
    for n in ("a", "b"):
        (tmp_path / f"{n}.yaml").write_text(
            yaml.safe_dump({"export": {"results_file": str(tmp_path / "same.json")}})
        )
    logger = logging.getLogger("collision-test")
    with caplog.at_level(logging.WARNING, logger="collision-test"):
        _warn_output_collisions(
            [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")], logger
        )
    assert "collision" in caplog.text.lower()


def test_style_applied_during_draw_and_restored(tmp_path):
    before = plt.rcParams["font.size"]
    seen = []
    orig = FitVisualizer._create_combined_figure

    def spy(self, *a, **k):
        seen.append(plt.rcParams["font.size"])
        return orig(self, *a, **k)

    cfg = {
        "visualization": {
            "output_dir": str(tmp_path),
            "formats": ["png"],
            "style": "nature",
        }
    }
    FitVisualizer._create_combined_figure = spy
    try:
        FitVisualizer().generate({"popt": [1.0]}, {"xdata": X, "ydata": X}, _lin, cfg)
    finally:
        FitVisualizer._create_combined_figure = orig
    assert seen == [V.STYLE_PRESETS["nature"]["font_size"]]
    assert plt.rcParams["font.size"] == before


def test_null_labels_and_density_label():
    v = FitVisualizer()
    fig = v._create_combined_figure(
        {"popt": [1.0]},
        {"xdata": X, "ydata": X},
        _lin,
        {"main_plot": {"x_label": None, "y_label": None}},
    )
    fig_h = v._create_histogram_figure(np.random.default_rng(0).normal(size=30), {})
    try:
        assert fig.axes[0].get_xlabel() == "x"
        assert fig.axes[0].get_ylabel() == "y"
        assert fig_h.axes[0].get_ylabel() == "Density"
    finally:
        plt.close(fig)
        plt.close(fig_h)
