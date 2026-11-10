"""Round-3 CLI regressions: bare-YAML-key (None) handling (max_nfev non-finite lives in test_review_r2_cli_fixes)."""

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import ConfigError
from nlsq.cli.result_exporter import ResultExporter
from nlsq.cli.visualization import FitVisualizer
from nlsq.cli.workflow_runner import WorkflowRunner


def _model(x, a, b):
    return a * np.exp(-b * x)


def _bare(tree):
    """Map every key of a nested dict to None (what a bare YAML key yields)."""
    return {k: (_bare(v) if isinstance(v, dict) else None) for k, v in tree.items()}


_VIS_KEYS = {
    "enabled": None,
    "output_dir": None,
    "filename_prefix": None,
    "formats": None,
    "style": None,
    "figure_size": None,
    "dpi": None,
    "font": None,
    "color_schemes": None,
    "active_scheme": None,
    "main_plot": {
        "show_grid": None,
        "grid_alpha": None,
        "legend": {"enabled": None, "location": None, "frameon": None},
        "data": {
            "color": None,
            "marker": None,
            "size": None,
            "alpha": None,
            "label": None,
            "show_errorbars": None,
            "capsize": None,
        },
        "fit": {
            "color": None,
            "n_points": None,
            "linewidth": None,
            "linestyle": None,
            "label": None,
        },
        "confidence_band": {"enabled": None, "level": None, "alpha": None},
        "annotation": {
            "enabled": None,
            "show_r_squared": None,
            "show_rmse": None,
            "show_chi_squared": None,
            "fontsize": None,
            "location": None,
        },
    },
    "residuals_plot": {
        "enabled": None,
        "type": None,
        "marker": None,
        "size": None,
        "alpha": None,
        "show_zero_line": None,
        "zero_line_style": None,
        "zero_line_color": None,
        "zero_line_width": None,
        "std_bands": {"enabled": True, "levels": None, "colors": None, "alpha": None},
    },
    "histogram": {
        "enabled": True,
        "bins": None,
        "color": None,
        "alpha": None,
        "edgecolor": None,
        "show_normal_fit": None,
    },
}


def test_visualization_all_bare_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0, 1, 20)
    res = {"popt": np.array([2.0, 1.0]), "pcov": np.eye(2) * 1e-4, "success": True}
    data = {"xdata": x, "ydata": _model(x, 2, 1) + 0.01 * np.sin(40 * x)}
    cfg = {"visualization": _bare(_VIS_KEYS)}
    cfg["visualization"]["residuals_plot"]["std_bands"]["enabled"] = True
    cfg["visualization"]["histogram"]["enabled"] = True
    cfg["visualization"]["enabled"] = True
    cfg["visualization"]["main_plot"]["confidence_band"]["enabled"] = True
    out = FitVisualizer().generate(res, data, _model, cfg)
    assert out, "default formats must be used when 'formats' is a bare key"
    assert all("None" not in p for p in out), "bare filename_prefix must not stringify"
    assert (tmp_path / "figures").is_dir()


def test_data_loader_bare_keys(tmp_path):
    x = np.linspace(0, 1, 6)
    ascii_f = tmp_path / "a.txt"
    np.savetxt(ascii_f, np.c_[x, 2 * x])
    cfg = {
        "format": None,
        "ascii": {
            "delimiter": None,
            "comment_char": None,
            "skip_header": None,
            "skip_footer": None,
            "dtype": None,
        },
        "columns": {"x": None, "y": None},
        "validation": {"require_finite": None, "min_points": None},
    }
    xs, ys, _ = DataLoader().load(ascii_f, cfg)
    np.testing.assert_allclose(xs, x)
    np.testing.assert_allclose(ys, 2 * x)

    csv_f = tmp_path / "a.csv"
    csv_f.write_text("x,y\n" + "\n".join(f"{a},{2 * a}" for a in x))
    cfg = {
        "format": "csv",
        "csv": {
            "delimiter": None,
            "header": None,
            "encoding": None,
            "missing_values": None,
            "skip_header": None,
        },
        "columns": {"x": None, "y": None},
        "validation": None,
    }
    _, ys, _ = DataLoader().load(csv_f, cfg)
    np.testing.assert_allclose(ys, 2 * x)

    npz_f = tmp_path / "a.npz"
    np.savez(npz_f, x=x, y=2 * x)
    cfg = {"format": "npz", "npz": {"x_key": None, "y_key": None}}
    _, ys, _ = DataLoader().load(npz_f, cfg)
    np.testing.assert_allclose(ys, 2 * x)


def test_data_loader_hdf5_bare_keys(tmp_path):
    h5py = pytest.importorskip("h5py")
    x = np.linspace(0, 1, 6)
    f = tmp_path / "a.h5"
    with h5py.File(f, "w") as h:
        h["/data/x"] = x
        h["/data/y"] = 2 * x
    cfg = {"format": "hdf5", "hdf5": {"x_path": None, "y_path": None}}
    _, ys, _ = DataLoader().load(f, cfg)
    np.testing.assert_allclose(ys, 2 * x)


def test_exporter_and_runner_bare_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0, 1, 20)
    np.savetxt(tmp_path / "d.txt", np.c_[x, _model(x, 2, 1)])
    cfg = yaml.safe_load(
        "data:\n  input_file: d.txt\n  format: ascii\n"
        "model:\n  type: builtin\n  name: exponential_decay\n"
        "fitting:\n  p0: [2.0, 1.0, 0.0]\n  absolute_sigma:\n  check_finite:\n"
        "  bounds:\n    lower:\n    upper:\n"
        "metadata:\n  workflow_name:\n  dataset_id:\n"
        "export:\n  results_file: out.json\n  format:\n  stdout:\n  skip_file_on_stdout:\n"
    )
    result = WorkflowRunner().run(cfg)
    assert result is not None
    # Exporter reads metadata/export bare keys without raising
    ResultExporter().export(result, cfg)
    assert (tmp_path / "out.json").exists()
