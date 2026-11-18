"""Regression tests for the 2026-11-16 CLI review (round 1)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError, FitError
from nlsq.cli.main import main
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.workflow_runner import WorkflowRunner

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "nlsq" / "cli" / "templates"


def _strict_json(path: Path) -> dict:
    def _bad(token: str):  # NaN / Infinity must never be written
        raise AssertionError(f"non-strict JSON constant {token}")

    return json.loads(path.read_text(), parse_constant=_bad)


# ---------------------------------------------------------------------------
# W1: custom model template documents keys the registry actually reads
# ---------------------------------------------------------------------------


def _doc_model_blocks() -> list[dict]:
    doc = (TEMPLATES / "custom_model_template.py").read_text(encoding="utf-8")
    doc = doc.split('"""')[1]
    blocks = re.findall(r"^( *)model:\n((?:\1 .*\n)+)", doc, re.MULTILINE)
    out = []
    for indent, body in blocks:
        out.append(yaml.safe_load(textwrap.dedent(f"{indent}model:\n{body}"))["model"])
    return out


def test_custom_template_docstring_has_model_blocks():
    assert len(_doc_model_blocks()) >= 2


@pytest.mark.parametrize("block", _doc_model_blocks())
def test_custom_template_documented_yaml_resolves(block, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(TEMPLATES / "custom_model_template.py", tmp_path / "m.py")
    cfg = dict(block)
    cfg["function"] = "damped_oscillator"  # the usage block's placeholder name
    for key in ("path", "file"):  # the documented placeholder path
        if key in cfg:
            cfg[key] = "m.py"
    model = ModelRegistry().get_model(cfg.get("path", ""), cfg)
    assert callable(model)


# ---------------------------------------------------------------------------
# W2: '#' comment preamble before a CSV header
# ---------------------------------------------------------------------------

_BODY = "0,1\n1,2\n2,3\n"


@pytest.mark.parametrize(
    "text, columns, csv_cfg",
    [
        ("# units: s, V\nx,y\n" + _BODY, {"x": "x", "y": "y"}, {}),
        ("# units: s, V\nx,y\n" + _BODY, {"x": 0, "y": 1}, {}),
        ("# generated\nx,y\n" + _BODY, {"x": 0, "y": 1}, {}),
        ("# a\n\n# b\nx,y\n" + _BODY, {"x": "x", "y": "y"}, {}),
        # skip_header counts physical lines, comments after it are skipped too
        ("junk line\n# note\nx,y\n" + _BODY, {"x": "y", "y": "x"}, {"skip_header": 1}),
    ],
)
def test_csv_comment_preamble_before_header(tmp_path, text, columns, csv_cfg):
    f = tmp_path / "d.csv"
    f.write_text(text)
    x, y, _ = DataLoader().load(f, {"columns": columns, "csv": csv_cfg})
    if columns == {"x": "y", "y": "x"}:
        x, y = y, x
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(y, [1, 2, 3])


@pytest.mark.parametrize("preamble", ["", "# c\n"])
def test_csv_multichar_delimiter_with_header(tmp_path, preamble):
    f = tmp_path / "d.csv"
    f.write_text(preamble + "x;;y\n0;;1\n1;;2\n2;;3\n")
    x, y, _ = DataLoader().load(
        f, {"columns": {"x": "x", "y": "y"}, "csv": {"delimiter": ";;"}}
    )
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(y, [1, 2, 3])


# ---------------------------------------------------------------------------
# W4: CSV with exactly one data row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header", [True, False])
def test_csv_single_data_row(tmp_path, header):
    f = tmp_path / "one.csv"
    f.write_text(("x,y\n" if header else "") + "0,1\n")
    x, y, _ = DataLoader().load(
        f, {"csv": {"header": header}, "validation": {"min_points": 1}}
    )
    assert x.tolist() == [0.0] and y.tolist() == [1.0]


def test_csv_single_column_still_a_column(tmp_path):
    f = tmp_path / "col.csv"
    f.write_text("0\n1\n2\n")
    with pytest.raises(DataLoadError, match="out of range"):
        DataLoader().load(f, {"csv": {"header": False}})


# ---------------------------------------------------------------------------
# S2 / agy: ASCII columns must be integer indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cols", [{"x": "t", "y": 1}, {"x": 0.0, "y": 1}])
def test_ascii_non_integer_columns_rejected_clearly(tmp_path, cols):
    f = tmp_path / "d.txt"
    f.write_text("0 1\n1 2\n2 3\n")
    with pytest.raises(DataLoadError, match="0-based integer"):
        DataLoader().load(f, {"columns": cols})


def test_hdf5_complex_dataset_gets_dtype_message(tmp_path):
    h5py = pytest.importorskip("h5py")
    f = tmp_path / "c.h5"
    with h5py.File(f, "w") as h:
        h["/data/x"] = np.arange(3.0) + 0j
        h["/data/y"] = np.arange(3.0)
    with pytest.raises(DataLoadError, match="unsupported dtype complex"):
        DataLoader().load(f, {})


# ---------------------------------------------------------------------------
# S3: ascii.dtype may not silently round data to float32
# ---------------------------------------------------------------------------


def test_ascii_float32_dtype_rejected(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("0 0.123456789012\n1 0.5\n2 0.75\n")
    with pytest.raises(DataLoadError, match="float64"):
        DataLoader().load(f, {"ascii": {"dtype": "float32"}})


# ---------------------------------------------------------------------------
# S4: non-UTF-8 CSV gives a DataLoadError naming csv.encoding
# ---------------------------------------------------------------------------


def test_csv_non_utf8_header_is_data_load_error(tmp_path):
    f = tmp_path / "d.csv"
    f.write_bytes(b"x\xb5,y\n0,1\n1,2\n2,3\n")
    with pytest.raises(DataLoadError, match="encoding"):
        DataLoader().load(f, {"columns": {"x": 0, "y": 1}})


# ---------------------------------------------------------------------------
# W3: a RuntimeError mentioning "max" is not "max_nfev exceeded"
# ---------------------------------------------------------------------------


def test_runtime_error_with_max_is_not_reported_as_max_nfev(monkeypatch):
    import nlsq

    def boom(**_kw):
        raise RuntimeError("cannot evaluate: x exceeds xmax of the lookup table")

    monkeypatch.setattr(nlsq, "curve_fit", boom)
    with pytest.raises(FitError) as ei:
        WorkflowRunner()._execute_fit(
            lambda x, a: a * x, np.arange(3.0), np.arange(3.0), None, None, None
        )
    assert "maximum function evaluations" not in str(ei.value)
    assert "xmax" in str(ei.value)


def test_runtime_error_with_maxfev_still_reported(monkeypatch):
    import nlsq

    def boom(**_kw):
        raise RuntimeError("Optimal parameters not found: maxfev=800 exceeded")

    monkeypatch.setattr(nlsq, "curve_fit", boom)
    with pytest.raises(FitError, match="maximum function evaluations"):
        WorkflowRunner()._execute_fit(
            lambda x, a: a * x, np.arange(3.0), np.arange(3.0), None, None, None
        )


# ---------------------------------------------------------------------------
# W5: underdetermined fit carries a warning in the JSON (exit stays 0)
# ---------------------------------------------------------------------------


def test_underdetermined_fit_exports_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("two.csv").write_text("x,y\n0,1\n1,0.5\n")
    Path("w.yaml").write_text(
        "data: {input_file: two.csv}\n"
        "model: {type: builtin, name: exponential_decay}\n"
        "fitting: {p0: [1, 1, 0]}\n"
        "export: {results_file: out.json}\n"
    )
    assert main(["fit", "w.yaml"]) == 0
    d = _strict_json(Path("out.json"))
    assert d["uncertainty_warnings"], d
    assert any("covariance" in w for w in d["uncertainty_warnings"])


def test_well_determined_fit_has_no_uncertainty_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0, 5, 30)
    np.savetxt(
        "w.csv",
        np.c_[x, 3 * np.exp(-0.7 * x) + 0.5 + 0.01 * np.sin(9 * x)],
        delimiter=",",
        header="x,y",
        comments="",
    )
    Path("w.yaml").write_text(
        "data: {input_file: w.csv}\n"
        "model: {type: builtin, name: exponential_decay}\n"
        "fitting: {p0: [1, 1, 0]}\n"
        "export: {results_file: out.json}\n"
    )
    assert main(["fit", "w.yaml"]) == 0
    assert "uncertainty_warnings" not in _strict_json(Path("out.json"))


# ---------------------------------------------------------------------------
# W6: shipped workflow template runs with method: hybrid_streaming
# ---------------------------------------------------------------------------


def _template_run(tmp_path, monkeypatch, method: str | None):
    monkeypatch.chdir(tmp_path)
    text = (TEMPLATES / "workflow_config_template.yaml").read_text(encoding="utf-8")
    if method is not None:
        text = text.replace('# method: "trf"', f'method: "{method}"')
    Path("workflow.yaml").write_text(text, encoding="utf-8")
    Path("data").mkdir()
    x = np.linspace(0, 5, 40)
    np.savetxt(
        "data/experiment.csv",
        np.c_[x, 3 * np.exp(-0.7 * x) + 0.5 + 0.01 * np.sin(9 * x)],
        delimiter=",",
        header="x,y",
        comments="",
    )
    return main(["fit", "workflow.yaml"])


def test_shipped_template_validates_and_runs(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        rc = _template_run(tmp_path, monkeypatch, None)
    assert rc == 0
    assert not [r for r in caplog.records if "Unknown" in r.getMessage()]
    assert _strict_json(tmp_path / "output" / "fit_results.json")["popt"]


@pytest.mark.slow
def test_shipped_template_runs_with_hybrid_streaming(tmp_path, monkeypatch, capsys):
    rc = _template_run(tmp_path, monkeypatch, "hybrid_streaming")
    err = capsys.readouterr().err
    assert rc == 0, err
    assert _strict_json(tmp_path / "output" / "fit_results.json")["popt"]
    # the options hybrid_streaming cannot honour are reported, not silent
    for key in ("ftol", "xtol", "gtol", "max_nfev"):
        assert key in err


def test_hybrid_streaming_drops_unsupported_options_with_warning(caplog):
    r = WorkflowRunner()
    cfg = {
        "fitting": {
            "method": "hybrid_streaming",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "max_nfev": 50,
            "loss": "linear",
        }
    }
    with caplog.at_level(logging.WARNING):
        _, _, kw = r._extract_fit_params(cfg, np.arange(3.0), np.arange(3.0), len)
    assert kw["method"] == "hybrid_streaming"
    for k in ("ftol", "xtol", "gtol", "max_nfev", "loss"):
        assert k not in kw
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "max_nfev" in msgs and "ftol" in msgs
    assert "loss" not in msgs  # 'linear' is the hybrid objective: nothing dropped


def test_hybrid_streaming_robust_loss_is_rejected():
    from nlsq.cli.errors import ConfigError

    with pytest.raises(ConfigError, match="loss"):
        WorkflowRunner()._extract_fit_params(
            {"fitting": {"method": "hybrid_streaming", "loss": "huber"}},
            np.arange(3.0),
            np.arange(3.0),
            len,
        )


def test_other_methods_still_forward_options():
    _, _, kw = WorkflowRunner()._extract_fit_params(
        {"fitting": {"method": "trf", "ftol": 1e-9, "max_nfev": 7, "loss": "huber"}},
        np.arange(3.0),
        np.arange(3.0),
        len,
    )
    assert kw["ftol"] == 1e-9 and kw["max_nfev"] == 7 and kw["loss"] == "huber"


# ---------------------------------------------------------------------------
# S1: visualization reuses the model resolved for the fit
# ---------------------------------------------------------------------------


def test_custom_model_file_executed_once_with_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "executed.log"
    Path("m.py").write_text(
        "import jax.numpy as jnp\n"
        f"open({str(marker)!r}, 'a').write('x')\n"
        "def line(x, a, b):\n    return a * x + b\n"
    )
    Path("d.csv").write_text("x,y\n" + "".join(f"{i},{2 * i + 1}\n" for i in range(8)))
    Path("w.yaml").write_text(
        "data: {input_file: d.csv}\n"
        "model: {type: custom, path: m.py, function: line, trusted: true}\n"
        "fitting: {p0: [1, 0]}\n"
        "export: {results_file: out.json}\n"
        "visualization: {enabled: true, output_dir: figs, formats: [png]}\n"
    )
    assert main(["fit", "w.yaml"]) == 0
    assert marker.read_text() == "x"


# ---------------------------------------------------------------------------
# Codex W4: inline "#" comment on the CSV header row
# ---------------------------------------------------------------------------


def test_csv_header_inline_comment_stripped(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("x,y # note\n0,1\n1,2\n")
    _x, y, _ = DataLoader().load(f, {"columns": {"x": "x", "y": "y"}})
    assert y.tolist() == [1.0, 2.0]


def test_csv_header_quoted_hash_is_kept(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text('x,"y # lit"\n0,1\n1,2\n')
    _, y, _ = DataLoader().load(f, {"columns": {"x": "x", "y": "y # lit"}})
    assert y.tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Codex W3: binary loaders validate stored shapes before flattening
# ---------------------------------------------------------------------------

_X = np.arange(6.0).reshape(2, 3)
_Y_BAD = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])  # (3, 2)


def test_npz_mismatched_shapes_rejected(tmp_path):
    f = tmp_path / "d.npz"
    np.savez(f, x=_X, y=_Y_BAD)
    with pytest.raises(DataLoadError, match="incompatible shapes"):
        DataLoader().load(f, {})


def test_hdf5_mismatched_shapes_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    f = tmp_path / "d.h5"
    with h5py.File(f, "w") as h:
        h["/data/x"] = _X
        h["/data/y"] = _Y_BAD
    with pytest.raises(DataLoadError, match="incompatible shapes"):
        DataLoader().load(f, {})


def test_npz_matching_grids_and_singleton_axes_still_load(tmp_path):
    f = tmp_path / "d.npz"
    np.savez(f, x=_X, y=2 * _X)
    x, y, _ = DataLoader().load(f, {})
    assert y.tolist() == (2 * x).tolist()
    g = tmp_path / "e.npz"
    np.savez(g, x=np.arange(4.0), y=np.arange(4.0).reshape(4, 1))
    assert len(DataLoader().load(g, {})[1]) == 4


# ---------------------------------------------------------------------------
# Codex W1/W2: confidence band follows the result's own noise/quantile rules
# ---------------------------------------------------------------------------


def _const(x, a):
    return a + 0 * np.asarray(x)


def test_confidence_band_known_sigma_uses_normal_quantile():
    from scipy import stats

    from nlsq.cli.visualization import FitVisualizer

    x = np.arange(3.0)
    lo, hi = FitVisualizer()._calculate_confidence_band(
        _const,
        x,
        np.array([1.0]),
        np.array([[1 / 3]]),
        0.95,
        ndata=3,
        sigma=np.ones(3),
        absolute_sigma=True,
    )
    half = stats.norm.ppf(0.975) * np.sqrt(1 / 3)
    np.testing.assert_allclose(hi - 1.0, half)
    np.testing.assert_allclose(1.0 - lo, half)


def test_confidence_band_estimated_sigma_uses_student_t():
    from scipy import stats

    from nlsq.cli.visualization import FitVisualizer

    _lo, hi = FitVisualizer()._calculate_confidence_band(
        _const, np.arange(3.0), np.array([1.0]), np.array([[1 / 3]]), 0.95, ndata=3
    )
    np.testing.assert_allclose(hi - 1.0, stats.t.ppf(0.975, 2) * np.sqrt(1 / 3))


def test_confidence_band_no_dof_is_nan_not_normal():
    from nlsq.cli.visualization import FitVisualizer

    with pytest.warns(RuntimeWarning, match="Underdetermined"):
        lo, hi = FitVisualizer()._calculate_confidence_band(
            lambda x, a, b: a * x + b,
            np.arange(2.0),
            np.array([1.0, 0.0]),
            np.eye(2),
            0.95,
            ndata=2,
        )
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_confidence_band_finite_near_domain_edge():
    from scipy import stats

    from nlsq.cli.visualization import FitVisualizer

    lo, hi = FitVisualizer()._calculate_confidence_band(
        lambda x, a: np.sqrt(a) + 0 * x,
        np.arange(3.0),
        np.array([1e-8]),
        np.array([[1e-16]]),
        0.95,
        ndata=3,
    )
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    sd = 1e-8 / (2 * np.sqrt(1e-8))
    np.testing.assert_allclose(hi - 1e-4, stats.t.ppf(0.975, 2) * sd, rtol=1e-2)


def test_unavailable_confidence_band_warns_and_is_omitted(tmp_path, caplog):
    import matplotlib.pyplot as plt

    from nlsq.cli.visualization import FitVisualizer

    cfg = {
        "visualization": {
            "enabled": True,
            "output_dir": str(tmp_path),
            "formats": ["png"],
            "main_plot": {"confidence_band": {"enabled": True}},
            "residuals_plot": {"enabled": False},
        }
    }
    data = {"xdata": np.arange(2.0), "ydata": np.arange(2.0), "sigma": None}
    result = {"popt": [1.0, 0.0], "pcov": np.eye(2).tolist()}
    with caplog.at_level(logging.WARNING), pytest.warns(RuntimeWarning):
        FitVisualizer().generate(result, data, lambda x, a, b: a * x + b, cfg)
    plt.close("all")
    assert any("confidence band" in r.getMessage().lower() for r in caplog.records)


def test_runner_result_carries_absolute_sigma(monkeypatch):
    import nlsq

    seen = {}
    real = nlsq.curve_fit

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    x = np.linspace(0, 1, 6)
    res = WorkflowRunner()._execute_fit(
        lambda x, a, b: a * x + b,
        x,
        2 * x + 1,
        None,
        np.full(6, 0.1),
        None,
        absolute_sigma=True,
    )
    assert res["absolute_sigma"] is True


# ---------------------------------------------------------------------------
# Round 2: hybrid_streaming with 2-D surface data, loader/band edge cases
# ---------------------------------------------------------------------------


def _plane_workflow(tmp_path, method):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (60, 2))
    z = 2 * xy[:, 0] - 3 * xy[:, 1] + 1
    np.savetxt(
        tmp_path / "s.csv", np.c_[xy, z], delimiter=",", header="x,y,z", comments=""
    )
    (tmp_path / "plane.py").write_text(
        "def plane(xy, a, b, c):\n    return a * xy[0] + b * xy[1] + c\n"
    )
    (tmp_path / "w.yaml").write_text(
        "data: {input_file: s.csv, columns: {x: x, y: y, z: z}}\n"
        "model: {type: custom, path: plane.py, function: plane, trusted: true}\n"
        f"fitting: {{p0: [1, 1, 0], method: {method}}}\n"
        "export: {results_file: out.json}\n"
    )


@pytest.mark.parametrize("method", ["trf", "hybrid_streaming"])
def test_surface_fit_works_for_every_method(tmp_path, monkeypatch, capsys, method):
    monkeypatch.chdir(tmp_path)
    _plane_workflow(tmp_path, method)
    assert main(["fit", "w.yaml"]) == 0, capsys.readouterr().err
    popt = _strict_json(tmp_path / "out.json")["popt"]
    np.testing.assert_allclose(popt, [2, -3, 1], atol=1e-3)


def test_csv_empty_after_comments_is_clear_error(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("# only a comment\n")
    with pytest.raises(DataLoadError, match="no data"):
        DataLoader().load(f, {})


def test_numpy_integer_columns_accepted_in_csv_and_ascii(tmp_path):
    c = tmp_path / "d.csv"
    c.write_text("x,y\n0,1\n1,2\n2,3\n")
    _, y, _ = DataLoader().load(c, {"columns": {"x": np.int64(0), "y": np.int64(1)}})
    assert y.tolist() == [1.0, 2.0, 3.0]
    a = tmp_path / "d.txt"
    a.write_text("0 1\n1 2\n2 3\n")
    _, y, _ = DataLoader().load(a, {"columns": {"x": np.int64(0), "y": np.int64(1)}})
    assert y.tolist() == [1.0, 2.0, 3.0]


def test_boolean_column_still_rejected(tmp_path):
    c = tmp_path / "d.csv"
    c.write_text("x,y\n0,1\n1,2\n")
    with pytest.raises(DataLoadError):
        DataLoader().load(c, {"columns": {"x": True, "y": 1}})


def test_numpy_style_commented_header_hint(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("# x,y\n0,1\n1,2\n")  # np.savetxt(header="x,y") default
    with pytest.raises(DataLoadError, match="comments"):
        DataLoader().load(f, {"columns": {"x": "x", "y": "y"}})


def test_confidence_band_model_domain_error_is_omitted(tmp_path, caplog):
    import matplotlib.pyplot as plt

    from nlsq.cli.visualization import FitVisualizer

    calls = []

    def picky(x, a):
        if np.size(x) == 500:  # the curve grid: first call draws the fit line,
            calls.append(1)  # every later one (the band's probes) is "outside"
            if len(calls) > 1:
                raise ValueError("a outside the model domain")
        return a * np.asarray(x)

    cfg = {
        "visualization": {
            "enabled": True,
            "output_dir": str(tmp_path),
            "formats": ["png"],
            "main_plot": {"confidence_band": {"enabled": True}},
            "residuals_plot": {"enabled": False},
        }
    }
    data = {"xdata": np.arange(5.0), "ydata": np.arange(5.0), "sigma": None}
    result = {"popt": [0.5], "pcov": [[0.01]]}
    with caplog.at_level(logging.WARNING):
        paths = FitVisualizer().generate(result, data, picky, cfg)
    plt.close("all")
    assert paths  # the main plot survives
    assert any("confidence band" in r.getMessage().lower() for r in caplog.records)
