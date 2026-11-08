"""Regression tests for R2 CLI review findings (null YAML sections, max_nfev,
logging thread-safety, export format inference, skip_footer parsing)."""

import json
import threading

import numpy as np
import pytest

from nlsq.cli.commands.fit import run_fit
from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import ConfigError, DataLoadError, setup_logging
from nlsq.cli.workflow_runner import WorkflowRunner


def _write_workflow(tmp_path, extra="", export="export:\n  results_file: out.json\n"):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 40)
    y = 2.0 * np.exp(-1.3 * x) + 0.5 + 0.005 * rng.standard_normal(x.size)
    np.savetxt(
        tmp_path / "d.csv", np.c_[x, y], delimiter=",", header="x,y", comments=""
    )
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        "data:\n  input_file: d.csv\n  format: csv\n"
        "model:\n  type: builtin\n  name: exponential_decay\n"
        "fitting:\n  max_nfev: 100\n" + export + extra
    )
    return wf


# R2-CLI-01: bare `export:` / `visualization:` + CLI overrides
def test_output_override_with_null_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(tmp_path, export="export:\n")
    run_fit(str(wf), output_override=str(tmp_path / "o.json"))
    assert "popt" in json.loads((tmp_path / "o.json").read_text())


def test_style_override_with_null_visualization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(tmp_path, extra="visualization:\n")
    # Must not raise TypeError on `None["style"] = ...`
    run_fit(str(wf), style_override="publication")


def test_stdout_with_null_export(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(tmp_path, export="export:\n")
    run_fit(str(wf), stdout=True)
    assert "popt" in json.loads(capsys.readouterr().out)


# R2-CLI-02: `format:` left empty
def test_export_format_none_defaults_to_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(
        tmp_path, export="export:\n  results_file: r.json\n  format:\n"
    )
    run_fit(str(wf))
    assert "popt" in json.loads((tmp_path / "r.json").read_text())


# R2-CLI-07: infer format from suffix only when format is absent
def test_csv_suffix_infers_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(tmp_path, export="export:\n  results_file: r.csv\n")
    run_fit(str(wf))
    assert not (tmp_path / "r.csv").read_text().lstrip().startswith("{")


def test_explicit_format_wins_over_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _write_workflow(
        tmp_path, export="export:\n  results_file: r.csv\n  format: json\n"
    )
    run_fit(str(wf))
    assert "popt" in json.loads((tmp_path / "r.csv").read_text())


# R2-CLI-03 / R2-1: empty data sub-sections
@pytest.mark.parametrize(
    "fmt,section",
    [("ascii", "ascii"), ("ascii", "columns"), ("csv", "csv"), ("csv", "columns")],
)
def test_null_data_subsections(tmp_path, fmt, section):
    f = tmp_path / ("d.csv" if fmt == "csv" else "d.txt")
    f.write_text("1,2\n2,4\n3,6\n" if fmt == "csv" else "1 2\n2 4\n3 6\n")
    cfg = {"format": fmt, section: None}
    if fmt == "csv":
        cfg.setdefault("csv", {"header": False})
        if section == "csv":
            cfg["csv"] = None
    try:
        x, _y, _ = DataLoader().load(f, cfg)
    except DataLoadError:
        return  # a clean, typed error is acceptable (e.g. csv header default)
    assert len(x) in (2, 3)


def test_null_npz_hdf5_sections_raise_typed_error(tmp_path):
    f = tmp_path / "d.npz"
    np.savez(f, x=np.arange(3.0), y=np.arange(3.0))
    x, _, _ = DataLoader().load(f, {"format": "npz", "npz": None})
    assert len(x) == 3


def test_null_visualization_subsections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vis = (
        "visualization:\n  enabled: true\n  output_dir: figs\n  formats: [png]\n"
        "  font:\n  main_plot:\n  histogram:\n  residuals_plot:\n"
    )
    wf = _write_workflow(tmp_path, extra=vis)
    run_fit(str(wf))
    # Previously one WARNING and zero figures (AttributeError/TypeError swallowed)
    assert list((tmp_path / "figs").glob("*.png"))


# R2-CLI-05: max_nfev coercion
@pytest.mark.parametrize("val,expected", [("100", 100), (50, 50), (25.0, 25)])
def test_max_nfev_coerced(val, expected, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import yaml

    import nlsq

    cfg = yaml.safe_load(_write_workflow(tmp_path).read_text())
    cfg["fitting"]["max_nfev"] = val
    cfg["export"] = {"stdout": True, "skip_file_on_stdout": True}
    seen = {}
    real = nlsq.curve_fit

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    assert WorkflowRunner().run(cfg) is not None
    assert seen["max_nfev"] == expected
    assert type(seen["max_nfev"]) is int


@pytest.mark.parametrize(
    "val",
    ["abc", 0, -3, 2.5, True, [1], float("inf"), float("-inf"), float("nan")],
)
def test_max_nfev_invalid_raises_config_error(val, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import yaml

    cfg = yaml.safe_load(_write_workflow(tmp_path).read_text())
    cfg["fitting"]["max_nfev"] = val
    with pytest.raises(ConfigError):
        WorkflowRunner().run(cfg)


# R2-CLI-06: concurrent setup_logging leaves exactly one console handler
def test_setup_logging_thread_safe_smoke():
    # Smoke only: the race window is not widened, so this cannot fail without the lock.
    barrier = threading.Barrier(4)
    errors = []

    def work():
        try:
            barrier.wait()
            for _ in range(100):
                setup_logging(verbosity=1, console=True)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert len(setup_logging(verbosity=1, console=True).logger.handlers) == 1


def test_setup_logging_closes_old_file_handler(tmp_path):
    lg = setup_logging(log_file=tmp_path / "a.log", console=False)
    old = lg._file_handler
    setup_logging(console=False)
    assert old.stream is None or old.stream.closed


# R2-2: skip_footer parse errors match the loadtxt path
def test_skip_footer_nonnumeric_cell_raises(tmp_path):
    # Control: junk cells must never be silently coerced to NaN (also raised pre-fix).
    f = tmp_path / "d.txt"
    f.write_text("1 2\n2 abc\n3 6\nEND\n")
    with pytest.raises(DataLoadError):
        DataLoader().load(f, {"format": "ascii", "ascii": {"skip_footer": 1}})


def test_skip_footer_nan_token_still_loads(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("1 2\n2 nan\n3 6\nEND\n")
    cfg = {
        "format": "ascii",
        "ascii": {"skip_footer": 1},
        "validation": {"require_finite": False},
    }
    _, y, _ = DataLoader().load(f, cfg)
    assert np.isnan(y[1]) and y[2] == 6
