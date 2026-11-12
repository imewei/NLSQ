"""Round-6b CLI regressions: model beside YAML, BOM, p0 length, messages, exporter."""

import json
import os

import numpy as np
import pytest
import yaml

from nlsq.cli.commands.fit import run_fit
from nlsq.cli.errors import CLIError, ConfigError, DataLoadError, ModelError
from nlsq.cli.result_exporter import ResultExporter

MODEL_SRC = """
def m(x, a, b):
    return a * x + b


def parameter_names():
    return ["slope", "offset"]
"""


def _project(root, model_path="model.py", model_src=MODEL_SRC, **extra_cfg):
    proj = root / "proj"
    proj.mkdir()
    x = np.linspace(0, 4, 30)
    np.savetxt(proj / "data.csv", np.c_[x, 2 * x + 1], delimiter=",")
    (proj / "model.py").write_text(model_src)
    cfg = {
        "data": {
            "input_file": "data.csv",
            "format": "csv",
            "columns": {"x": 0, "y": 1},
        },
        "model": {"type": "custom", "path": model_path, "function": "m"},
        "export": {"results_file": str(root / "default_out.json"), "format": "json"},
        **extra_cfg,
    }
    yml = proj / "cw.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    return proj, yml


def test_model_beside_yaml_loads_from_sibling_cwd(tmp_path, monkeypatch):
    _, yml = _project(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    result = run_fit(str(yml))
    np.testing.assert_allclose(result["popt"], [2.0, 1.0], rtol=1e-6)


def test_model_dotdot_escape_still_rejected(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text(MODEL_SRC)
    _, yml = _project(tmp_path, model_path="../outside/evil.py")
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    with pytest.raises(ModelError, match="traversal"):
        run_fit(str(yml))


def test_ascii_with_utf8_bom(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "d.txt"
    p.write_bytes(b"\xef\xbb\xbf0.0 1.0\n1.0 3.0\n2.0 5.0\n")
    x, _y, _ = DataLoader().load(str(p), {"format": "ascii"})
    np.testing.assert_allclose(x, [0, 1, 2])
    p2 = tmp_path / "e.txt"
    p2.write_bytes(b"\xef\xbb\xbf0.0 1.0\n1.0 3.0\n2.0 5.0\n9 9\n")
    x, _, _ = DataLoader().load(
        str(p2), {"format": "ascii", "ascii": {"skip_footer": 1}}
    )
    np.testing.assert_allclose(x, [0, 1, 2])


def test_wrong_length_p0_names_p0(tmp_path):
    _, yml = _project(tmp_path, fitting={"p0": [1.0, 1.0, 1.0]})
    with pytest.raises(ConfigError, match=r"fitting\.p0.*3.*2 parameter"):
        run_fit(str(yml))


def test_nan_error_does_not_promise_missing_option(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "n.csv"
    p.write_text("0,1\n1,nan\n2,5\n")
    with pytest.raises(DataLoadError) as ei:
        DataLoader().load(str(p), {"format": "csv", "columns": {"x": 0, "y": 1}})
    assert "require_finite: false" not in str(ei.value)
    assert "require_finite: false" not in str(getattr(ei.value, "suggestion", ""))


def test_unwritable_results_dir_is_cli_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    cfg = {
        "export": {"results_file": str(blocker / "sub" / "r.json"), "format": "json"}
    }
    with pytest.raises(CLIError, match="results directory"):
        ResultExporter().export({"popt": [1.0], "pcov": [[1.0]]}, cfg)


def test_parameter_names_from_model_file(tmp_path):
    out = tmp_path / "r.json"
    _, yml = _project(tmp_path, export={"results_file": str(out), "format": "json"})
    run_fit(str(yml))
    assert json.loads(out.read_text())["parameter_names"] == ["slope", "offset"]


def test_parameter_names_wrong_length_ignored_yaml_wins(tmp_path):
    out = tmp_path / "r.json"
    bad = MODEL_SRC.replace('["slope", "offset"]', '["only_one"]')
    _, yml = _project(
        tmp_path, model_src=bad, export={"results_file": str(out), "format": "json"}
    )
    run_fit(str(yml))
    assert "parameter_names" not in json.loads(out.read_text())

    out2 = tmp_path / "r2.json"
    cfg = yaml.safe_load(yml.read_text())
    cfg["export"]["results_file"] = str(out2)
    cfg["model"]["parameter_names"] = ["a", "b"]
    yml.write_text(yaml.safe_dump(cfg))
    run_fit(str(yml))
    assert json.loads(out2.read_text())["parameter_names"] == ["a", "b"]
