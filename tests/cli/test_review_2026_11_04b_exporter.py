"""Regression tests for the 2026-11-04 result_exporter review findings."""

import datetime
import json
import math

import numpy as np
import pytest

from nlsq.cli.errors import CLIError
from nlsq.cli.result_exporter import ResultExporter

# YAML turns `dataset_id: 2026-11-04` into a date object.
DATE_CFG = {"metadata": {"dataset_id": datetime.date(2026, 11, 4)}}


def _cfg(path, fmt="json", **extra):
    return {"export": {"results_file": str(path), "format": fmt}, **extra}


def test_json_export_failure_keeps_previous_file(tmp_path):
    p = tmp_path / "out.json"
    p.write_text('{"old": "good result"}')
    bad = {(1, 2): 1}  # tuple key: unserializable
    with pytest.raises(CLIError):
        ResultExporter().export(
            {"popt": [1.0], "pcov": [[1.0]], "message": bad}, _cfg(p)
        )
    assert json.loads(p.read_text()) == {"old": "good result"}
    assert list(tmp_path.iterdir()) == [p]  # no stray temp file


def test_json_export_date_metadata(tmp_path):
    p = tmp_path / "out.json"
    ResultExporter().export({"popt": [1.0], "pcov": [[1.0]]}, {**_cfg(p), **DATE_CFG})
    assert json.loads(p.read_text())["metadata"]["dataset_id"] == "2026-11-04"


def test_stdout_date_metadata(capsys):
    cfg = {"export": {"stdout": True, "skip_file_on_stdout": True}, **DATE_CFG}
    ResultExporter().export({"popt": [1.0], "pcov": [[1.0]]}, cfg)
    assert json.loads(capsys.readouterr().out)["metadata"]["dataset_id"] == "2026-11-04"


def test_stdout_unserializable_raises_cli_error():
    cfg = {"export": {"stdout": True}, "metadata": {"dataset_id": 1}}
    data = ResultExporter()._prepare_export_data({"popt": [1.0]}, cfg)
    data["bad"] = {(1, 2): 1}  # tuple key: unserializable
    with pytest.raises(CLIError):
        ResultExporter()._export_stdout(data)


def test_uncertainties_per_entry():
    ex = ResultExporter()
    d = ex._prepare_export_data(
        {"popt": [1.0, 2.0], "pcov": [[0.04, 0], [0, np.inf]]}, {}
    )
    assert d["uncertainties"][0] == pytest.approx(0.2)
    assert d["uncertainties"][1] == math.inf
    assert "uncertainty_warnings" not in d

    d = ex._prepare_export_data(
        {"popt": [1.0, 2.0], "pcov": [[0.04, 0], [0, -1e-30]]}, {}
    )
    assert d["uncertainties"][0] == pytest.approx(0.2)
    assert math.isnan(d["uncertainties"][1])
    assert len(d["uncertainty_warnings"]) == 1
    assert "1" in d["uncertainty_warnings"][0]


def test_csv_extra_parameter_names_raise(tmp_path):
    with pytest.raises(CLIError):
        ResultExporter().export(
            {"popt": [1.0, 2.0], "pcov": [[1, 0], [0, 1]]},
            _cfg(tmp_path / "o.csv", "csv", model={"parameter_names": ["a", "b", "c"]}),
        )


def test_csv_fewer_names_still_padded(tmp_path):
    p = tmp_path / "o.csv"
    ResultExporter().export(
        {"popt": [1.0, 2.0], "pcov": [[1, 0], [0, 1]]},
        _cfg(p, "csv", model={"parameter_names": ["a"]}),
    )
    rows = p.read_text().splitlines()
    assert rows[1].startswith("a,") and rows[2].startswith("p1,")


def test_zero_d_popt_csv(tmp_path):
    p = tmp_path / "o.csv"
    ResultExporter().export({"popt": np.float64(3.0), "pcov": [[1.0]]}, _cfg(p, "csv"))
    assert p.read_text().splitlines()[1].startswith("p0,3.0")
