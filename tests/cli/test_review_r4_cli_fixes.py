"""Round-4 CLI regressions: batch bare filename_prefix, default model bounds, fitting keys."""

import logging

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from nlsq.cli.commands.batch import run_batch
from nlsq.cli.errors import ConfigError
from nlsq.cli.workflow_runner import WorkflowRunner


def _decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def _data(tmp_path):
    x = np.linspace(0, 4, 40)
    path = tmp_path / "d.txt"
    np.savetxt(path, np.c_[x, _decay(x, 2.0, 1.0, 0.5)])
    return path


def _cfg(data, fitting=None):
    return {
        "data": {"input_file": str(data), "format": "ascii"},
        "model": {"type": "builtin", "name": "exponential_decay"},
        "fitting": fitting if fitting is not None else {"p0": [2.0, 1.0, 0.5]},
    }


def test_batch_bare_filename_prefix_gets_workflow_stem(tmp_path):
    data = _data(tmp_path)
    figs = tmp_path / "figures"
    paths = []
    for name in ("a", "b"):
        cfg = _cfg(data)
        cfg["visualization"] = yaml.safe_load(
            f"enabled: true\noutput_dir: {figs}\nfilename_prefix:\nformats: [png]\n"
        )
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        paths.append(str(p))
    run_batch(paths, summary_file=str(tmp_path / "summary.json"))
    names = sorted(f.name for f in figs.iterdir())
    assert names == ["a_combined.png", "b_combined.png"], names


def test_default_bounds_skipped_for_lm(tmp_path):
    cfg = _cfg(_data(tmp_path), {"p0": [2.0, 1.0, 0.5], "method": "lm"})
    result = WorkflowRunner().run(cfg)
    assert result["success"]


def test_default_bounds_logged_for_trf(tmp_path, caplog):
    cfg = _cfg(_data(tmp_path), {"p0": [2.0, 1.0, 0.5], "method": "trf"})
    with caplog.at_level(logging.INFO, logger="nlsq.cli.workflow_runner"):
        WorkflowRunner().run(cfg)
    assert any("default bounds" in r.getMessage() for r in caplog.records)


def test_unknown_fitting_key_warns_with_suggestion(tmp_path, caplog):
    cfg = _cfg(_data(tmp_path), {"p0": [2.0, 1.0, 0.5], "maxfev": 50})
    with caplog.at_level(logging.WARNING, logger="nlsq.cli.workflow_runner"):
        WorkflowRunner().run(cfg)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("maxfev" in m and "max_nfev" in m for m in msgs), msgs


# absolute_sigma already raised for non-bools before the fix; one control case.
@pytest.mark.parametrize(
    "key,value",
    [
        ("check_finite", "false"),
        ("check_finite", 1),
        ("check_finite", 0.0),
        ("absolute_sigma", "false"),
    ],
)
def test_fitting_bool_keys_must_be_bool(tmp_path, key, value):
    cfg = _cfg(_data(tmp_path), {"p0": [2.0, 1.0, 0.5], key: value})
    with pytest.raises(ConfigError):
        WorkflowRunner().run(cfg)
