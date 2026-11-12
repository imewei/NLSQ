"""Regression tests for black-box CLI fuzz findings (fit/config/loader)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from nlsq.cli.main import main

CUSTOM_MODEL = (
    "import jax.numpy as jnp\n\n\ndef model(x, a, b, c):\n"
    "    return a * jnp.exp(-b * x) + c\n"
)


def _csv(path: Path) -> Path:
    x = np.linspace(0, 4, 50)
    y = 2.5 * np.exp(-1.3 * x) + 0.5
    rows = ["x,y,s"] + [f"{a},{b},0.01" for a, b in zip(x, y, strict=True)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _cfg(tmp_path: Path, **over) -> dict:
    cfg = {
        "data": {
            "input_file": str(_csv(tmp_path / "d.csv")),
            "format": "csv",
            "columns": {"x": 0, "y": 1},
        },
        "model": {"type": "builtin", "name": "exponential_decay"},
        "fitting": {"p0": [2.0, 1.0, 0.0]},
        "export": {"results_file": str(tmp_path / "out.json")},
    }
    cfg.update(over)
    return {k: v for k, v in cfg.items() if v is not None}


def _write(tmp_path: Path, cfg: dict, name: str = "w.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


def _env():
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu", NLSQ_SKIP_GPU_CHECK="1", QT_QPA_PLATFORM="offscreen"
    )
    return env


# H1 ------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.serial
def test_stdout_mode_is_pure_json_with_hybrid_streaming(tmp_path):
    cfg = _cfg(tmp_path, export=None)
    cfg["fitting"]["method"] = "hybrid_streaming"
    w = _write(tmp_path, cfg)
    proc = subprocess.run(
        [sys.executable, "-m", "nlsq.cli.main", "fit", str(w), "--stdout"],
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
        cwd=tmp_path,
        timeout=300,
    )
    doc = json.loads(proc.stdout)  # whole stdout must be the JSON document
    assert proc.returncode == 0, proc.stderr[-500:]
    assert isinstance(doc, dict) and doc, "stdout JSON must be a non-empty fit result"


# H2 ------------------------------------------------------------------------
def test_negative_column_index_rejected(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg["data"]["columns"] = {"x": 0, "y": -1}
    assert main(["fit", str(_write(tmp_path, cfg))]) == 1
    assert "columns.y" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


# M1 ------------------------------------------------------------------------
@pytest.mark.parametrize("section", ["data", "model", "fitting", "export"])
def test_non_mapping_section_is_clean_error(tmp_path, capsys, section):
    cfg = _cfg(tmp_path)
    cfg[section] = [1, 2]
    assert main(["fit", str(_write(tmp_path, cfg))]) == 1
    err = capsys.readouterr().err
    assert section in err and "mapping" in err


def test_non_mapping_section_in_batch_is_clean_error(tmp_path):
    from nlsq.cli.errors import CLIError

    cfg = _cfg(tmp_path)
    cfg["data"] = [1, 2]
    w = _write(tmp_path, cfg)
    try:
        from nlsq.cli.commands.batch import run_batch

        results = run_batch([str(w)], max_workers=1, continue_on_error=True)
    except CLIError:
        return
    assert results[0]["status"] == "failed"
    assert "AttributeError" not in str(results[0])


# M2 ------------------------------------------------------------------------
def test_non_utf8_yaml_is_clean_error(tmp_path, capsys):
    w = tmp_path / "bad.yaml"
    w.write_bytes(b"data: \xff\xfe\xfa\x80\n")
    assert main(["fit", str(w)]) == 1
    assert "Configuration Error" in capsys.readouterr().err


# M3 ------------------------------------------------------------------------
def test_config_output_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config", "--workflow", "-o", "sub/dir/x.yaml"]) == 0
    assert (tmp_path / "sub/dir/x.yaml").is_file()


def test_config_output_existing_directory_is_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d").mkdir()
    assert main(["config", "--workflow", "-o", "d", "--force"]) == 1
    assert "directory" in capsys.readouterr().err
    assert not list((tmp_path / "d").iterdir())


def test_config_output_parent_is_file_is_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f").write_text("x")
    assert main(["config", "--workflow", "-o", "f/x.yaml"]) == 1
    assert "Error" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["", "DIR"])
def test_fit_output_override_empty_or_directory(tmp_path, capsys, bad):
    (tmp_path / "DIR").mkdir()
    target = str(tmp_path / "DIR") if bad == "DIR" else bad
    w = _write(tmp_path, _cfg(tmp_path))
    assert main(["fit", str(w), "-o", target]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Invalid --output path" in err  # early ConfigError, not a late write failure


# M6 ------------------------------------------------------------------------
def test_missing_export_section_is_error(tmp_path, capsys):
    w = _write(tmp_path, _cfg(tmp_path, export=None))
    assert main(["fit", str(w)]) == 1
    assert "No output file specified" in capsys.readouterr().err


def test_missing_export_section_ok_with_stdout(tmp_path, capsys):
    w = _write(tmp_path, _cfg(tmp_path, export=None))
    assert main(["fit", str(w), "--stdout"]) == 0
    json.loads(capsys.readouterr().out)


# M7 ------------------------------------------------------------------------
def test_custom_model_path_resolves_beside_yaml(tmp_path, monkeypatch):
    (tmp_path / "cm").mkdir()
    (tmp_path / "cm" / "ok.py").write_text(CUSTOM_MODEL)
    cfg = _cfg(tmp_path)
    cfg["model"] = {"type": "custom", "path": "cm/ok.py", "function": "model"}
    cfg["export"] = {"results_file": str(tmp_path / "out.json")}
    (tmp_path / "clean").mkdir()
    w = tmp_path / "clean" / "rel.yaml"
    (tmp_path / "clean" / "cm").mkdir()
    (tmp_path / "clean" / "cm" / "ok.py").write_text(CUSTOM_MODEL)
    w.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)  # CWD is the parent: only YAML-dir lookup finds it
    (tmp_path / "cm" / "ok.py").unlink()
    assert main(["fit", "clean/rel.yaml"]) == 0
    assert (tmp_path / "out.json").is_file()


# LOW a ---------------------------------------------------------------------
def test_filename_prefix_with_traversal_rejected(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg["visualization"] = {
        "enabled": True,
        "output_dir": str(tmp_path / "figs"),
        "filename_prefix": "../../evil",
        "formats": ["png"],
    }
    assert main(["fit", str(_write(tmp_path, cfg))]) == 1
    assert "filename_prefix" in capsys.readouterr().err
    assert not (tmp_path / "figs").exists()


# LOW b ---------------------------------------------------------------------
def test_formats_bare_string_is_one_format(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["visualization"] = {
        "enabled": True,
        "output_dir": str(tmp_path / "figs"),
        "filename_prefix": "p",
        "formats": "png",
    }
    assert main(["fit", str(_write(tmp_path, cfg))]) == 0
    assert (tmp_path / "figs" / "p_combined.png").is_file()


# LOW c ---------------------------------------------------------------------
def test_nonconverged_fit_overwriting_results_warns(tmp_path, capsys):
    out = tmp_path / "out.json"
    out.write_text("{}")
    cfg = _cfg(tmp_path)
    cfg["fitting"]["max_nfev"] = 1
    cfg["fitting"]["p0"] = [50.0, 9.0, 40.0]
    rc = main(["fit", str(_write(tmp_path, cfg))])
    if rc == 0:
        pytest.skip("fit converged in one evaluation")
    assert "overwrote existing results file" in capsys.readouterr().err
