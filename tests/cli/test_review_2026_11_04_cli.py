"""Regression tests for the 2026-11-04 review (cli subpackage)."""

import importlib._bootstrap_external as bootstrap_external
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.errors import ConfigError, ModelError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import validate_model
from nlsq.cli.workflow_runner import WorkflowRunner

BENIGN = (
    "import jax.numpy as jnp\n\ndef model(x, a, b):\n    return a * jnp.exp(-b * x)\n"
)


def _config(path: Path) -> dict:
    return {"type": "custom", "path": str(path), "function": "model"}


def test_utf7_coding_cookie_cannot_hide_payload(tmp_path, monkeypatch):
    """Validator must see what the interpreter executes (PEP 263 cookie)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m7.py").write_text(
        "# -*- coding: utf-7 -*-\n"
        "import jax.numpy as jnp\n"
        "#+AAo-open(+ACI-PWNED_UTF7+ACI-, +ACI-w+ACI-).write(+ACI-x+ACI-)\n"
        "def model(x, a, b):\n    return a * jnp.exp(-b * x)\n"
    )
    assert not validate_model(Path("m7.py")).is_valid
    with pytest.raises(ModelError):
        ModelRegistry().get_model("custom", _config(Path("m7.py")))
    assert not (tmp_path / "PWNED_UTF7").exists()


def test_planted_pyc_is_not_executed(tmp_path, monkeypatch):
    """A __pycache__ .pyc mimicking the benign source must not run."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "benign.py"
    src.write_text(BENIGN)
    st = src.stat()
    evil = compile("open('PWNED_PYC', 'w').write('x')\n" + BENIGN, str(src), "exec")
    pyc = Path(importlib.util.cache_from_source(str(src)))
    pyc.parent.mkdir(exist_ok=True)
    pyc.write_bytes(
        bootstrap_external._code_to_timestamp_pyc(evil, int(st.st_mtime), st.st_size)
    )
    model = ModelRegistry().get_model("custom", _config(Path("benign.py")))
    assert callable(model)
    assert not (tmp_path / "PWNED_PYC").exists()


def test_validate_model_accepts_explicit_source(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(BENIGN)
    assert validate_model(p, source=BENIGN.encode()).is_valid
    assert not validate_model(p, source=b"import os\nos.system('x')\n").is_valid


@pytest.mark.parametrize("bounds", [[[0, 10], [0, 5], [0, 1]], [0, 10, 5], "0,10"])
def test_malformed_fitting_bounds_raise(bounds):
    runner = WorkflowRunner()
    with pytest.raises(ConfigError, match=r"fitting\.bounds"):
        runner._extract_fit_params(
            {"fitting": {"bounds": bounds}}, np.zeros(3), np.zeros(3), lambda x, a: x
        )


def test_wellformed_fitting_bounds_still_accepted():
    runner = WorkflowRunner()
    _, b, _ = runner._extract_fit_params(
        {"fitting": {"bounds": [[0, 0], [1, 1]]}}, np.zeros(3), np.zeros(3), lambda x: x
    )
    assert b is not None and len(b) == 2
