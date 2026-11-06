"""Regression tests for round-5 CLI fixes (validator denylist, p0/bounds errors)."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.errors import ModelError
from nlsq.cli.model_validation import validate_model
from nlsq.cli.workflow_runner import WorkflowRunner

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/scripts/10_cli-commands/models"
TAIL = "\ndef model(x, a, b):\n    return a * x + b\n"

# Benign payloads: the validator only parses them; nothing here is ever executed.
BLOCKED = [
    "import cProfile\ncProfile.run('1 + 1')\n",
    "import profile\nprofile.run('1 + 1')\n",
    "import trace\ntrace.Trace().run('1 + 1')\n",
    "import bdb\nbdb.Bdb().run('1 + 1')\n",
    "import pdb\npdb.run('1 + 1')\n",
    "import webbrowser\nwebbrowser.open('about:blank')\n",
    "import doctest\n",
    "import sched\n",
    "import atexit\n",
    "import signal\n",
    "import resource\n",
    "import threading\n",
    "import sqlite3\n",
    "import tempfile\n",
    "import zipfile\n",
    "import tarfile\n",
    "import logging\n",
    "import unittest\n",
    "import tkinter\n",
    "import site\n",
    "import zipimport\n",
    "import imp\n",
    "import venv\n",
    "import ensurepip\n",
    "import setuptools\n",
    "import compileall\n",
    "import py_compile\n",
    "import socketserver\n",
    "import xmlrpc.client\n",
    "import ssl\n",
    "import joblib\n",
    "import sympy\n",
    "import numexpr\n",
    "import h5py\n",
    "import torch\n",
    "import requests\n",
    "from joblib import load\n",
]


@pytest.mark.parametrize("payload", BLOCKED)
def test_validator_rejects_exec_and_spawn_primitives(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "evil.py"
    f.write_text(payload + TAIL)
    assert not validate_model(f).is_valid


def test_legit_numeric_imports_still_validate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "ok.py"
    f.write_text(
        "import math\nimport functools\nimport numpy as np\nimport jax\n"
        "import jax.numpy as jnp\nfrom scipy.special import erf\n"
        "from typing import Any\n" + TAIL
    )
    result = validate_model(f)
    assert result.is_valid, result.violations


@pytest.mark.parametrize(
    "src",
    [
        *sorted(EXAMPLES.glob("*.py")),
        ROOT / "nlsq/cli/templates/custom_model_template.py",
    ],
    ids=lambda p: p.name,
)
def test_shipped_models_still_validate(src, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / src.name
    shutil.copyfile(src, f)
    result = validate_model(f)
    assert result.is_valid, result.violations


def _model():
    def m(x, a, b):
        return a * x + b

    return m


def test_failing_bounds_is_an_error_not_unconstrained(tmp_path):
    """R5-GUI-2: a broken user bounds() must not silently drop the bounds."""
    m = _model()

    def bad_bounds():
        raise ValueError("bounds() returned bounds of length 4, but model has 2")

    m.bounds = bad_bounds
    with pytest.raises(ModelError, match="bounds"):
        WorkflowRunner()._extract_fit_params(
            {"fitting": {}}, np.arange(3.0), np.arange(3.0), m
        )


def test_failing_estimate_p0_warns_and_falls_back(caplog):
    """R5-GUI-2: a broken estimate_p0 is logged, then the default p0 is used."""
    m = _model()

    def bad_p0(xdata, ydata):
        raise ValueError("boom-p0")

    m.estimate_p0 = bad_p0
    with caplog.at_level(logging.WARNING):
        p0, _, _ = WorkflowRunner()._extract_fit_params(
            {"fitting": {"p0": "auto"}}, np.arange(3.0), np.arange(3.0), m
        )
    assert p0 is None
    assert "boom-p0" in caplog.text
