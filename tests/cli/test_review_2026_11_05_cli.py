"""Regression tests for the 2026-11-05 CLI review (validator, helpers, batch)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.commands import batch as batch_mod
from nlsq.cli.errors import CLIError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import validate_model

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/scripts/10_cli-commands/models"
HEADER = "import jax.numpy as jnp\n"
MODEL = "\ndef model(x, a, b):\n    return a * jnp.exp(-b * x)\n"


@pytest.mark.parametrize(
    "payload",
    [
        "import yaml\nyaml.unsafe_load('!!python/object/apply:subprocess.call [[touch, PWNED]]')\n",
        "from yaml import FullLoader, load\nload('x', Loader=FullLoader)\n",
        "import pandas as pd\npd.read_pickle('x.pkl')\n",
        "from pandas import read_pickle\nread_pickle('x.pkl')\n",
        "PAYLOAD = '!!python/object/apply:os.system [id]'\n",
    ],
)
def test_validator_rejects_yaml_pandas_deserialisation(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "evil.py"
    f.write_text(HEADER + payload + MODEL)
    assert not validate_model(f).is_valid


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


def _example_model(tmp_path, monkeypatch, fn):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "cm.py"
    shutil.copyfile(EXAMPLES / "custom_models.py", f)
    return ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": fn}
    )


@pytest.mark.parametrize(
    ("fn", "n"), [("michaelis_menten", 2), ("damped_oscillator", 4)]
)
def test_example_helpers_found_for_each_model(tmp_path, monkeypatch, fn, n):
    """CLI-2: `<fn>_estimate_p0` / `<fn>_bounds` naming is resolved."""
    model = _example_model(tmp_path, monkeypatch, fn)
    x = np.linspace(0.1, 5, 40)
    y = np.abs(np.sin(x)) + 0.5
    assert len(model.estimate_p0(x, y)) == n
    lo, hi = model.bounds()
    assert len(lo) == len(hi) == n


def test_mismatched_bare_helpers_never_yield_wrong_length(tmp_path, monkeypatch):
    """CLI-2: a p0/bounds whose length != parameter count is rejected."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text(
        HEADER
        + "\ndef only(x, a, b):\n    return a * x + b\n"
        + "\ndef estimate_p0(xdata, ydata):\n    return [1.0, 2.0, 3.0, 4.0]\n"
        + "\ndef bounds():\n    return ([0.0] * 4, [1.0] * 4)\n"
    )
    model = ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": "only"}
    )
    with pytest.raises(ValueError, match="parameter"):
        model.estimate_p0(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="parameter"):
        model.bounds()


def test_batch_reports_skipped_workflows(tmp_path, monkeypatch):
    """CLI-3: --no-continue-on-error lists skipped workflows in results/summary."""
    from nlsq.cli.commands import fit

    def fake(workflow_path, **kw):
        if "bad" in workflow_path:
            raise CLIError("boom")
        return {"success": True}

    monkeypatch.setattr(fit, "run_fit", fake)
    paths = []
    for n in ("bad.yaml", "ok.yaml", "ok2.yaml"):
        (tmp_path / n).write_text("x: 1\n")
        paths.append(str(tmp_path / n))
    summary_file = tmp_path / "sum.json"
    results = batch_mod.run_batch(
        paths, summary_file=str(summary_file), max_workers=1, continue_on_error=False
    )
    assert [r["status"] for r in results].count("skipped") == 2
    import json

    summary = json.loads(summary_file.read_text())
    assert summary["skipped"] == 2
    assert summary["failed"] == 1
    assert summary["succeeded"] == 0
    assert len(summary["skipped_workflows"]) == 2
