"""Custom model helpers can decline to supply initial parameters."""

import numpy as np
import pytest

from nlsq.cli.workflow_runner import WorkflowRunner


@pytest.mark.serial
def test_none_estimator_uses_default_initial_guess(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "def model(x, a, b):\n    return a * x + b\n"
        "def estimate_p0(x, y):\n    return None\n"
        "def bounds():\n    return None\n"
    )
    runner = WorkflowRunner()
    config = {
        "model": {"type": "custom", "path": str(model_file), "function": "model"},
        "fitting": {"p0": "auto"},
    }
    x = np.arange(6.0)
    y = 2 * x + 3
    model = runner._resolve_model(config)
    p0, bounds, kwargs = runner._extract_fit_params(config, x, y, model)
    assert p0 is None
    assert bounds is None
    result = runner._execute_fit(model, x, y, p0, None, bounds, **kwargs)
    np.testing.assert_allclose(result["popt"], [2, 3], rtol=1e-6)


def test_explicit_initial_guess_skips_none_estimator():
    def model(x, a):
        return a * x

    def unexpected_estimator(x, y):
        pytest.fail("Explicit p0 must bypass the estimator")

    model.estimate_p0 = unexpected_estimator
    p0, _, _ = WorkflowRunner()._extract_fit_params(
        {"fitting": {"p0": [2.0]}}, np.arange(3.0), np.arange(3.0), model
    )
    np.testing.assert_array_equal(p0, [2.0])
