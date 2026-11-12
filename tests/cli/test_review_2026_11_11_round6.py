"""Round-6 regressions: batch stdout redirect race, export/formats validation, Hybrid verbose."""

import pathlib
import sys

import numpy as np
import pytest
import yaml

from nlsq.cli.errors import ConfigError
from nlsq.cli.visualization import normalize_formats


def _workflow(tmp_path, name, **export):
    x = np.linspace(0, 4, 40)
    data = tmp_path / f"{name}.csv"
    np.savetxt(data, np.c_[x, 2 * np.exp(-1.3 * x)], delimiter=",")
    cfg = {
        "data": {"input_file": str(data), "format": "csv", "columns": {"x": 0, "y": 1}},
        "model": {"type": "builtin", "name": "exponential_decay"},
        "export": export,
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_batch_stdout_workflows_do_not_leak_redirect(tmp_path):
    from nlsq.cli.commands.batch import run_batch

    paths = [
        _workflow(tmp_path, f"w{i}", stdout=True, skip_file_on_stdout=True)
        for i in range(4)
    ]
    orig = sys.stdout
    try:
        run_batch(paths, max_workers=4)
        leaked = sys.stdout is not orig
    finally:
        sys.stdout = orig
    assert not leaked


def test_null_export_fails_fast(tmp_path):
    from nlsq.cli.commands.fit import run_fit

    path = _workflow(tmp_path, "n")
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    cfg["export"] = None
    pathlib.Path(path).write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError):
        run_fit(path)


def test_non_list_formats_is_config_error():
    with pytest.raises(ConfigError):
        normalize_formats(5)
    assert normalize_formats("png") == ["png"]


def test_hybrid_config_verbose_is_honoured(capsys):
    import jax.numpy as jnp

    from nlsq import HybridStreamingConfig, curve_fit_large
    from nlsq.streaming.specs import Hybrid

    x = np.linspace(0, 4, 2000)
    y = 2 * np.exp(-1.3 * x)

    def m(x, a, b):
        return a * jnp.exp(-b * x)

    curve_fit_large(
        m, x, y, p0=[1.0, 1.0], strategy=Hybrid(config=HybridStreamingConfig(verbose=2))
    )
    loud = capsys.readouterr().out
    curve_fit_large(m, x, y, p0=[1.0, 1.0], strategy=Hybrid())
    quiet = capsys.readouterr().out
    assert loud and not quiet
