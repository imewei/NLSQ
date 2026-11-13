"""NLSQ_SAVE_ITERATIONS saves one history file per fit at the default log level."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _fit(bounds):
    x = np.linspace(0, 4, 40)
    y = 2.0 * np.exp(-1.3 * x) + 0.01 * np.cos(7 * x)
    kwargs = {} if bounds is None else {"bounds": bounds}
    return curve_fit(_model, x, y, p0=[1.0, 1.0], **kwargs)


@pytest.mark.parametrize("bounds", [None, ([0.0, 0.0], [10.0, 10.0])])
def test_fit_saves_iteration_history(tmp_path, monkeypatch, bounds):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    _fit(bounds)
    files = sorted(tmp_path.glob("optimization_history_*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as data:
        assert {"iter", "cost"} <= set(data.files)
        assert len(data["iter"]) >= 2
        assert data["cost"][-1] <= data["cost"][0]

    _fit(bounds)  # a second fit writes its own file, not the accumulated history
    files2 = sorted(tmp_path.glob("optimization_history_*.npz"))
    assert len(files2) == 2
    sizes = []
    for f in files2:
        with np.load(f) as data:
            sizes.append(len(data["iter"]))
    assert sizes[0] == sizes[1]


def test_nothing_saved_when_unset_or_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("NLSQ_SAVE_ITERATIONS", raising=False)
    monkeypatch.chdir(tmp_path)
    _fit(None)
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", "0")
    _fit(None)
    assert not list(tmp_path.rglob("*.npz"))
