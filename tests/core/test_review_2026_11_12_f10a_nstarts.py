"""nlsq.fit(workflow=...) forwards n_starts=None; it must mean "use the default"."""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq


def _f(x, a, w, c):
    return a * jnp.sin(w * x) + c


def _data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 2000)
    y = np.asarray(_f(x, 2.0, 3.3, 0.2)) + 0.05 * rng.standard_normal(x.size)
    b = (np.array([0.0, 0.1, -3.0]), np.array([5.0, 8.0, 3.0]))
    return x, y, b, np.array([2.5, 1.2, 0.0])


def _ssr(x, y, r):
    return float(np.sum((np.asarray(_f(x, *np.asarray(r[0]))) - y) ** 2))


@pytest.mark.parametrize("workflow", ["auto_global", "hpc"])
def test_workflow_default_n_starts_is_ten_not_one(workflow):
    x, y, b, p0 = _data()
    r = nlsq.fit(_f, x, y, p0=p0, bounds=b, workflow=workflow)
    diag = r.get("multistart_diagnostics")
    assert diag is not None
    assert diag["n_starts_configured"] == 10
    assert not diag["bypassed"]
    # noise floor: 2000 * 0.05**2 = 5.0
    assert _ssr(x, y, r) < 6.0
