"""Round-21 regressions: silent failure paths that must be loud."""

import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit, curve_fit_large
from nlsq.global_optimization import CMAESConfig


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def test_chunked_fit_nan_policy_raise_overrides_check_finite_false():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 3000)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
    y[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        curve_fit_large(
            _model,
            x,
            y,
            p0=[1, 0.1],
            size_threshold=1000,
            chunk_size=1000,
            nan_policy="raise",
            check_finite=False,
        )


def test_cmaes_without_refine_reports_failure_when_never_finite():
    x = np.linspace(0, 5, 300)
    y = 2.0 * np.exp(-0.7 * x)

    def mnan(x, a, b):
        return jnp.sqrt(-a - 1.0) * jnp.exp(-b * x)

    cfg = CMAESConfig(max_generations=5, refine_with_nlsq=False, seed=0)
    r = curve_fit(
        mnan,
        x,
        y,
        p0=[1, 1],
        bounds=([0, 0], [10, 10]),
        method="cmaes",
        cmaes_config=cfg,
    )
    assert not r.success


@pytest.mark.serial
def test_multistart_warns_when_winner_did_not_converge():
    # Subprocess: under pytest the per-start warning filter leaks, so the
    # in-process warning list cannot tell the winner's warning apart.
    code = """
import warnings
import numpy as np, jax.numpy as jnp
from nlsq import curve_fit
rng = np.random.default_rng(0)
x = np.linspace(0, 5, 300)
y = 2.0 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
m = lambda x, a, b: a * jnp.exp(-b * x)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = curve_fit(m, x, y, p0=[1, 1], bounds=([0, 0], [10, 10]), max_nfev=2,
                  multistart=True, n_starts=4)
n = sum("OptimizeWarning" in i.category.__name__ for i in w)
print("RESULT", r.success, n)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("RESULT"))
    assert line.split()[1] == "False"
    assert int(line.split()[2]) >= 1
