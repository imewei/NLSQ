"""CMA-ES refinement failure must not be reported as a successful fit."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.global_optimization import CMAESConfig
from nlsq.global_optimization.cmaes_config import is_evosax_available

pytestmark = pytest.mark.skipif(
    not is_evosax_available(), reason="evosax not installed"
)


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _bad_jac(x, a, b):
    raise RuntimeError("jac boom")


@pytest.fixture
def data():
    x = np.linspace(0, 4, 100)
    return x, 2.5 * np.exp(-1.3 * x)


def _cmaes_kwargs():
    return {
        "p0": [1.0, 1.0],
        "bounds": ([0.0, 0.0], [10.0, 10.0]),
        "method": "cmaes",
        "cmaes_config": CMAESConfig(max_generations=10, popsize=8, seed=0),
    }


def test_refinement_failure_reported_as_failure(data):
    x, y = data
    # CMA-ES ignores jac; only the NLSQ refinement step uses it and raises.
    res = curve_fit(_model, x, y, jac=_bad_jac, **_cmaes_kwargs())
    assert res.success is False
    assert res.status == 0
    assert "refinement failed" in res.message

    out = curve_fit(_model, x, y, jac=_bad_jac, full_output=True, **_cmaes_kwargs())
    assert out[4] != 1
    assert "refinement failed" in out[3]


def test_normal_cmaes_fit_emits_no_unknown_kwarg_warning(data):
    x, y = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = curve_fit(_model, x, y, **_cmaes_kwargs())
    assert res.success
    assert not [w for w in caught if "unknown keyword" in str(w.message)]
