"""OptimizationRecovery proposes only methods the solver accepts."""

import contextlib
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit
from nlsq.stability.recovery import OptimizationRecovery

B = (np.array([0.0, -300.0]), np.array([10.0, 10.0]))


def _methods(state):
    rec = OptimizationRecovery(max_retries=3)
    return [
        rec._switch_algorithm("optimization_error", state, r).get("method")
        for r in range(3)
    ]


def test_never_proposes_dogbox():
    assert "dogbox" not in _methods({"method": "trf", "x": np.ones(2)})
    assert "dogbox" not in _methods({"method": "lm", "x": np.ones(2)})


def test_bounded_trf_is_not_switched_to_lm():
    assert _methods({"method": "trf", "x": np.ones(2), "bounds": B}) == ["trf"] * 3
    inf = (-np.inf, np.inf)
    assert _methods({"method": "trf", "x": np.ones(2), "bounds": inf})[0] == "lm"


def test_unbounded_trf_to_lm_and_lm_to_trf():
    assert _methods({"method": "trf", "x": np.ones(2)})[0] == "lm"
    assert _methods({"method": "lm", "x": np.ones(2)})[0] == "trf"


def test_unchanged_proposals_are_not_retried():
    calls = []

    def opt(**state):
        calls.append(state)
        return {"success": False, "cost": np.inf}

    rec = OptimizationRecovery(max_retries=1)
    ok, _ = rec.recover_from_failure(
        "optimization_error", {"method": "trf", "x": np.ones(2), "bounds": B}, opt
    )
    assert not ok
    # _reformulate_problem proposes nothing and a bounded trf has no switch
    # target, so those strategies must not burn a solver call each.
    assert all(c != {"method": "trf", "x": c["x"], "bounds": B} for c in calls)


def test_bounded_fit_with_recovery_emits_no_strategy_failed_warnings():
    x = np.linspace(0, 5, 30)
    y = 2.0 * np.exp(-0.5 * x)
    cf = CurveFit(enable_recovery=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):
            cf.curve_fit(
                lambda x, a, b: a * jnp.exp(-b * x),
                x,
                y,
                p0=[1, -250.0],
                bounds=([0, -300], [10, 10]),
            )
    msgs = [str(w.message) for w in rec if "_switch_algorithm failed" in str(w.message)]
    assert not msgs, msgs
