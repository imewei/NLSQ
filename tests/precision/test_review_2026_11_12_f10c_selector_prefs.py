"""Explicit user preferences must not be loosened by data-driven adjustments."""

import numpy as np

from nlsq.precision.algorithm_selector import AlgorithmSelector


def _analysis(**kw):
    base = {"n_points": 200, "n_params": 2}
    base.update(kw)
    return base


def test_accuracy_preference_survives_noise_and_conditioning():
    sel = AlgorithmSelector()
    for extra in ({"is_noisy": True}, {"condition_estimate": 1e12}):
        rec = sel.select_algorithm(_analysis(**extra), {"prioritize": "accuracy"})
        assert (rec["ftol"], rec["xtol"], rec["gtol"]) == (1e-10, 1e-10, 1e-10)


def test_no_preference_still_relaxes_for_noise():
    rec = AlgorithmSelector().select_algorithm(_analysis(is_noisy=True), None)
    assert rec["ftol"] == 1e-6 and rec["xtol"] == 1e-6


def test_speed_preference_unchanged():
    rec = AlgorithmSelector().select_algorithm(
        _analysis(n_points=200000), {"prioritize": "speed"}
    )
    assert rec["ftol"] == 1e-6 and rec["max_nfev"] == 100


def test_auto_select_algorithm_accuracy_on_noisy_data():
    from nlsq.precision.algorithm_selector import auto_select_algorithm

    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    y = 0.1 * x + rng.normal(size=200)
    rec = auto_select_algorithm(
        lambda x, a, b: a * x + b,
        x,
        y,
        p0=[1, 1],
        user_preferences={"prioritize": "accuracy"},
    )
    assert rec["ftol"] == 1e-10 and rec["xtol"] == 1e-10 and rec["gtol"] == 1e-10
