"""Round 8: hybrid legacy-callback rejection and telemetry rates in 0-100."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.telemetry import (
    DefenseLayerTelemetry,
    get_defense_telemetry,
    reset_defense_telemetry,
)


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    x = jnp.linspace(0, 4, 200)
    y = _model(x, 2.0, 0.7) + 0.01 * jnp.sin(7 * x)
    return x, y


def _optimizer():
    return AdaptiveHybridStreamingOptimizer(
        HybridStreamingConfig(
            warmup_iterations=5,
            max_warmup_iterations=15,
            gauss_newton_max_iterations=5,
            enable_warm_start_detection=False,
            verbose=0,
        )
    )


# ---------------------------------------------------------------- callbacks
def test_legacy_three_arg_callback_rejected_up_front():
    x, y = _data()
    with pytest.raises(TypeError, match=r"\(iteration, cost, params, info\)"):
        _optimizer().fit(
            (x, y), _model, jnp.array([1.0, 1.0]), callback=lambda p, loss, it: None
        )


def test_four_arg_and_keyword_callbacks_still_called():
    x, y = _data()
    calls = []

    def positional(iteration, cost, params, info):
        calls.append(("pos", iteration))

    def keyword(*, iteration, cost, params, info):
        calls.append(("kw", iteration))

    for cb in (positional, keyword):
        _optimizer().fit((x, y), _model, jnp.array([1.0, 1.0]), callback=cb, verbose=0)
    assert {c[0] for c in calls} == {"pos", "kw"}


def test_star_args_callback_accepted():
    x, y = _data()
    seen = []
    _optimizer().fit(
        (x, y), _model, jnp.array([1.0, 1.0]), callback=lambda *a: seen.append(a)
    )
    assert seen


# --------------------------------------------------------------- telemetry
def test_rates_with_synthetic_counts_are_percentages():
    t = DefenseLayerTelemetry()
    for _ in range(2):
        t.record_warmup_start()
    for _ in range(30):
        t.record_lbfgs_iteration()
    for _ in range(15):
        t.record_layer4_clip(2.0, 1.0)
    t.record_layer2_lr_mode("careful", 0.5)
    t.record_layer2_lr_mode("refinement", 0.05)
    t.record_lbfgs_history_fill(10)
    # The same step reported by both detectors counts once in the rate.
    t.record_lbfgs_line_search_failure(3, "exc")
    t.record_lbfgs_line_search_failure(3, "no decrease")
    t.record_lbfgs_line_search_failure(4, "no decrease")
    r = t.get_trigger_rates()
    assert r["layer4_clip_rate"] == pytest.approx(50.0)
    assert r["lbfgs_line_search_failure_rate"] == pytest.approx(100.0 * 2 / 30)
    assert r["layer2_careful_rate"] == pytest.approx(50.0)
    assert r["lbfgs_history_buffer_fill_rate"] == pytest.approx(50.0)
    assert t.lbfgs_line_search_failures == 3
    assert all(0.0 <= v <= 100.0 for v in r.values())


def test_rates_zero_when_nothing_recorded():
    assert set(DefenseLayerTelemetry().get_trigger_rates().values()) == {0.0}


def test_rates_in_range_after_real_hybrid_fit():
    reset_defense_telemetry()
    x, y = _data()
    _optimizer().fit((x, y), _model, jnp.array([1.0, 1.0]), verbose=0)
    t = get_defense_telemetry()
    assert t.total_lbfgs_iterations > 0
    rates = t.get_trigger_rates()
    assert all(0.0 <= v <= 100.0 for v in rates.values()), rates


def test_rates_in_range_after_curve_fit_hybrid_streaming():
    reset_defense_telemetry()
    x = np.linspace(0, 4, 300)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * np.sin(7 * x)
    curve_fit(_model, x, y, p0=[1.0, 1.0], method="hybrid_streaming")
    rates = get_defense_telemetry().get_trigger_rates()
    assert all(0.0 <= v <= 100.0 for v in rates.values()), rates
