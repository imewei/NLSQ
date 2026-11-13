"""S2-1 / S2-2: group-variance ranges are bounds-checked; GN baseline cost
includes the group-variance term."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
from nlsq.streaming.phases.warmup import WarmupPhase
from nlsq.streaming.validators import validate_group_variance_bounds

P0 = jnp.array([0.0, 10.0, 10.0])  # 3 parameters; group (1, 5) overruns


def _model(x, a, b, c):
    return a + b * x + c * x**2


def _config(indices=((1, 5),), enabled=True, **kw):
    return HybridStreamingConfig(
        normalize=False,
        enable_group_variance_regularization=enabled,
        group_variance_indices=list(indices),
        group_variance_lambda=1.0,
        verbose=0,
        **kw,
    )


def _data():
    x = np.linspace(0.0, 1.0, 32)
    return x, _model(x, 0.0, 1.0, 1.0)


def test_helper_rejects_end_past_n_params():
    with pytest.raises(ValueError, match=r"group_variance_indices\[0\].*end \(5\)"):
        validate_group_variance_bounds(_config(), 3)


def test_helper_accepts_in_range_and_disabled():
    validate_group_variance_bounds(_config(((0, 3),)), 3)
    validate_group_variance_bounds(_config(enabled=False), 3)


def test_warmup_phase_rejects_out_of_range_group():
    phase = WarmupPhase(_config(), MagicMock())
    with pytest.raises(ValueError, match="exceeds the number of parameters"):
        phase.run(_data(), P0, [], {})


def test_gauss_newton_phase_rejects_out_of_range_group():
    phase = GaussNewtonPhase(_config(), MagicMock())
    with pytest.raises(ValueError, match="exceeds the number of parameters"):
        phase.run(_data(), P0, [], {})


def test_production_phase1_rejects_out_of_range_group():
    opt = AdaptiveHybridStreamingOptimizer(_config())
    with pytest.raises(ValueError, match="exceeds the number of parameters"):
        opt._run_phase1_warmup(_data(), _model, P0, None)


def test_production_phase2_rejects_out_of_range_group():
    opt = AdaptiveHybridStreamingOptimizer(_config())
    opt._setup_normalization(_model, P0, None)
    with pytest.raises(ValueError, match="exceeds the number of parameters"):
        opt._run_phase2_gauss_newton(_data(), opt.normalized_params)


def test_gn_baseline_cost_includes_group_variance_term(monkeypatch):
    """The first rejected step must be compared against the regularized cost.

    Raw SSR from the JTJ pass is 150 and the trial cost is also 150, which
    the old code read as zero change ("converged"). The regularized baseline
    is 100, so the step is a real 50% worsening and the phase keeps going.
    """
    phase = GaussNewtonPhase(
        _config(((0, 2),), gauss_newton_max_iterations=3), MagicMock()
    )
    n = 3
    monkeypatch.setattr(
        phase,
        "_accumulate_jtj_jtr",
        lambda *a, **k: (jnp.zeros((n, n)), jnp.zeros(n), 150.0),
    )
    monkeypatch.setattr(phase, "_compute_cost", lambda *a, **k: 100.0)
    monkeypatch.setattr(
        phase,
        "_gauss_newton_iteration",
        lambda *a, **k: {
            "new_params": P0,
            "new_cost": 150.0,
            "gradient_norm": 1.0,
            "actual_reduction": -50.0,
            "trust_radius": 1.0,
        },
    )
    result = phase.run(_data(), P0, [], {})
    assert result["convergence_reason"] != "Cost change below tolerance"
    assert result["iterations"] == 3
