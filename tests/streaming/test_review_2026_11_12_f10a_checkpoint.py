"""Tournament checkpoints (bool survival_mask) must round-trip through load()."""

import numpy as np

from nlsq.global_optimization import GlobalOptimizationConfig
from nlsq.global_optimization.tournament import TournamentSelector
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState


def _state(selector):
    return CheckpointState(
        current_phase=1,
        normalized_params=np.ones(2),
        phase1_optimizer_state=None,
        phase2_JTJ_accumulator=np.eye(2),
        phase2_JTr_accumulator=np.ones(2),
        best_params_global=np.ones(2),
        best_cost_global=1.0,
        phase_history=[{"phase": 0}],
        normalizer=None,
        tournament_selector=selector,
        multistart_candidates=np.ones((4, 2)),
    )


def test_tournament_checkpoint_roundtrip(tmp_path):
    cfg = GlobalOptimizationConfig(n_starts=4)
    sel = TournamentSelector(np.random.default_rng(0).normal(size=(4, 2)), cfg)
    sel.survival_mask[1] = False  # exercise a non-trivial bool mask
    mgr = CheckpointManager(HybridStreamingConfig(enable_multistart=True))
    path = tmp_path / "ck.h5"
    mgr.save(path, _state(sel))

    out = mgr.load(path, cfg)

    assert out.tournament_selector is not None
    assert out.tournament_selector.survival_mask.dtype == bool
    np.testing.assert_array_equal(
        out.tournament_selector.survival_mask, sel.survival_mask
    )
    np.testing.assert_array_equal(out.tournament_selector.candidates, sel.candidates)
