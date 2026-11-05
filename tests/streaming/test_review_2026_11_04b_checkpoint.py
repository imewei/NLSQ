"""Regression tests for the 2026-11-04 checkpoint review findings."""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState
from nlsq.utils.safe_serialize import SafeSerializationError

jax.config.update("jax_enable_x64", True)

P = jnp.array([1.0, 2.0])


def _state(opt_state=None, **kw):
    base = {
        "current_phase": 1,
        "normalized_params": P,
        "phase1_optimizer_state": opt_state,
        "phase2_JTJ_accumulator": None,
        "phase2_JTr_accumulator": None,
        "best_params_global": P,
        "best_cost_global": 1.0,
        "phase_history": [{"k": 1}],
        "normalizer": None,
        "tournament_selector": None,
        "multistart_candidates": jnp.ones((3, 2)),
    }
    base.update(kw)
    return CheckpointState(**base)


@pytest.mark.serial
def test_clipped_chain_state_is_saved(tmp_path):
    """L-BFGS state is found by type even behind a gradient-clip element."""
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.lbfgs())
    mgr = CheckpointManager(HybridStreamingConfig())
    path = tmp_path / "c.h5"
    mgr.save(path, _state(opt.init(P)))
    with h5py.File(path, "r") as f:
        assert "params" in f["phase_state/phase1_optimizer_state"]
    assert mgr.load(path).phase1_optimizer_state is not None


@pytest.mark.serial
def test_checkpoint_failure_does_not_degrade_fit(tmp_path, monkeypatch):
    """A failing save is logged and the fit result is unaffected."""
    monkeypatch.setattr(
        CheckpointManager,
        "save",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    x = np.linspace(0, 5, 2000)
    y = 2 * np.exp(-0.7 * x) + 0.01 * np.random.default_rng(0).normal(size=x.size)
    cfg = HybridStreamingConfig(
        enable_checkpoints=True,
        checkpoint_dir=str(tmp_path),
        checkpoint_frequency=1,
        gradient_clip_value=1.0,
    )
    r = AdaptiveHybridStreamingOptimizer(cfg).fit(
        (x, y), lambda x, a, b: a * jnp.exp(-b * x), jnp.array([1.0, 0.3]), verbose=0
    )
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0, 0.7], atol=0.05)


def test_failed_save_keeps_previous_checkpoint(tmp_path):
    mgr = CheckpointManager(HybridStreamingConfig())
    path = tmp_path / "b.h5"
    mgr.save(path, _state())
    bad = _state(phase_history=[{"labels": np.array(["x"])}])
    with pytest.raises(SafeSerializationError):
        mgr.save(path, bad)
    ld = mgr.load(path)
    assert ld.phase_history == [{"k": 1}]
    assert ld.multistart_candidates is not None
    assert not list(tmp_path.glob("*.tmp"))


def test_torn_file_without_marker_is_rejected(tmp_path):
    mgr = CheckpointManager(HybridStreamingConfig())
    path = tmp_path / "t.h5"
    mgr.save(path, _state())
    with h5py.File(path, "a") as f:
        del f.attrs["completion_marker"]
    with pytest.raises(ValueError, match="incomplete"):
        mgr.load(path)
    with h5py.File(path, "a") as f:  # legacy 3.0 files have no marker
        f.attrs["version"] = "3.0"
    assert mgr.load(path).current_phase == 1
