"""Round-7: streaming checkpoint loader validates shape/size/sparsity."""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState

jax.config.update("jax_enable_x64", True)

P = jnp.array([1.0, 2.0])


def _state(**kw):
    base = {
        "current_phase": 2,
        "normalized_params": P,
        "phase1_optimizer_state": optax.lbfgs().init(P),
        "phase2_JTJ_accumulator": jnp.eye(2),
        "phase2_JTr_accumulator": jnp.zeros(2),
        "best_params_global": P,
        "best_cost_global": 1.0,
        "phase_history": [{"k": 1}],
        "normalizer": None,
        "tournament_selector": None,
        "multistart_candidates": jnp.ones((3, 2)),
    }
    base.update(kw)
    return CheckpointState(**base)


@pytest.fixture
def saved(tmp_path):
    mgr = CheckpointManager(HybridStreamingConfig())
    path = tmp_path / "c.h5"
    mgr.save(path, _state())
    return mgr, path


def _replace(path, name, first_rows=None, **kw):
    with h5py.File(path, "r+") as f:
        g = f["phase_state"]
        del g[name]
        d = g.create_dataset(name, **kw)
        if first_rows is not None:
            d[: len(first_rows)] = first_rows


def test_valid_round_trip_unaffected(saved):
    mgr, path = saved
    st = mgr.load(path)
    assert st.phase2_JTJ_accumulator.shape == (2, 2)
    assert st.multistart_candidates.shape == (3, 2)
    assert st.phase1_optimizer_state is not None


@pytest.mark.serial
def test_sparse_jtj_bomb_rejected(saved):
    mgr, path = saved
    _replace(
        path,
        "phase2_jtj_accumulator",
        first_rows=np.ones(1024),
        shape=(100_000_000,),
        dtype="f8",
        chunks=(1024,),
        compression="gzip",
    )
    with pytest.raises(ValueError):
        mgr.load(path)


def test_one_d_jtj_rejected(saved):
    mgr, path = saved
    _replace(path, "phase2_jtj_accumulator", data=np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        mgr.load(path)


def test_jtr_length_mismatch_rejected(saved):
    mgr, path = saved
    _replace(path, "phase2_jtr_accumulator", data=np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        mgr.load(path)


def test_non_numeric_dtype_rejected(saved):
    mgr, path = saved
    _replace(path, "best_params_global", data=np.array([b"a", b"b"]))
    with pytest.raises(ValueError, match="dtype"):
        mgr.load(path)


def test_sparse_small_chunks_rejected(saved):
    mgr, path = saved
    _replace(
        path,
        "multistart_candidates",
        first_rows=[[1.0, 1.0]],
        shape=(4, 2),
        dtype="f8",
        chunks=(1, 2),
    )
    with pytest.raises(ValueError, match="unallocated"):
        mgr.load(path)
