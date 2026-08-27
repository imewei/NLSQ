import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.checkpoint import (
    CMAESCheckpointState,
    deserialize_evosax_state,
    deserialize_key,
    serialize_evosax_state,
    serialize_key,
)


def _make_evosax_state():
    from evosax.algorithms import CMA_ES

    es = CMA_ES(population_size=8, solution=jnp.zeros(3))
    params = es.default_params
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    return es.init(subkey, jnp.zeros(3), params)


def test_key_round_trip():
    key = jax.random.key(42)
    data = serialize_key(key)
    assert data.dtype == np.uint32
    restored = deserialize_key(data)
    assert bool((jax.random.key_data(restored) == jax.random.key_data(key)).all())


def test_evosax_state_round_trip():
    state = _make_evosax_state()
    d = serialize_evosax_state(state)
    restored = deserialize_evosax_state(d, state)
    assert restored.generation_counter == state.generation_counter
    np.testing.assert_array_equal(np.asarray(restored.mean), np.asarray(state.mean))
    np.testing.assert_array_equal(np.asarray(restored.C), np.asarray(state.C))


def test_checkpoint_state_construction():
    state = CMAESCheckpointState(
        generation_counter=5,
        mean=jnp.zeros(3),
        std=jnp.array(0.5),
        p_std=jnp.zeros(3),
        p_c=jnp.zeros(3),
        C=jnp.eye(3),
        B=jnp.eye(3),
        D=jnp.ones(3),
        best_solution=jnp.zeros(3),
        best_fitness=1.0,
        key_data=np.array([0, 0], dtype=np.uint32),
        fitness_history=[1.0, 0.5],
        popsize=8,
    )
    assert state.generation_counter == 5
    assert state.popsize == 8


import hashlib
from pathlib import Path

from nlsq.global_optimization.checkpoint import (
    HPCCheckpointManager,
    compute_fingerprint,
)


def _sample_state():
    return CMAESCheckpointState(
        generation_counter=5,
        mean=jnp.zeros(3),
        std=jnp.array(0.5),
        p_std=jnp.zeros(3),
        p_c=jnp.zeros(3),
        C=jnp.eye(3),
        B=jnp.eye(3),
        D=jnp.ones(3),
        best_solution=jnp.array([1.0, 2.0, 3.0]),
        best_fitness=0.01,
        key_data=np.array([1, 2], dtype=np.uint32),
        fitness_history=[1.0, 0.5, 0.2, 0.05, 0.01],
        popsize=8,
    )


def _sample_fingerprint():
    return compute_fingerprint(
        model_id="test_model_v1",
        xdata=np.linspace(0, 1, 10),
        ydata=np.linspace(0, 1, 10),
        sigma=None,
        bounds=(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
        config_fields={
            "popsize": 8,
            "sigma": 0.5,
            "tol_fun": 1e-8,
            "tol_x": 1e-8,
            "seed": 1,
        },
    )


def test_save_load_round_trip(tmp_path):
    manager = HPCCheckpointManager()
    fp = _sample_fingerprint()
    path = tmp_path / "run1.h5"
    manager.save(path, _sample_state(), fp)
    restored = manager.load(path, fp)
    assert restored.generation_counter == 5
    assert restored.fitness_history == [1.0, 0.5, 0.2, 0.05, 0.01]
    np.testing.assert_array_equal(np.asarray(restored.best_solution), [1.0, 2.0, 3.0])


def test_load_missing_file_raises(tmp_path):
    manager = HPCCheckpointManager()
    with pytest.raises(FileNotFoundError):
        manager.load(tmp_path / "missing.h5", _sample_fingerprint())


def test_load_fingerprint_mismatch_raises(tmp_path):
    manager = HPCCheckpointManager()
    fp = _sample_fingerprint()
    path = tmp_path / "run1.h5"
    manager.save(path, _sample_state(), fp)
    bad_fp = dict(fp)
    bad_fp["data_hash"] = "deadbeef" * 8
    with pytest.raises(ValueError, match="fingerprint"):
        manager.load(path, bad_fp)


def test_save_is_atomic_leaves_no_tmp_file(tmp_path):
    manager = HPCCheckpointManager()
    path = tmp_path / "run1.h5"
    manager.save(path, _sample_state(), _sample_fingerprint())
    leftover_tmp = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp == []
    assert path.exists()


def test_save_rotates_previous_to_bak(tmp_path):
    manager = HPCCheckpointManager()
    fp = _sample_fingerprint()
    path = tmp_path / "run1.h5"
    state1 = _sample_state()
    manager.save(path, state1, fp)
    state2 = _sample_state()
    state2.generation_counter = 10
    manager.save(path, state2, fp)
    assert (tmp_path / "run1.h5.bak").exists()
    restored_bak = manager.load(tmp_path / "run1.h5.bak", fp)
    assert restored_bak.generation_counter == 5
    restored_current = manager.load(path, fp)
    assert restored_current.generation_counter == 10


def test_load_falls_back_to_bak_when_primary_is_corrupt(tmp_path):
    """FR8: a torn/corrupt primary must NOT be a load failure as long as
    the rotated .bak (written by the previous successful save) is intact
    -- save() rotates specifically so this fallback always has something
    to recover from."""
    manager = HPCCheckpointManager()
    fp = _sample_fingerprint()
    path = tmp_path / "run1.h5"
    state1 = _sample_state()
    manager.save(path, state1, fp)  # only save -- path has no .bak yet
    bak_path = tmp_path / "run1.h5.bak"
    # Simulate a second, torn save: rotate the good primary to .bak by hand
    # (mirroring what save() does), then corrupt what would be the new
    # primary, since we can't easily interrupt save() mid-write here.
    path.replace(bak_path)
    path.write_bytes(b"not a valid hdf5 file")

    restored = manager.load(path, fp)
    assert restored.generation_counter == 5  # recovered from .bak


def test_fingerprint_includes_actual_bounds_arrays(tmp_path):
    """Spec section 6 requires the fingerprint group to store the real
    bounds arrays for auditability, not only a hash -- a hash alone can't
    be inspected by a human debugging a mismatch."""
    manager = HPCCheckpointManager()
    fp = _sample_fingerprint()
    path = tmp_path / "run1.h5"
    manager.save(path, _sample_state(), fp)
    import h5py

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f["fingerprint"]["bounds_lower"][()],
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            f["fingerprint"]["bounds_upper"][()],
            [10.0, 10.0, 10.0],
        )
