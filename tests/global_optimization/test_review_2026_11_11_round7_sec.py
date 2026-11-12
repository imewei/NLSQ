"""Round-7: CMA-ES checkpoint loader rejects sparse/oversized/misshapen datasets."""

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.checkpoint import (
    CMAESCheckpointState,
    HPCCheckpointManager,
    compute_fingerprint,
)


def _state():
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
        fitness_history=[1.0],
        popsize=8,
    )


def _fp():
    return compute_fingerprint(
        model_id="m",
        xdata=np.linspace(0, 1, 10),
        ydata=np.linspace(0, 1, 10),
        sigma=None,
        bounds=(np.zeros(3), np.ones(3)),
        config_fields={
            "popsize": 8,
            "sigma": 0.5,
            "tol_fun": 1e-8,
            "tol_x": 1e-8,
            "seed": 1,
        },
    )


def _sparse(f, name, shape, chunks):
    del f["state"][name]
    d = f["state"].create_dataset(
        name, shape=shape, dtype="f8", chunks=chunks, compression="gzip"
    )
    d[tuple(slice(0, c) for c in chunks)] = 1.0


@pytest.fixture
def saved(tmp_path):
    mgr, fp, path = HPCCheckpointManager(), _fp(), tmp_path / "c.h5"
    mgr.save(path, _state(), fp)
    return mgr, fp, path


def test_valid_round_trip_unaffected(saved):
    mgr, fp, path = saved
    assert mgr.load(path, fp).generation_counter == 5


@pytest.mark.serial
def test_sparse_bomb_on_mean_rejected(saved):
    mgr, fp, path = saved
    with h5py.File(path, "r+") as f:
        _sparse(f, "mean", (100_000_000,), (1024,))
    with pytest.raises(ValueError, match=r"exceeds|unallocated"):
        mgr.load(path, fp)


@pytest.mark.serial
def test_sparse_chunks_small_shape_rejected(saved):
    mgr, fp, path = saved
    with h5py.File(path, "r+") as f:
        del f["state"]["best_solution"]
        d = f["state"].create_dataset(
            "best_solution", shape=(3,), dtype="f8", chunks=(1,), compression="gzip"
        )
        d[0] = 1.0  # 1 of 3 chunks written
    with pytest.raises(ValueError, match="unallocated"):
        mgr.load(path, fp)


def test_wrong_shape_rejected(saved):
    mgr, fp, path = saved
    with h5py.File(path, "r+") as f:
        del f["state"]["C"]
        f["state"].create_dataset("C", data=np.zeros(9))  # 1-D where (n, n) expected
    with pytest.raises(ValueError, match="shape"):
        mgr.load(path, fp)
