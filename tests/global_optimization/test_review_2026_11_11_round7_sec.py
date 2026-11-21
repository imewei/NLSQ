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


def _tamper_external_link(tmp_path, state):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as o:
        o["x"] = np.zeros(3)
    del state["mean"]
    state["mean"] = h5py.ExternalLink(str(other), "/x")


def _tamper_external_storage(tmp_path, state):
    raw = tmp_path / "raw.bin"
    raw.write_bytes(bytes(24))
    n = state["mean"].shape[0]
    del state["mean"]
    state.create_dataset(
        "mean", shape=(n,), dtype="<f8", external=[(str(raw), 0, 8 * n)]
    )


def _tamper_virtual(tmp_path, state):
    src = tmp_path / "src.h5"
    n = state["mean"].shape[0]
    with h5py.File(src, "w") as o:
        o["x"] = np.zeros(n)
    layout = h5py.VirtualLayout(shape=(n,), dtype="<f8")
    layout[:] = h5py.VirtualSource(str(src), "x", shape=(n,))
    del state["mean"]
    state.create_virtual_dataset("mean", layout)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (_tamper_external_link, "external link"),
        (_tamper_external_storage, "external or virtual storage"),
        (_tamper_virtual, "external or virtual storage"),
    ],
)
def test_cmaes_load_rejects_data_stored_outside_the_file(
    saved, tmp_path, tamper, message
):
    mgr, fp, path = saved
    with h5py.File(path, "a") as f:
        tamper(tmp_path, f["state"])
    with pytest.raises(ValueError, match=message):
        mgr.load(path, fp)


def test_cmaes_foreign_storage_is_rejected_before_other_checks(saved, tmp_path):
    # opening f["state"] already follows a link, so the guard must come first
    mgr, fp, path = saved
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as o:
        o.create_group("g")
    with h5py.File(path, "a") as f:
        del f["state"]
        f["state"] = h5py.ExternalLink(str(other), "/g")
        del f.attrs["completion_marker"]
    with pytest.raises(ValueError, match="external link"):
        mgr.load(path, fp)
