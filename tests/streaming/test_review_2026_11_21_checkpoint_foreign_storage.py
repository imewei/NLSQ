"""Checkpoints are untrusted input: data must live inside the file itself."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import (
    CheckpointManager,
    CheckpointState,
    reject_foreign_storage,
)


def _saved_checkpoint(tmp_path):
    state = CheckpointState(
        current_phase=1,
        normalized_params=np.zeros(2),
        phase1_optimizer_state=None,
        phase2_JTJ_accumulator=None,
        phase2_JTr_accumulator=None,
        best_params_global=np.zeros(2),
        best_cost_global=1.0,
        phase_history=[],
        normalizer=None,
        tournament_selector=None,
        multistart_candidates=None,
    )
    manager = CheckpointManager(HybridStreamingConfig())
    path = tmp_path / "ck.h5"
    manager.save(path, state)
    return manager, path


def _external_storage(tmp_path, group):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"TOPSECRETKEY1234")  # two float64
    del group["best_params_global"]
    group.create_dataset(
        "best_params_global", shape=(2,), dtype="<f8", external=[(str(secret), 0, 16)]
    )


def _external_link(tmp_path, group):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as o:
        o["x"] = np.array([7.0, 8.0])
    del group["normalized_params"]
    group["normalized_params"] = h5py.ExternalLink(str(other), "/x")


def _virtual(tmp_path, group):
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as o:
        o["x"] = np.array([7.0, 8.0])
    layout = h5py.VirtualLayout(shape=(2,), dtype="<f8")
    layout[:] = h5py.VirtualSource(str(src), "x", shape=(2,))
    del group["best_params_global"]
    group.create_virtual_dataset("best_params_global", layout)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (_external_storage, "external or virtual storage"),
        (_external_link, "external link"),
        (_virtual, "external or virtual storage"),
    ],
)
def test_checkpoint_load_rejects_data_stored_outside_the_file(
    tmp_path, tamper, message
):
    manager, path = _saved_checkpoint(tmp_path)
    with h5py.File(path, "a") as f:
        tamper(tmp_path, f["phase_state"])
    with pytest.raises(ValueError, match=message):
        manager.load(path)


def test_untampered_checkpoint_passes_the_guard(tmp_path):
    _, path = _saved_checkpoint(tmp_path)
    with h5py.File(path, "r") as f:
        reject_foreign_storage(f)
