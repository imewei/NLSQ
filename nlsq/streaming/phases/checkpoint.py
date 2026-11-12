"""Checkpoint management for streaming optimization.

This module contains the CheckpointManager class that handles
save/load checkpoint operations for streaming optimizer state.

Checkpoints enable resumption of long-running optimizations
and provide recovery from failures.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-not-found,import-untyped]
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-not-found,import-untyped]

from nlsq.utils.safe_serialize import safe_dumps, safe_loads

if TYPE_CHECKING:
    from jax import Array

    from nlsq.global_optimization.config import GlobalOptimizationConfig
    from nlsq.global_optimization.tournament import TournamentSelector
    from nlsq.precision.parameter_normalizer import ParameterNormalizer
    from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _find_lbfgs_state(tree: Any) -> Any | None:
    """Return the ScaleByLBFGSState inside a (possibly chained) optax state."""
    from optax._src.transform import (  # type: ignore[import-not-found,import-untyped]
        ScaleByLBFGSState,
    )

    if isinstance(tree, ScaleByLBFGSState):
        return tree
    if isinstance(tree, tuple):
        for item in tree:
            found = _find_lbfgs_state(item)
            if found is not None:
                return found
    return None


# Element cap for any array read from a checkpoint (2**26 f8 = 512 MiB).
MAX_CHECKPOINT_ELEMENTS = 1 << 26


def check_checkpoint_dataset(
    dset: Any,
    name: str,
    *,
    shape: tuple[int | None, ...] | None = None,
    kinds: str = "fiu",
) -> None:
    """Validate an HDF5 dataset from metadata only, before anything is read.

    Rejects non-datasets, dtypes whose kind is not in ``kinds`` (default
    float/int/uint; the tournament survival mask also allows bool), shapes above
    ``MAX_CHECKPOINT_ELEMENTS``, a mismatch with the expected ``shape`` (``None``
    entries are wildcards), and
    chunked datasets with unallocated chunks (a tiny sparse file would
    otherwise expand to the full declared shape). Raises ``ValueError``.
    """
    if not isinstance(dset, h5py.Dataset):
        raise ValueError(f"Checkpoint entry '{name}' is not a dataset")
    if dset.dtype.kind not in kinds:
        raise ValueError(
            f"Checkpoint dataset '{name}' has non-numeric dtype {dset.dtype}"
        )
    if math.prod(dset.shape) > MAX_CHECKPOINT_ELEMENTS:
        raise ValueError(
            f"Checkpoint dataset '{name}' shape {tuple(dset.shape)} exceeds "
            f"{MAX_CHECKPOINT_ELEMENTS} elements",
        )
    if shape is not None and (
        len(shape) != dset.ndim
        or any(e is not None and e != a for e, a in zip(shape, dset.shape, strict=True))
    ):
        raise ValueError(
            f"Checkpoint dataset '{name}' has shape {tuple(dset.shape)}, expected {shape}",
        )
    if dset.chunks is not None:
        expected = math.prod(
            -(-s // c) for s, c in zip(dset.shape, dset.chunks, strict=True)
        )
        if dset.id.get_num_chunks() < expected:
            raise ValueError(
                f"Checkpoint dataset '{name}' has unallocated chunks "
                f"({dset.id.get_num_chunks()} of {expected}); sparse data is not supported",
            )


# Byte cap for opaque (void) scalar entries such as the JSON phase history.
MAX_CHECKPOINT_BYTES = 1 << 26


def read_checkpoint_scalar(dset: Any, name: str) -> Any:
    """Validate a numeric dataset is a scalar (shape ``()``), then read it."""
    check_checkpoint_dataset(dset, name, shape=())
    return dset[()]


def read_checkpoint_bytes(dset: Any, name: str) -> bytes:
    """Validate an opaque scalar dataset (``np.void``) from metadata, then read it."""
    if not isinstance(dset, h5py.Dataset):
        raise ValueError(f"Checkpoint entry '{name}' is not a dataset")
    if dset.dtype.kind != "V" or dset.shape != ():
        raise ValueError(
            f"Checkpoint dataset '{name}' must be an opaque scalar, got dtype "
            f"{dset.dtype} shape {tuple(dset.shape)}"
        )
    if dset.dtype.itemsize > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"Checkpoint dataset '{name}' holds {dset.dtype.itemsize} bytes, "
            f"above the {MAX_CHECKPOINT_BYTES}-byte limit"
        )
    return bytes(dset[()])


def read_checkpoint_array(
    dset: Any,
    name: str,
    *,
    shape: tuple[int | None, ...] | None = None,
) -> Array:
    """``check_checkpoint_dataset`` then read as a JAX array."""
    check_checkpoint_dataset(dset, name, shape=shape)
    return jnp.array(dset)


@dataclass
class CheckpointState:
    """State container for checkpoint save/load operations.

    This dataclass captures all optimizer state that needs to be
    persisted across checkpoint boundaries.

    Attributes
    ----------
    current_phase : int
        Current optimization phase (0-3).
    normalized_params : Array | None
        Parameters in normalized space.
    phase1_optimizer_state : Any | None
        Optax L-BFGS optimizer state.
    phase2_JTJ_accumulator : Array | None
        Accumulated J^T J matrix for Phase 2.
    phase2_JTr_accumulator : Array | None
        Accumulated J^T r vector for Phase 2.
    best_params_global : Array | None
        Best parameters found globally.
    best_cost_global : float
        Best cost value globally.
    phase_history : list[dict[str, Any]]
        Complete phase history.
    normalizer : ParameterNormalizer | None
        Parameter normalizer (strategy, scales, offsets).
    tournament_selector : TournamentSelector | None
        Tournament selector for multi-start optimization.
    multistart_candidates : Array | None
        Multi-start candidate parameters.
    """

    current_phase: int
    normalized_params: Array | None
    phase1_optimizer_state: Any | None
    phase2_JTJ_accumulator: Array | None
    phase2_JTr_accumulator: Array | None
    best_params_global: Array | None
    best_cost_global: float
    phase_history: list[dict[str, Any]]
    normalizer: ParameterNormalizer | None
    tournament_selector: TournamentSelector | None
    multistart_candidates: Array | None


class CheckpointManager:
    """Manages checkpoint save/load operations for streaming optimizer.

    This class encapsulates all checkpoint I/O logic, including:
    - HDF5 file format versioning
    - Optimizer state serialization
    - Tournament and normalizer state handling
    - Guard clause-based validation

    Parameters
    ----------
    config : HybridStreamingConfig
        Configuration for streaming optimization.

    Attributes
    ----------
    config : HybridStreamingConfig
        Configuration object.
    version : str
        Checkpoint format version (currently "3.0").
    """

    # 3.1 adds the completion marker; 3.0 files (no marker) still load.
    VERSION = "3.1"

    def __init__(self, config: HybridStreamingConfig) -> None:
        """Initialize CheckpointManager.

        Parameters
        ----------
        config : HybridStreamingConfig
            Configuration for streaming optimization.
        """
        self.config = config

    def save(self, checkpoint_path: str | Path, state: CheckpointState) -> None:
        """Save checkpoint with phase-specific state to HDF5 file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file (.h5).
        state : CheckpointState
            Optimizer state to save.

        Notes
        -----
        Checkpoint format version 3.0 includes:
        - current_phase: Current phase number
        - normalized_params: Parameters in normalized space
        - phase1_optimizer_state: Optax L-BFGS state
        - phase2_jtj_accumulator: Accumulated J^T J matrix
        - phase2_jtr_accumulator: Accumulated J^T r vector
        - best_params_global: Best parameters found globally
        - best_cost_global: Best cost value globally
        - phase_history: Complete phase history
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename so a failed save never clobbers
        # the previous good checkpoint.
        with tempfile.TemporaryDirectory(dir=checkpoint_path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / checkpoint_path.name
            self._write_all(tmp_path, state)
            os.replace(tmp_path, checkpoint_path)

    def _write_all(self, tmp_path: Path, state: CheckpointState) -> None:
        with h5py.File(tmp_path, "w") as f:
            self._write_metadata(f)
            phase_state = f.create_group("phase_state")
            self._write_phase_state(phase_state, state)
            self._write_optimizer_state(phase_state, state)
            self._write_accumulators(phase_state, state)
            self._write_best_params(phase_state, state)
            self._write_phase_history(phase_state, state)
            self._write_normalizer_state(phase_state, state)
            self._write_tournament_state(phase_state, state)
            self._write_multistart_candidates(phase_state, state)
            # Written last: load() treats its absence as a torn write.
            f.attrs["completion_marker"] = True
            f.flush()
            os.fsync(f.id.get_vfd_handle())

    def load(
        self,
        checkpoint_path: str | Path,
        global_config: GlobalOptimizationConfig | None = None,
    ) -> CheckpointState:
        """Load checkpoint and restore phase-specific state.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file (.h5).
        global_config : GlobalOptimizationConfig | None
            Configuration for tournament reconstruction.

        Returns
        -------
        CheckpointState
            Restored optimizer state.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        ValueError
            If checkpoint version is incompatible.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with h5py.File(checkpoint_path, "r") as f:
            self._validate_version(f)
            phase_state = f["phase_state"]

            current_phase = int(
                read_checkpoint_scalar(phase_state["current_phase"], "current_phase")
            )
            n = self._infer_n_params(phase_state)
            normalized_params = self._read_normalized_params(phase_state, n)
            phase1_optimizer_state = self._read_optimizer_state(phase_state, n)
            phase2_JTJ, phase2_JTr = self._read_accumulators(phase_state, n)
            best_params, best_cost = self._read_best_params(phase_state, n)
            phase_history = self._read_phase_history(phase_state)
            tournament_selector = self._read_tournament_state(
                phase_state,
                global_config,
            )
            multistart_candidates = self._read_multistart_candidates(phase_state, n)

            return CheckpointState(
                current_phase=current_phase,
                normalized_params=normalized_params,
                phase1_optimizer_state=phase1_optimizer_state,
                phase2_JTJ_accumulator=phase2_JTJ,
                phase2_JTr_accumulator=phase2_JTr,
                best_params_global=best_params,
                best_cost_global=best_cost,
                phase_history=phase_history,
                normalizer=None,  # Full normalizer reconstruction requires bounds/p0
                tournament_selector=tournament_selector,
                multistart_candidates=multistart_candidates,
            )

    def _write_metadata(self, f: h5py.File) -> None:
        """Write version metadata to HDF5 file."""
        f.attrs["version"] = self.VERSION
        f.attrs["timestamp"] = time.time()

    def _write_phase_state(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write current phase and normalized parameters."""
        phase_state.create_dataset("current_phase", data=state.current_phase)
        if state.normalized_params is not None:
            phase_state.create_dataset(
                "normalized_params",
                data=state.normalized_params,
            )

    def _write_optimizer_state(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write Phase 1 optimizer state (Optax L-BFGS)."""
        if state.phase1_optimizer_state is None:
            return

        # Locate by type: gradient clipping puts a clip state at index 0.
        inner_state = _find_lbfgs_state(state.phase1_optimizer_state)
        if inner_state is None:
            import logging

            logging.getLogger(__name__).warning(
                "Unsupported Phase 1 optimizer state for checkpointing. "
                "Expected L-BFGS (ScaleByLBFGSState). Skipping optimizer state.",
            )
            return

        opt_state_group = phase_state.create_group("phase1_optimizer_state")
        opt_state_group.create_dataset("count", data=int(inner_state.count))
        opt_state_group.attrs["optimizer_type"] = "lbfgs"
        opt_state_group.create_dataset("params", data=inner_state.params)
        opt_state_group.create_dataset("updates", data=inner_state.updates)
        opt_state_group.create_dataset(
            "diff_params_memory",
            data=inner_state.diff_params_memory,
        )
        opt_state_group.create_dataset(
            "diff_updates_memory",
            data=inner_state.diff_updates_memory,
        )
        opt_state_group.create_dataset(
            "weights_memory",
            data=inner_state.weights_memory,
        )

    def _write_accumulators(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write Phase 2 accumulators."""
        if state.phase2_JTJ_accumulator is not None:
            phase_state.create_dataset(
                "phase2_jtj_accumulator",
                data=state.phase2_JTJ_accumulator,
            )
        if state.phase2_JTr_accumulator is not None:
            phase_state.create_dataset(
                "phase2_jtr_accumulator",
                data=state.phase2_JTr_accumulator,
            )

    def _write_best_params(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write best parameters tracking."""
        if state.best_params_global is not None:
            phase_state.create_dataset(
                "best_params_global",
                data=state.best_params_global,
            )
        phase_state.create_dataset("best_cost_global", data=state.best_cost_global)

    def _convert_jax_to_numpy(self, obj: Any) -> Any:
        """Recursively convert JAX arrays to numpy for serialization.

        Parameters
        ----------
        obj : Any
            Object that may contain JAX arrays.

        Returns
        -------
        Any
            Object with JAX arrays converted to numpy arrays.
        """
        # Check for JAX array (without importing jax at module level)
        if hasattr(obj, "device") and hasattr(obj, "block_until_ready"):
            # This is a JAX array - convert to numpy
            return np.asarray(obj)
        if isinstance(obj, dict):
            return {k: self._convert_jax_to_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_jax_to_numpy(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._convert_jax_to_numpy(item) for item in obj)
        return obj

    def _write_phase_history(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write phase history using safe JSON serialization."""
        if not state.phase_history:
            return

        # Convert any JAX arrays to numpy before serialization
        phase_history_converted = self._convert_jax_to_numpy(state.phase_history)
        phase_history_bytes = safe_dumps(phase_history_converted)
        phase_state.create_dataset("phase_history", data=np.void(phase_history_bytes))

    def _write_normalizer_state(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write normalization state."""
        if state.normalizer is None:
            return

        norm_group = phase_state.create_group("normalizer_state")
        norm_group.attrs["strategy"] = state.normalizer.strategy
        if state.normalizer.scales is not None:
            norm_group.create_dataset("scales", data=state.normalizer.scales)
        if state.normalizer.offsets is not None:
            norm_group.create_dataset("offsets", data=state.normalizer.offsets)

    def _write_tournament_state(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write tournament selector state."""
        if state.tournament_selector is None:
            return

        tournament_group = phase_state.create_group("tournament_state")
        tournament_checkpoint = state.tournament_selector.to_checkpoint()
        for key, value in tournament_checkpoint.items():
            # Convert any JAX arrays to numpy before serialization
            value_converted = self._convert_jax_to_numpy(value)
            if isinstance(value_converted, (list, dict)):
                tournament_bytes = safe_dumps(value_converted)
                tournament_group.create_dataset(key, data=np.void(tournament_bytes))
            elif value_converted is not None:
                try:
                    tournament_group.create_dataset(key, data=value_converted)
                except TypeError:
                    tournament_bytes = safe_dumps(value_converted)
                    tournament_group.create_dataset(key, data=np.void(tournament_bytes))

    def _write_multistart_candidates(
        self,
        phase_state: h5py.Group,
        state: CheckpointState,
    ) -> None:
        """Write multi-start candidates."""
        if state.multistart_candidates is not None:
            phase_state.create_dataset(
                "multistart_candidates",
                data=state.multistart_candidates,
            )

    def _validate_version(self, f: h5py.File) -> None:
        """Validate checkpoint version compatibility."""
        version = f.attrs.get("version", "1.0")
        if not version.startswith("3."):
            raise ValueError(
                f"Incompatible checkpoint version: {version} (expected 3.x)",
            )
        if version != "3.0" and not f.attrs.get("completion_marker", False):
            raise ValueError("Checkpoint is incomplete (missing completion marker)")

    @staticmethod
    def _infer_n_params(phase_state: h5py.Group) -> int | None:
        """Parameter count from the first 1-D parameter-sized dataset (metadata only)."""
        for key in (
            "normalized_params",
            "best_params_global",
            "phase2_jtr_accumulator",
        ):
            d = phase_state.get(key)
            if isinstance(d, h5py.Dataset) and d.ndim == 1:
                return int(d.shape[0])
        return None

    def _read_normalized_params(
        self, phase_state: h5py.Group, n: int | None = None
    ) -> Array | None:
        """Read normalized parameters."""
        if "normalized_params" not in phase_state:
            return None
        return read_checkpoint_array(
            phase_state["normalized_params"], "normalized_params", shape=(n,)
        )

    def _read_optimizer_state(
        self, phase_state: h5py.Group, n: int | None = None
    ) -> Any | None:
        """Read Phase 1 optimizer state.

        The saved L-BFGS state is for diagnostics only and is NOT resumable:
        the returned tuple lacks the other chain elements (e.g. the line-search
        state), so it does not match ``optax.lbfgs(...).init`` and cannot be
        passed to ``update``. Re-initialise the optimizer to resume.
        """
        if "phase1_optimizer_state" not in phase_state:
            return None

        opt_state_group = phase_state["phase1_optimizer_state"]
        count = int(
            read_checkpoint_scalar(
                opt_state_group["count"], "phase1_optimizer_state/count"
            )
        )
        optimizer_type = opt_state_group.attrs.get("optimizer_type", "lbfgs")

        if optimizer_type != "lbfgs":
            raise ValueError(
                "Unsupported Phase 1 optimizer state in checkpoint. "
                "Expected L-BFGS state.",
            )

        from optax._src.transform import (  # type: ignore[import-not-found,import-untyped]
            ScaleByLBFGSState,
        )

        def _arr(key: str, shape: tuple[int | None, ...]) -> Array:
            return read_checkpoint_array(
                opt_state_group[key], f"phase1_optimizer_state/{key}", shape=shape
            )

        lbfgs_state = ScaleByLBFGSState(
            count=count,
            params=_arr("params", (n,)),
            updates=_arr("updates", (n,)),
            diff_params_memory=_arr("diff_params_memory", (None, n)),
            diff_updates_memory=_arr("diff_updates_memory", (None, n)),
            weights_memory=_arr("weights_memory", (None,)),
        )
        return (lbfgs_state, optax.EmptyState())

    def _read_accumulators(
        self,
        phase_state: h5py.Group,
        n: int | None = None,
    ) -> tuple[Array | None, Array | None]:
        """Read Phase 2 accumulators (J^T J is (n, n), J^T r is (n,))."""
        phase2_JTJ = None
        phase2_JTr = None
        if "phase2_jtj_accumulator" in phase_state:
            phase2_JTJ = read_checkpoint_array(
                phase_state["phase2_jtj_accumulator"],
                "phase2_jtj_accumulator",
                shape=(n, n),
            )
        if "phase2_jtr_accumulator" in phase_state:
            phase2_JTr = read_checkpoint_array(
                phase_state["phase2_jtr_accumulator"],
                "phase2_jtr_accumulator",
                shape=(n,),
            )
        return phase2_JTJ, phase2_JTr

    def _read_best_params(
        self, phase_state: h5py.Group, n: int | None = None
    ) -> tuple[Array | None, float]:
        """Read best parameters tracking."""
        best_params = None
        if "best_params_global" in phase_state:
            best_params = read_checkpoint_array(
                phase_state["best_params_global"], "best_params_global", shape=(n,)
            )
        best_cost = float(
            read_checkpoint_scalar(phase_state["best_cost_global"], "best_cost_global")
        )
        return best_params, best_cost

    def _read_phase_history(self, phase_state: h5py.Group) -> list[dict[str, Any]]:
        """Read phase history."""
        if "phase_history" not in phase_state:
            return []
        phase_history_bytes = read_checkpoint_bytes(
            phase_state["phase_history"], "phase_history"
        )
        return safe_loads(phase_history_bytes)

    def _read_tournament_state(
        self,
        phase_state: h5py.Group,
        global_config: GlobalOptimizationConfig | None,
    ) -> TournamentSelector | None:
        """Read tournament selector state."""
        if "tournament_state" not in phase_state:
            return None
        if not self.config.enable_multistart:
            return None
        if global_config is None:
            return None

        from nlsq.global_optimization.tournament import TournamentSelector

        tournament_group = phase_state["tournament_state"]
        tournament_checkpoint: dict[str, Any] = {}
        for key in tournament_group:
            dset = tournament_group[key]
            name = f"tournament_state/{key}"
            if isinstance(dset, h5py.Dataset) and dset.dtype.kind == "V":
                tournament_checkpoint[key] = safe_loads(
                    read_checkpoint_bytes(dset, name)
                )
            else:
                check_checkpoint_dataset(dset, name, kinds="fiub")
                value = dset[()]
                tournament_checkpoint[key] = (
                    np.array(value) if hasattr(value, "__len__") else value
                )

        return TournamentSelector.from_checkpoint(tournament_checkpoint, global_config)

    def _read_multistart_candidates(
        self, phase_state: h5py.Group, n: int | None = None
    ) -> Array | None:
        """Read multi-start candidates, shape (n_starts, n_params)."""
        if "multistart_candidates" not in phase_state:
            return None
        return read_checkpoint_array(
            phase_state["multistart_candidates"],
            "multistart_candidates",
            shape=(None, n),
        )


__all__ = ["CheckpointManager", "CheckpointState"]
