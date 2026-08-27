"""Checkpoint state and serialization helpers for CMA-ES resume.

Independent of nlsq/streaming/phases/checkpoint.py -- that module's
CheckpointState/CheckpointManager are tightly coupled to
HybridStreamingConfig and streaming-phase state; cloning only the generic
HDF5/versioning conventions here, not the class itself.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped,import-not-found]
import jax
import jax.numpy as jnp
import numpy as np

from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps, safe_loads

if TYPE_CHECKING:
    from evosax.algorithms.distribution_based.cma_es import (  # type: ignore[import-not-found,import-untyped]
        State as EvosaxState,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "CMAESCheckpointState",
    "deserialize_evosax_state",
    "deserialize_key",
    "serialize_evosax_state",
    "serialize_key",
]

_EVOSAX_ARRAY_FIELDS = (
    "mean",
    "std",
    "p_std",
    "p_c",
    "C",
    "B",
    "D",
    "best_solution",
    "best_fitness",
)


@dataclass
class CMAESCheckpointState:
    """Full CMA-ES state needed to resume a `restart_strategy="none"` run.

    See docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md
    section 5 for the field-to-source mapping.
    """

    generation_counter: int
    mean: jax.Array
    std: jax.Array
    p_std: jax.Array
    p_c: jax.Array
    C: jax.Array
    B: jax.Array
    D: jax.Array
    best_solution: jax.Array
    best_fitness: float | jax.Array  # jax.Array after a load(): best_fitness
    # is one of _EVOSAX_ARRAY_FIELDS, so _load_one reconstructs it via
    # jnp.asarray(...) like the other evosax array fields, not as a plain
    # float. Consumers already coerce with float()/jnp.asarray() as needed.
    key_data: np.ndarray
    fitness_history: list[float] = field(default_factory=list)
    popsize: int = 0


def serialize_key(key: jax.Array) -> np.ndarray:
    """Convert a typed JAX PRNG key to a plain uint32 array for storage."""
    return np.asarray(jax.random.key_data(key))


def deserialize_key(data: np.ndarray) -> jax.Array:
    """Reconstruct a typed JAX PRNG key from `serialize_key`'s output."""

    return jax.random.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32))


def serialize_evosax_state(
    state: EvosaxState | CMAESCheckpointState,
) -> dict[str, np.ndarray | int]:
    """Convert an evosax CMA_ES State (a JAX-array-valued flax.struct.dataclass)
    -- or a `CMAESCheckpointState`, which duck-types the same field names --
    to a plain dict of numpy arrays, safe for HDF5 storage. `HPCCheckpointManager.save()`
    calls this directly so the code path real checkpoints go through is the
    same one this module's serialization round-trip tests exercise."""
    out: dict[str, np.ndarray | int] = {
        "generation_counter": int(state.generation_counter),
    }
    for name in _EVOSAX_ARRAY_FIELDS:
        out[name] = np.asarray(getattr(state, name))
    return out


def deserialize_evosax_state(
    d: dict[str, Any],
    template_state: EvosaxState,
) -> EvosaxState:
    """Rebuild an evosax State from `serialize_evosax_state`'s output.

    `template_state` supplies the dataclass shape/type (from a fresh
    `es.init(...)` call with the same popsize/n_params) -- only its field
    values are replaced, not its structure.
    """

    replacements: dict[str, Any] = {
        name: jnp.asarray(d[name]) for name in _EVOSAX_ARRAY_FIELDS
    }
    replacements["generation_counter"] = int(d["generation_counter"])  # type: ignore[typeddict-item]
    return template_state.replace(**replacements)


__all__ += [
    "CheckpointFingerprintMismatch",
    "HPCCheckpointManager",
    "compute_fingerprint",
]

_VERSION = "1.0"


def _hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def compute_fingerprint(
    model_id: str,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    bounds: tuple[np.ndarray, np.ndarray],
    config_fields: dict[str, Any],
) -> dict[str, Any]:
    """Compute the identity fingerprint stored with a checkpoint and
    checked on resume (spec section 6). Never includes checkpoint_dir/
    checkpoint_interval/run_id/model_id-as-a-config-field -- those are
    orchestration knobs, not identity of the optimization problem.

    Returns a mix of string/int fields (compared by equality on load) and
    the raw bounds_lower/bounds_upper arrays (stored for human
    auditability per spec section 6 -- not used for the equality check,
    bounds_hash covers that)."""
    data_hasher = hashlib.sha256()
    data_hasher.update(_hash_array(np.asarray(xdata)).encode())
    data_hasher.update(_hash_array(np.asarray(ydata)).encode())
    if sigma is not None:
        data_hasher.update(_hash_array(np.asarray(sigma)).encode())

    config_hash = hashlib.sha256(
        safe_dumps({k: config_fields[k] for k in sorted(config_fields)}),
    ).hexdigest()

    lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
    return {
        "model_id": model_id,
        "data_hash": data_hasher.hexdigest(),
        "n_params": len(np.atleast_1d(lb)),
        "bounds_hash": hashlib.sha256(lb.tobytes() + ub.tobytes()).hexdigest(),
        "config_hash": config_hash,
        "bounds_lower": lb,
        "bounds_upper": ub,
    }


# Fingerprint fields compared for equality on load(); bounds_lower/upper are
# stored for auditability only (bounds_hash already covers their identity
# check, and array-valued attrs would need special-cased comparison).
_FINGERPRINT_COMPARISON_KEYS = (
    "model_id",
    "data_hash",
    "n_params",
    "bounds_hash",
    "config_hash",
)


class CheckpointFingerprintMismatch(ValueError):
    """Raised when a checkpoint's stored fingerprint doesn't match the
    current run's inputs. A `ValueError` subclass for backward
    compatibility with generic `except ValueError` handling, but distinct
    from the plain `ValueError`s `_load_one` raises for a torn write or
    version mismatch: those mean the file is unusable and `.bak` is a
    legitimate recovery (FR8); this means the file loaded and parsed fine,
    it just belongs to a different run (FR6's deliberate hard stop) --
    `HPCCheckpointManager.load()` never falls back to `.bak` for this one.
    """


class HPCCheckpointManager:
    """Owns all checkpoint file I/O for CMA-ES resume. CMAESOptimizer calls
    this; it never touches h5py directly (keeps the optimizer a pure
    numerical engine, per the three-brain architecture review)."""

    VERSION = _VERSION

    @staticmethod
    def bak_path(path: str | Path) -> Path:
        """The `.bak` path `save()`/`load()` use for a given checkpoint
        path -- the single source of truth for this naming convention, so
        callers (e.g. CMAESOptimizer's resume-on-entry check) never need to
        reconstruct it independently."""
        path = Path(path)
        return path.with_suffix(path.suffix + ".bak")

    def save(
        self,
        path: str | Path,
        state: CMAESCheckpointState,
        fingerprint: dict[str, Any],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bak_path = self.bak_path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with h5py.File(tmp_path, "w") as f:
            f.attrs["timestamp"] = time.time()

            state_group = f.create_group("state")
            for name, value in serialize_evosax_state(state).items():
                state_group.create_dataset(name, data=value)
            state_group.create_dataset("key_data", data=state.key_data)
            state_group.create_dataset("popsize", data=state.popsize)
            state_group.create_dataset(
                "fitness_history",
                data=np.void(safe_dumps(state.fitness_history)),
            )

            fp_group = f.create_group("fingerprint")
            for k, v in fingerprint.items():
                if isinstance(v, np.ndarray):
                    fp_group.create_dataset(k, data=v)
                else:
                    fp_group.attrs[k] = v

            # Written last: load() treats its absence as a torn write.
            f.attrs["version"] = self.VERSION
            f.attrs["completion_marker"] = True
            f.flush()
            os.fsync(f.id.get_vfd_handle())

        if path.exists():
            path.replace(bak_path)
        os.replace(tmp_path, path)

        # fsync the containing directory too: a bare file fsync only
        # guarantees the file's *contents* survive a crash, not that the
        # rename's directory-entry update does (POSIX rename durability
        # requires syncing the directory that holds the renamed entry).
        # os.open() cannot open a directory on Windows -- this hardening
        # step is POSIX-only; skip it there rather than crash (the file
        # rename itself, via os.replace() above, already happened).
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def load(
        self,
        path: str | Path,
        expected_fingerprint: dict[str, Any],
    ) -> CMAESCheckpointState:
        path = Path(path)
        bak_path = self.bak_path(path)

        try:
            return self._load_one(path, expected_fingerprint)
        except CheckpointFingerprintMismatch:
            # The primary loaded and parsed fine -- it just doesn't belong
            # to this run. Never fall back to .bak for this: the .bak
            # almost always shares the primary's fingerprint (both come
            # from the same run), so a fallback would just re-raise this
            # same mismatch after a misleading "falling back to .bak" log
            # message. Re-raise immediately as the clean, unambiguous
            # hard stop FR6 requires.
            raise
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            KeyError,
            SafeSerializationError,
        ) as primary_error:
            if not bak_path.exists():
                raise
            logger.warning(
                "Primary checkpoint %s failed to load (%s); falling back to %s",
                path,
                primary_error,
                bak_path,
            )
            return self._load_one(bak_path, expected_fingerprint)

    def _load_one(
        self,
        path: Path,
        expected_fingerprint: dict[str, Any],
    ) -> CMAESCheckpointState:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with h5py.File(path, "r") as f:
            if not bool(f.attrs.get("completion_marker", False)):
                raise ValueError(
                    f"Checkpoint at {path} is missing its completion marker "
                    "(torn write) -- refusing to load.",
                )
            version = f.attrs.get("version")
            if version != self.VERSION:
                raise ValueError(
                    f"Checkpoint version {version!r} != expected {self.VERSION!r}",
                )

            fp_group = f["fingerprint"]
            for k in _FINGERPRINT_COMPARISON_KEYS:
                expected_v = expected_fingerprint[k]
                actual_v = fp_group.attrs.get(k)
                if actual_v != expected_v:
                    raise CheckpointFingerprintMismatch(
                        f"Checkpoint fingerprint mismatch on field {k!r}: "
                        f"checkpoint has {actual_v!r}, current run has "
                        f"{expected_v!r}. Refusing to resume from an "
                        "incompatible checkpoint -- use a different run_id "
                        "or checkpoint_dir if this is intentionally a new run.",
                    )

            state_group = f["state"]
            kwargs: dict[str, Any] = {
                "generation_counter": int(state_group["generation_counter"][()]),
                "key_data": np.asarray(state_group["key_data"][()]),
                "popsize": int(state_group["popsize"][()]),
                "fitness_history": safe_loads(
                    bytes(state_group["fitness_history"][()]),
                ),
            }
            for name in _EVOSAX_ARRAY_FIELDS:
                kwargs[name] = jnp.asarray(state_group[name][()])

            return CMAESCheckpointState(**kwargs)
