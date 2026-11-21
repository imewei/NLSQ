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
import tempfile
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
    """Convert a typed JAX PRNG key to a plain uint32 array for storage.

    Parameters
    ----------
    key : jax.Array
        Typed JAX PRNG key.

    Returns
    -------
    numpy.ndarray
        Raw key data from ``jax.random.key_data``.
    """
    return np.asarray(jax.random.key_data(key))


def deserialize_key(data: np.ndarray) -> jax.Array:
    """Reconstruct a typed JAX PRNG key from `serialize_key`'s output.

    Parameters
    ----------
    data : numpy.ndarray
        Raw key data; converted to ``uint32``.

    Returns
    -------
    jax.Array
        Typed PRNG key built with ``jax.random.wrap_key_data``.
    """

    return jax.random.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32))


def serialize_evosax_state(
    state: EvosaxState | CMAESCheckpointState,
) -> dict[str, np.ndarray | int]:
    """Convert an evosax CMA_ES State (a JAX-array-valued flax.struct.dataclass)
    -- or a `CMAESCheckpointState`, which duck-types the same field names --
    to a plain dict of numpy arrays, safe for HDF5 storage. `HPCCheckpointManager.save()`
    calls this directly so the code path real checkpoints go through is the
    same one this module's serialization round-trip tests exercise.

    Parameters
    ----------
    state : EvosaxState or CMAESCheckpointState
        State to convert.

    Returns
    -------
    dict
        ``generation_counter`` as an ``int`` plus one numpy array per evosax
        array field of ``state``.
    """
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

    Parameters
    ----------
    d : dict
        Output of `serialize_evosax_state`.
    template_state : EvosaxState
        State whose structure is reused.

    Returns
    -------
    EvosaxState
        ``template_state`` with the array fields and ``generation_counter``
        replaced by the values in ``d``.
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
    orchestration knobs, not identity of the optimization problem. ``p0`` is
    not part of the identity either: a resumed run continues from the stored
    state, which supersedes any new ``p0``.

    Parameters
    ----------
    model_id : str
        Model identifier stored in the fingerprint.
    xdata : numpy.ndarray
        Independent data, hashed into ``data_hash``.
    ydata : numpy.ndarray
        Dependent data, hashed into ``data_hash``.
    sigma : numpy.ndarray or None
        Uncertainties, hashed into ``data_hash`` when not None.
    bounds : tuple of numpy.ndarray
        ``(lower, upper)`` parameter bounds.
    config_fields : dict
        Identity-relevant configuration, hashed into ``config_hash`` with keys
        in sorted order.

    Returns
    -------
    dict
        A mix of string/int fields (compared by equality on load) and
        the raw bounds_lower/bounds_upper arrays (stored for human
        auditability per spec section 6 -- not used for the equality check,
        bounds_hash covers that).
    """
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
        reconstruct it independently.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path.

        Returns
        -------
        Path
            ``path`` with ``.bak`` appended to its suffix.
        """
        path = Path(path)
        return path.with_suffix(path.suffix + ".bak")

    def save(
        self,
        path: str | Path,
        state: CMAESCheckpointState,
        fingerprint: dict[str, Any],
    ) -> None:
        """Write a CMA-ES checkpoint atomically, keeping the previous one as ``.bak``.

        The file is written to a temporary directory next to ``path``, fsynced,
        and moved into place with ``os.replace``. An intact existing checkpoint
        is first rotated to the backup path; a corrupt primary is not, so it
        cannot overwrite the only good backup.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path; parent directories are created.
        state : CMAESCheckpointState
            Optimizer state to store.
        fingerprint : dict
            Problem fingerprint stored with the state and checked by ``load``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bak_path = self.bak_path(path)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name
            self._write_checkpoint(tmp_path, state, fingerprint)

            # Rotate only an intact primary: after a backup recovery the primary is
            # the corrupt file, and rotating it would destroy the only good copy.
            if self._is_intact(path, fingerprint):
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

    def _is_intact(self, path: Path, fingerprint: dict[str, Any]) -> bool:
        """True if ``load()`` would not need the backup for ``path``.

        A fingerprint mismatch is intact: the file is a valid checkpoint of another run.
        """
        try:
            self._load_one(path, fingerprint)
        except CheckpointFingerprintMismatch:
            return True
        except Exception:
            return False
        return True

    def _write_checkpoint(
        self, tmp_path: Path, state: CMAESCheckpointState, fingerprint: dict[str, Any]
    ) -> None:
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

    def load(
        self,
        path: str | Path,
        expected_fingerprint: dict[str, Any],
    ) -> CMAESCheckpointState:
        """Load a CMA-ES checkpoint, falling back to the ``.bak`` copy.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path.
        expected_fingerprint : dict
            Fingerprint of the current problem; the stored one must match.

        Returns
        -------
        CMAESCheckpointState
            The restored optimizer state.

        Raises
        ------
        CheckpointFingerprintMismatch
            If the checkpoint belongs to a different problem. The backup is not
            tried in that case.
        FileNotFoundError
            If neither the primary nor a backup exists.
        ValueError, OSError, KeyError
            If the primary is unreadable or malformed (torn write, wrong version,
            invalid shapes, not an HDF5 file, missing group) and no readable
            backup exists; the backup's own error propagates unchanged. A backup
            from a different problem raises ``CheckpointFingerprintMismatch``.
        """
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
        except Exception as primary_error:  # any unreadable/malformed primary
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

        from nlsq.streaming.phases.checkpoint import (
            check_checkpoint_dataset,
            read_checkpoint_bytes,
            read_checkpoint_scalar,
            reject_foreign_storage,
        )

        with h5py.File(path, "r") as f:
            # first: opening f["state"] would already follow an external link
            reject_foreign_storage(f)
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

            state_group = f["state"]
            # Metadata-only validation before any read: a tiny sparse file must
            # not expand to its declared shape (n from `mean`; C/B are (n, n)).
            n = int(state_group["mean"].shape[0]) if state_group["mean"].ndim else None
            vec = (n,)
            expected = {
                "mean": vec,
                "C": (n, n),
                "B": (n, n),
                "D": vec,
                "p_c": vec,
                "p_std": vec,
                "best_solution": vec,
            }
            # A wrong-length key would pass here and only fail in wrap_key_data on
            # resume, outside the load()/.bak fallback.
            check_checkpoint_dataset(
                state_group["key_data"],
                "key_data",
                shape=jax.random.key_data(jax.random.key(0)).shape,
            )
            for name in _EVOSAX_ARRAY_FIELDS:
                check_checkpoint_dataset(
                    state_group[name], name, shape=expected.get(name)
                )
            kwargs: dict[str, Any] = {
                "generation_counter": int(
                    read_checkpoint_scalar(
                        state_group["generation_counter"], "generation_counter"
                    )
                ),
                "key_data": np.asarray(state_group["key_data"][()]),
                "popsize": int(
                    read_checkpoint_scalar(state_group["popsize"], "popsize")
                ),
                "fitness_history": safe_loads(
                    read_checkpoint_bytes(
                        state_group["fitness_history"], "fitness_history"
                    ),
                ),
            }
            for name in _EVOSAX_ARRAY_FIELDS:
                kwargs[name] = jnp.asarray(state_group[name][()])

            state = CMAESCheckpointState(**kwargs)

            # Compared only after the whole payload validated: a mismatch must mean
            # "a well-formed checkpoint of another run", never "malformed".
            fp_group = f["fingerprint"]
            # Every required field must be present before any value is compared:
            # a missing one is a malformed file, not another run's checkpoint.
            missing = [
                k for k in _FINGERPRINT_COMPARISON_KEYS if k not in fp_group.attrs
            ]
            if missing:
                raise ValueError(
                    f"Checkpoint at {path} is missing fingerprint field(s) {missing}."
                )
            for k in _FINGERPRINT_COMPARISON_KEYS:
                expected_v = expected_fingerprint[k]
                actual_v = fp_group.attrs[k]
                if actual_v != expected_v:
                    raise CheckpointFingerprintMismatch(
                        f"Checkpoint fingerprint mismatch on field {k!r}: "
                        f"checkpoint has {actual_v!r}, current run has "
                        f"{expected_v!r}. Refusing to resume from an "
                        "incompatible checkpoint -- use a different run_id "
                        "or checkpoint_dir if this is intentionally a new run.",
                    )

            return state
