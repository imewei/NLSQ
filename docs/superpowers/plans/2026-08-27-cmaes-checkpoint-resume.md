# CMA-ES Checkpoint/Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `CMAESOptimizer` (with `restart_strategy="none"`) crash/preemption-safe checkpoint and resume, so a long HPC fit killed mid-run can continue from its last saved generation instead of losing all progress.

**Architecture:** New standalone module `nlsq/global_optimization/checkpoint.py` (state dataclass + HDF5 manager, no dependency on the existing `nlsq/streaming/phases/checkpoint.py`). `CMAESOptimizer.fit()` builds the `HPCCheckpointManager`, checkpoint path, fingerprint, and signal-handling `threading.Event` up front (main-thread-guarded), then passes them as plain parameters into `_run_cmaes`/`_run_cmaes_single`; the loop itself only calls `.save()`/`.load()` and checks the event -- it never constructs a path, a fingerprint, or touches the `signal` module. `fit()`'s own `try`/`finally` restores signal handlers on every exit path. `_fit_with_hpc`/`_fit_with_auto_global` in `minpack.py` pass checkpoint fields (and, separately, a previously-dropped `method` override) through instead of discarding them.

**Tech Stack:** Python 3.12+, JAX (typed PRNG keys via `jax.random.key`/`key_data`/`wrap_key_data`), evosax `CMA_ES`, h5py, `nlsq.utils.safe_serialize.safe_dumps`/`safe_loads`.

**Spec:** `docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md`

## Global Constraints

- Checkpointing requires `config.seed` to be an explicit `int` (spec FR2) — never silently proceed with an OS-entropy seed.
- Checkpointing requires `config.model_id` **and** `config.run_id` to be explicit non-`None` strings (spec FR4) — never fingerprint the model closure, never let an unset run_id silently collide with an unrelated run sharing the same `checkpoint_dir`.
- `restart_strategy="bipop"` + checkpointing raises `NotImplementedError` (spec FR3) — out of scope for this plan.
- Every save is atomic: tmp file + fsync + `os.replace`, prior good file rotated to `.bak`, containing directory fsynced after the rename (spec FR8).
- `load()` falls back to `.bak` whenever the primary exists but fails to load — this is mandatory, not best-effort (spec FR8).
- A fingerprint mismatch on load raises `ValueError` — never silently starts fresh in the same file (spec FR6).
- `fit()`, not `_run_cmaes_single`, constructs the checkpoint manager/path/fingerprint and registers/restores signal handlers; the optimizer method only receives these as parameters (grilling-session architecture decision).
- Signal handler registration only happens on the main thread; off it, periodic interval saves remain the only crash-safety mechanism (spec FR7).
- `fit()`'s `method` parameter must actually reach `MethodSelector.select()` for every `workflow='auto_global'`/`'hpc'` caller, not only when checkpointing is enabled (spec FR9).
- No code in this plan touches `MultiStartOrchestrator`, `LargeDatasetFitter`, `AdaptiveHybridStreamingOptimizer`, or multi-device paths.

---

### Task 1: Checkpoint state dataclass + evosax/PRNG serialization helpers

**Files:**
- Create: `nlsq/global_optimization/checkpoint.py`
- Test: `tests/global_optimization/test_checkpoint.py`

**Interfaces:**
- Produces: `CMAESCheckpointState` (dataclass, fields per spec §5), `serialize_evosax_state(state) -> dict[str, np.ndarray]`, `deserialize_evosax_state(d: dict, template_state) -> evosax State` (uses `template_state.replace(...)` — evosax `State` is a `flax.struct.dataclass`, verified via direct introspection: `hasattr(state, '_replace')` is `False`, `hasattr(state, 'replace')` is `True`; do not use `._replace`, that is the NamedTuple API and does not exist on this type), `serialize_key(key: jax.Array) -> np.ndarray`, `deserialize_key(data: np.ndarray) -> jax.Array`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/global_optimization/test_checkpoint.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_optimization/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nlsq.global_optimization.checkpoint'`

- [ ] **Step 3: Write the implementation**

```python
# nlsq/global_optimization/checkpoint.py
"""Checkpoint state and serialization helpers for CMA-ES resume.

Independent of nlsq/streaming/phases/checkpoint.py -- that module's
CheckpointState/CheckpointManager are tightly coupled to
HybridStreamingConfig and streaming-phase state; cloning only the generic
HDF5/versioning conventions here, not the class itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

if TYPE_CHECKING:
    from evosax.algorithms.distribution_based.cma_es import State as EvosaxState

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
    best_fitness: float
    key_data: np.ndarray
    fitness_history: list[float] = field(default_factory=list)
    popsize: int = 0


def serialize_key(key: jax.Array) -> np.ndarray:
    """Convert a typed JAX PRNG key to a plain uint32 array for storage."""
    return np.asarray(jax.random.key_data(key))


def deserialize_key(data: np.ndarray) -> jax.Array:
    """Reconstruct a typed JAX PRNG key from `serialize_key`'s output."""
    return jax.random.wrap_key_data(np.asarray(data, dtype=np.uint32))


def serialize_evosax_state(state: EvosaxState) -> dict[str, np.ndarray | int]:
    """Convert an evosax CMA_ES State (a JAX-array-valued flax.struct.dataclass) to a
    plain dict of numpy arrays, safe for HDF5 storage."""
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
    import jax.numpy as jnp

    replacements = {name: jnp.asarray(d[name]) for name in _EVOSAX_ARRAY_FIELDS}
    replacements["generation_counter"] = int(d["generation_counter"])
    return template_state.replace(**replacements)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_optimization/test_checkpoint.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/checkpoint.py tests/global_optimization/test_checkpoint.py
git commit -m "feat(global_optimization): add CMA-ES checkpoint state + serialization helpers"
```

---

### Task 2: HPCCheckpointManager — atomic HDF5 save/load with fingerprint

**Files:**
- Modify: `nlsq/global_optimization/checkpoint.py`
- Test: `tests/global_optimization/test_checkpoint.py` (append)

**Interfaces:**
- Consumes: `CMAESCheckpointState` (Task 1).
- Produces: `compute_fingerprint(model_id: str, xdata: np.ndarray, ydata: np.ndarray, sigma: np.ndarray | None, bounds: tuple[np.ndarray, np.ndarray], config_fields: dict) -> dict[str, Any]` (string/int fields for equality comparison plus raw `bounds_lower`/`bounds_upper` arrays for auditability), `HPCCheckpointManager` with `.save(path: Path, state: CMAESCheckpointState, fingerprint: dict) -> None` and `.load(path: Path, expected_fingerprint: dict) -> CMAESCheckpointState` (raises `ValueError` on fingerprint mismatch, `FileNotFoundError` only when neither the primary nor its `.bak` exist; falls back to `.bak` — logging a warning — whenever the primary exists but fails to load).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/global_optimization/test_checkpoint.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_optimization/test_checkpoint.py -v`
Expected: FAIL with `ImportError: cannot import name 'HPCCheckpointManager'`

- [ ] **Step 3: Write the implementation**

```python
# append to nlsq/global_optimization/checkpoint.py
import hashlib
import os
import time
from pathlib import Path

import h5py

from nlsq.utils.safe_serialize import safe_dumps, safe_loads

__all__ += ["HPCCheckpointManager", "compute_fingerprint"]

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
        "n_params": int(len(np.atleast_1d(lb))),
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


class HPCCheckpointManager:
    """Owns all checkpoint file I/O for CMA-ES resume. CMAESOptimizer calls
    this; it never touches h5py directly (keeps the optimizer a pure
    numerical engine, per the three-brain architecture review)."""

    VERSION = _VERSION

    def save(
        self,
        path: str | Path,
        state: CMAESCheckpointState,
        fingerprint: dict[str, Any],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bak_path = path.with_suffix(path.suffix + ".bak")
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with h5py.File(tmp_path, "w") as f:
            f.attrs["timestamp"] = time.time()

            state_group = f.create_group("state")
            state_group.create_dataset(
                "generation_counter",
                data=state.generation_counter,
            )
            for name in _EVOSAX_ARRAY_FIELDS:
                state_group.create_dataset(name, data=np.asarray(getattr(state, name)))
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
        bak_path = path.with_suffix(path.suffix + ".bak")

        try:
            return self._load_one(path, expected_fingerprint)
        except (FileNotFoundError, OSError, ValueError) as primary_error:
            if not path.exists():
                # No primary at all (e.g. first-ever save never happened) --
                # a .bak fallback here would silently resume from an even
                # older, more-stale state than the caller expects. Only
                # fall back when the primary exists but is unusable.
                raise
            if not bak_path.exists():
                raise
            logger.warning(
                "Primary checkpoint %s failed to load (%s); falling back " "to %s",
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
                    raise ValueError(
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
                kwargs[name] = jax.numpy.asarray(state_group[name][()])

            return CMAESCheckpointState(**kwargs)
```

Add `import logging` and `logger = logging.getLogger(__name__)` to the
module's top-level imports (alongside the existing `import hashlib` /
`import os` / `import time` block).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_optimization/test_checkpoint.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/checkpoint.py tests/global_optimization/test_checkpoint.py
git commit -m "feat(global_optimization): add HPCCheckpointManager with atomic writes + fingerprint check"
```

---

### Task 3: CMAESConfig checkpoint fields + validation

**Files:**
- Modify: `nlsq/global_optimization/cmaes_config.py`
- Test: `tests/global_optimization/test_cmaes_config.py` (existing file — append; if it doesn't exist, create it)

**Interfaces:**
- Produces: `CMAESConfig` gains `checkpoint_dir: str | Path | None = None`, `checkpoint_interval: int = 10`, `run_id: str | None = None`, `model_id: str | None = None`. `_validate()` raises `ValueError` per Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/global_optimization/test_cmaes_config.py
import pytest

from nlsq.global_optimization.cmaes_config import CMAESConfig


def test_checkpoint_requires_seed():
    # model_id and run_id both supplied so this isolates the seed check.
    with pytest.raises(ValueError, match="seed"):
        CMAESConfig(
            checkpoint_dir="/tmp/ckpt",
            model_id="m1",
            run_id="r1",
            seed=None,
        )


def test_checkpoint_requires_model_id():
    # seed and run_id both supplied so this isolates the model_id check.
    with pytest.raises(ValueError, match="model_id"):
        CMAESConfig(checkpoint_dir="/tmp/ckpt", seed=1, run_id="r1", model_id=None)


def test_checkpoint_requires_run_id():
    # seed and model_id both supplied so this isolates the run_id check.
    # Required for the same reason model_id/seed are (grilling-session
    # decision): an unset run_id defaulting to a shared filename would let
    # unrelated runs collide on the same checkpoint file.
    with pytest.raises(ValueError, match="run_id"):
        CMAESConfig(checkpoint_dir="/tmp/ckpt", seed=1, model_id="m1", run_id=None)


def test_checkpoint_rejects_bipop():
    with pytest.raises(NotImplementedError, match="bipop"):
        CMAESConfig(
            checkpoint_dir="/tmp/ckpt",
            seed=1,
            model_id="m1",
            run_id="r1",
            restart_strategy="bipop",
        )


def test_checkpoint_with_none_strategy_and_all_required_fields_succeeds():
    config = CMAESConfig(
        checkpoint_dir="/tmp/ckpt",
        seed=1,
        model_id="m1",
        run_id="r1",
        restart_strategy="none",
    )
    assert config.checkpoint_dir == "/tmp/ckpt"
    assert config.checkpoint_interval == 10


def test_checkpoint_interval_must_be_positive():
    with pytest.raises(ValueError, match="checkpoint_interval"):
        CMAESConfig(
            checkpoint_dir="/tmp/ckpt",
            seed=1,
            model_id="m1",
            run_id="r1",
            restart_strategy="none",
            checkpoint_interval=0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_optimization/test_cmaes_config.py -v -k checkpoint`
Expected: FAIL — `CMAESConfig.__init__() got an unexpected keyword argument 'checkpoint_dir'`

- [ ] **Step 3: Write the implementation**

Add fields to the `CMAESConfig` dataclass (after `seed: int | None = None`):

```python
# Checkpoint/resume (CMA-ES with restart_strategy="none" only -- see
# docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md)
checkpoint_dir: str | None = None
checkpoint_interval: int = 10
run_id: str | None = None
model_id: str | None = None
```

Add to `_validate()` (after the existing `restart_strategy` check):

```python
if self.checkpoint_dir is not None:
    if self.seed is None:
        raise ValueError(
            "checkpoint_dir requires a fixed seed for reproducible "
            "resume; set seed=<int>.",
        )
    if self.model_id is None:
        raise ValueError(
            "checkpoint_dir requires model_id (a stable string "
            "identifying the model function) -- the closure cannot "
            "be safely fingerprinted automatically.",
        )
    if self.run_id is None:
        raise ValueError(
            "checkpoint_dir requires run_id (a stable string identifying "
            "this run) -- an unset run_id would silently collide with "
            "unrelated runs sharing the same checkpoint_dir.",
        )
    if self.restart_strategy == "bipop":
        raise NotImplementedError(
            "Checkpoint/resume is not implemented for "
            "restart_strategy='bipop' yet. Set restart_strategy="
            "'none' to use checkpointing.",
        )
    if self.checkpoint_interval < 1:
        raise ValueError(
            f"checkpoint_interval must be >= 1, got " f"{self.checkpoint_interval}",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_optimization/test_cmaes_config.py -v -k checkpoint`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/cmaes_config.py tests/global_optimization/test_cmaes_config.py
git commit -m "feat(global_optimization): add checkpoint fields + validation to CMAESConfig"
```

---

### Task 4: Wire periodic checkpoint save into `_run_cmaes_single`

**Files:**
- Modify: `nlsq/global_optimization/cmaes_optimizer.py:_run_cmaes_single` (currently lines 563-681)
- Test: `tests/global_optimization/test_cmaes_checkpoint_integration.py`

**Interfaces:**
- Consumes: `CMAESCheckpointState`, `HPCCheckpointManager`, `compute_fingerprint`, `serialize_key` (Tasks 1-2); `CMAESConfig.checkpoint_dir/checkpoint_interval/run_id/model_id` (Task 3).
- Produces: `fit()` builds the `HPCCheckpointManager`, `checkpoint_path`, and `fingerprint` dict (it already has `xdata`/`ydata`/`sigma`/bounds) and passes them into `_run_cmaes`/`_run_cmaes_single` as plain parameters -- per the grilling-session decision, `_run_cmaes_single` never constructs a path or fingerprint itself, it only calls `manager.save(...)` with values it was handed. Saves a checkpoint every `checkpoint_interval` generations plus unconditionally at loop exit, when `checkpoint_manager` is not `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/global_optimization/test_cmaes_checkpoint_integration.py
import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization.checkpoint import HPCCheckpointManager
from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer


def exponential_decay(x, a, b):
    return a * jnp.exp(-b * x)


def test_checkpoint_file_written_during_fit(tmp_path):
    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))

    config = CMAESConfig(
        max_generations=20,
        restart_strategy="none",
        seed=7,
        model_id="exponential_decay_v1",
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=5,
        run_id="test-run-1",
    )
    optimizer = CMAESOptimizer(config=config)
    optimizer.fit(exponential_decay, x, y, bounds=([0.0, 0.0], [10.0, 2.0]))

    checkpoint_path = tmp_path / "test-run-1.h5"
    assert checkpoint_path.exists()

    manager = HPCCheckpointManager()
    with __import__("h5py").File(checkpoint_path, "r") as f:
        assert bool(f.attrs["completion_marker"])
        # Loop ran to max_generations=20 (no early convergence expected
        # this fast) -- final saved generation_counter should be > 0.
        assert int(f["state"]["generation_counter"][()]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/global_optimization/test_cmaes_checkpoint_integration.py -v`
Expected: FAIL — no checkpoint file exists (feature not wired yet)

- [ ] **Step 3: Write the implementation**

Per the grilling-session decision, path/fingerprint construction moves to
`fit()` (which already has `xdata_jax`/`ydata_jax`/`sigma_jax`/
`lower_bounds`/`upper_bounds` and computes `popsize`); `_run_cmaes`/
`_run_cmaes_single` receive them as plain parameters and never build
anything themselves.

In `fit()`, right after the existing `popsize` computation block and
before the existing `# Track wall time` / `start_time = time.perf_counter()`
lines, insert:

```python
from nlsq.global_optimization.checkpoint import (
    HPCCheckpointManager,
    compute_fingerprint,
)

checkpoint_manager = None
checkpoint_path = None
checkpoint_fingerprint = None
if self.config.checkpoint_dir is not None:
    from pathlib import Path

    checkpoint_manager = HPCCheckpointManager()
    checkpoint_path = Path(self.config.checkpoint_dir) / f"{self.config.run_id}.h5"
    checkpoint_fingerprint = compute_fingerprint(
        model_id=self.config.model_id,
        xdata=np.asarray(xdata_jax),
        ydata=np.asarray(ydata_jax),
        sigma=np.asarray(sigma_jax) if sigma_jax is not None else None,
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        config_fields={
            "popsize": popsize,
            "sigma": self.config.sigma,
            "tol_fun": self.config.tol_fun,
            "tol_x": self.config.tol_x,
            "seed": self.config.seed,
        },
    )
```

(`self.config.run_id`/`self.config.model_id` are guaranteed non-`None`
here -- Task 3's validation raises at `CMAESConfig` construction time if
`checkpoint_dir` is set without them.)

Then change the existing call

```python
best_params_unbounded, best_fitness, generations = self._run_cmaes(
    fitness_fn,
    initial_solution,
    popsize,
    n_params,
    diagnostics,
)
```

to

```python
best_params_unbounded, best_fitness, generations = self._run_cmaes(
    fitness_fn,
    initial_solution,
    popsize,
    n_params,
    diagnostics,
    checkpoint_manager=checkpoint_manager,
    checkpoint_path=checkpoint_path,
    checkpoint_fingerprint=checkpoint_fingerprint,
)
```

`_run_cmaes` gains matching keyword-only parameters and forwards them only
to `_run_cmaes_single` (never to `_run_cmaes_with_bipop` -- Task 3's guard
means checkpointing is never enabled when `restart_strategy="bipop"`, so
that function's signature is untouched in this plan):

```python
def _run_cmaes(
    self,
    fitness_fn: Callable,
    initial_solution: jax.Array,
    popsize: int,
    n_params: int,
    diagnostics: CMAESDiagnostics,
    *,
    checkpoint_manager: HPCCheckpointManager | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
) -> tuple[jax.Array, jax.Array, int]:
    if self.config.restart_strategy == "bipop":
        return self._run_cmaes_with_bipop(
            fitness_fn,
            initial_solution,
            popsize,
            n_params,
            diagnostics,
        )
    return self._run_cmaes_single(
        fitness_fn,
        initial_solution,
        popsize,
        n_params,
        diagnostics,
        checkpoint_manager=checkpoint_manager,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
```

And `_run_cmaes_single` itself:

```python
def _run_cmaes_single(
    self,
    fitness_fn: Callable,
    initial_solution: jax.Array,
    popsize: int,
    n_params: int,
    diagnostics: CMAESDiagnostics,
    *,
    checkpoint_manager: HPCCheckpointManager | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
) -> tuple[jax.Array, jax.Array, int]:
    from evosax.algorithms import CMA_ES

    from nlsq.global_optimization.checkpoint import (
        CMAESCheckpointState,
        serialize_key,
    )

    checkpointing = checkpoint_manager is not None

    logger.info(
        f"Starting CMA-ES: popsize={popsize}, max_gen={self.config.max_generations}",
    )

    es = CMA_ES(population_size=popsize, solution=initial_solution)
    params = es.default_params
    params = params.replace(std_init=self.config.sigma)

    if self.config.seed is not None:
        key = jax.random.key(self.config.seed)
    else:
        key = jax.random.key(np.random.randint(0, 2**31))
    key, subkey = jax.random.split(key)
    state = es.init(subkey, initial_solution, params)

    best_solution = initial_solution
    best_fitness = jnp.array(jnp.inf)
    convergence_reason = "max_generations"
    start_gen = 0

    # (Resume-on-entry logic added in Task 5 goes here, before the
    # milestones/loop setup below -- it would overwrite `state`, `key`,
    # `start_gen`, `best_solution`, `best_fitness`, and
    # `diagnostics.fitness_history` if a valid checkpoint exists.)

    milestones: dict[int, str] = {}
    for pct, label in ((0.25, "25%"), (0.50, "50%"), (0.75, "75%")):
        gen_idx = int(self.config.max_generations * pct)
        if gen_idx not in milestones:
            milestones[gen_idx] = label

    def _save_checkpoint(gen_idx: int, current_key: jax.Array) -> None:
        checkpoint_state = CMAESCheckpointState(
            generation_counter=gen_idx + 1,
            mean=state.mean,
            std=state.std,
            p_std=state.p_std,
            p_c=state.p_c,
            C=state.C,
            B=state.B,
            D=state.D,
            best_solution=best_solution,
            best_fitness=float(best_fitness),
            key_data=serialize_key(current_key),
            fitness_history=list(diagnostics.fitness_history),
            popsize=popsize,
        )
        checkpoint_manager.save(
            checkpoint_path, checkpoint_state, checkpoint_fingerprint
        )

    gen = start_gen - 1
    for gen in range(start_gen, self.config.max_generations):
        key, key_ask, key_tell = jax.random.split(key, 3)

        population, state = es.ask(key_ask, state, params)
        fitness = fitness_fn(population)
        state, _metrics = es.tell(key_tell, population, fitness, state, params)

        if state.best_fitness < best_fitness:
            best_fitness = state.best_fitness
            best_solution = state.best_solution

        diagnostics.fitness_history.append(float(best_fitness))

        if checkpointing and (gen + 1) % self.config.checkpoint_interval == 0:
            _save_checkpoint(gen, key)

        if float(state.std) < self.config.tol_x:
            logger.info(
                f"CMA-ES converged at generation {gen + 1}: "
                f"std={float(state.std):.2e} < tol_x={self.config.tol_x:.2e}",
            )
            convergence_reason = "xtol"
            break

        if gen + 1 in milestones:
            logger.info(
                f"CMA-ES progress {milestones[gen + 1]}: "
                f"gen={gen + 1}/{self.config.max_generations}, "
                f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.2e}",
            )

        if logger.isEnabledFor(logging.DEBUG) and (gen + 1) % 10 == 0:
            logger.debug(
                f"Generation {gen + 1}/{self.config.max_generations}: "
                f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.6e}",
            )

    if checkpointing:
        _save_checkpoint(gen, key)

    diagnostics.final_sigma = float(state.std)
    diagnostics.convergence_reason = convergence_reason
    diagnostics.total_restarts = 0

    return best_solution, best_fitness, gen + 1
```

Add `from pathlib import Path` and `from typing import Any` to
`cmaes_optimizer.py`'s top-level imports if not already present (`Any` is
already imported per the existing `from typing import TYPE_CHECKING, Any`
line; `Path` is new).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/global_optimization/test_cmaes_checkpoint_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/cmaes_optimizer.py tests/global_optimization/test_cmaes_checkpoint_integration.py
git commit -m "feat(global_optimization): periodic checkpoint save in CMA-ES single-run loop"
```

---

### Task 5: Resume-on-entry + determinism test

**Files:**
- Modify: `nlsq/global_optimization/cmaes_optimizer.py:_run_cmaes_single`
- Test: `tests/global_optimization/test_cmaes_checkpoint_integration.py` (append)

**Interfaces:**
- Consumes: `HPCCheckpointManager.load` (Task 2), `deserialize_evosax_state`/`deserialize_key` (Task 1).
- Produces: `_run_cmaes_single` resumes from a valid on-disk checkpoint at entry instead of starting fresh; satisfies spec FR6 and NFR1.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/global_optimization/test_cmaes_checkpoint_integration.py
def test_resume_matches_uninterrupted_run(tmp_path):
    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))
    bounds = ([0.0, 0.0], [10.0, 2.0])
    shared_kwargs = dict(
        max_generations=30,
        restart_strategy="none",
        seed=11,
        model_id="exponential_decay_v1",
    )

    # Uninterrupted reference run.
    ref_config = CMAESConfig(**shared_kwargs)
    ref_optimizer = CMAESOptimizer(config=ref_config)
    ref_result = ref_optimizer.fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    # Interrupted run: only 15 generations, saved via checkpoint_interval=15.
    part_dir = tmp_path / "part"
    part_config = CMAESConfig(
        **shared_kwargs,
        max_generations=15,
        checkpoint_dir=str(part_dir),
        checkpoint_interval=15,
        run_id="resume-test",
    )
    part_optimizer = CMAESOptimizer(config=part_config)
    part_optimizer.fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    # Resumed run: same checkpoint_dir/run_id, max_generations back to 30 --
    # must run exactly 15 more generations and land on the same result as
    # the uninterrupted 30-generation run.
    resume_config = CMAESConfig(
        **shared_kwargs,
        checkpoint_dir=str(part_dir),
        checkpoint_interval=15,
        run_id="resume-test",
    )
    resume_optimizer = CMAESOptimizer(config=resume_config)
    resume_result = resume_optimizer.fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    np.testing.assert_array_equal(resume_result["popt"], ref_result["popt"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/global_optimization/test_cmaes_checkpoint_integration.py::test_resume_matches_uninterrupted_run -v`
Expected: FAIL — resumed run currently restarts from generation 0 (ignores existing checkpoint), landing on 15 total generations of its own vs. the 30-generation reference; `popt` differs.

- [ ] **Step 3: Write the implementation**

Insert this block in `_run_cmaes_single`, right after `state = es.init(subkey, initial_solution, params)` and before `best_solution = initial_solution` (replacing the placeholder comment left in Task 4):

```python
checkpoint_bak_path = None
if checkpointing:
    checkpoint_bak_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
# Check for .bak too, not just the primary: if a crash deleted or never
# finished writing the primary but a prior successful save's rotated .bak
# survives, resume must still attempt it -- gating on the primary alone
# would silently skip HPCCheckpointManager.load()'s own mandatory .bak
# fallback (FR8) entirely and start fresh instead (caught in the third
# review pass).
if checkpointing and (checkpoint_path.exists() or checkpoint_bak_path.exists()):
    loaded = checkpoint_manager.load(checkpoint_path, checkpoint_fingerprint)
    state = deserialize_evosax_state(
        {
            "generation_counter": loaded.generation_counter,
            "mean": loaded.mean,
            "std": loaded.std,
            "p_std": loaded.p_std,
            "p_c": loaded.p_c,
            "C": loaded.C,
            "B": loaded.B,
            "D": loaded.D,
            "best_solution": loaded.best_solution,
            "best_fitness": loaded.best_fitness,
        },
        state,
    )
    key = deserialize_key(loaded.key_data)
    best_solution = loaded.best_solution
    best_fitness = jnp.asarray(loaded.best_fitness)
    diagnostics.fitness_history = list(loaded.fitness_history)
    start_gen = loaded.generation_counter
    logger.info(
        f"Resumed CMA-ES from checkpoint at generation {start_gen} "
        f"({checkpoint_path})",
    )
```

Extend Task 4's `from nlsq.global_optimization.checkpoint import (...)`
block inside `_run_cmaes_single` to also pull in `deserialize_evosax_state`
and `deserialize_key` (still no `HPCCheckpointManager`/`compute_fingerprint`
import here -- those stay in `fit()` only, per the grilling-session
architecture decision):

```python
from nlsq.global_optimization.checkpoint import (
    CMAESCheckpointState,
    deserialize_evosax_state,
    deserialize_key,
    serialize_key,
)
```

Also change the loop's exit condition to guard against a checkpoint whose `generation_counter` already reached `max_generations` (fully-converged-and-resumed edge case): if `start_gen >= self.config.max_generations`, skip the `for` loop entirely and go straight to `diagnostics.final_sigma = float(state.std)` using the loaded state's `std`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/global_optimization/test_cmaes_checkpoint_integration.py -v`
Expected: PASS (both tests in the file)

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/cmaes_optimizer.py tests/global_optimization/test_cmaes_checkpoint_integration.py
git commit -m "feat(global_optimization): resume CMA-ES from checkpoint on entry"
```

---

### Task 6: Preemption signal handling (flag-only handler, safe-point save)

**Files:**
- Modify: `nlsq/global_optimization/cmaes_optimizer.py:_run_cmaes_single`
- Test: `tests/global_optimization/test_cmaes_signal_handling.py`

**Interfaces:**
- Produces: `fit()` (not `_run_cmaes_single` -- moved per the grilling-session
  architecture decision, same reasoning as Task 4's path/fingerprint move)
  creates a `threading.Event`, registers `SIGTERM`/`SIGUSR1` handlers that
  set it -- only on the main thread, and only when checkpointing is enabled
  -- and wraps the `_run_cmaes(...)` call in `try`/`finally` to restore the
  original handlers afterward. `_run_cmaes`/`_run_cmaes_single` receive the
  event as a plain parameter; the loop only checks it once per completed
  generation and, if set, saves and raises `CMAESPreempted` (new exception
  class, exported from `cmaes_optimizer.py`). Neither function touches the
  `signal` module.

- [ ] **Step 1: Write the failing test**

```python
# tests/global_optimization/test_cmaes_signal_handling.py
import signal
import subprocess
import sys
import textwrap
import time

import pytest


def test_sigterm_mid_run_leaves_valid_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "sigterm-test.h5"
    script = textwrap.dedent(
        f"""
        import numpy as np
        from nlsq.global_optimization.cmaes_config import CMAESConfig
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def exponential_decay(x, a, b):
            import jax.numpy as jnp
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 2000)
        y = np.asarray(exponential_decay(x, 2.5, 0.5))
        config = CMAESConfig(
            max_generations=100000,
            restart_strategy="none",
            seed=3,
            model_id="exponential_decay_v1",
            checkpoint_dir={str(tmp_path)!r},
            checkpoint_interval=1,
            run_id="sigterm-test",
        )
        optimizer = CMAESOptimizer(config=config)
        try:
            optimizer.fit(exponential_decay, x, y, bounds=([0.0, 0.0], [10.0, 2.0]))
        except SystemExit as e:
            print(f"EXIT_CODE:{{e.code}}")
        """,
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Bounded readiness handshake instead of a fixed sleep: cold JAX/evosax
    # imports and JIT warmup can take longer than any single fixed delay,
    # and the SIGTERM handler is only installed once fit() actually starts
    # checkpointing -- signalling before that point would hit Python's
    # default SIGTERM disposition (process exits -15, not 75). Wait for
    # generation 1's checkpoint (checkpoint_interval=1) to prove the loop
    # -- and therefore the handler -- is live.
    deadline = time.monotonic() + 60.0
    while not checkpoint_path.exists():
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"process exited before writing any checkpoint: "
                f"returncode={proc.returncode} stdout={stdout} stderr={stderr}",
            )
        if time.monotonic() > deadline:
            proc.kill()
            pytest.fail("timed out waiting for first checkpoint to appear")
        time.sleep(0.05)

    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=30)

    assert "EXIT_CODE:75" in stdout, f"stderr={stderr}"
    assert checkpoint_path.exists()
    import h5py

    with h5py.File(checkpoint_path, "r") as f:
        assert bool(f.attrs["completion_marker"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/global_optimization/test_cmaes_signal_handling.py -v`
Expected: FAIL — no `SystemExit`/exit code 75 (signal handling not wired yet); process either runs to completion or is killed uncleanly.

- [ ] **Step 3: Write the implementation**

Add near the top of `cmaes_optimizer.py`:

```python
class CMAESPreempted(SystemExit):
    """Raised when a preemption signal (SIGTERM/SIGUSR1) is caught after a
    checkpoint has been safely written. Exit code 75 lets a wrapping HPC
    resubmission script distinguish a clean checkpointed stop from a crash."""

    def __init__(self, generation: int) -> None:
        super().__init__(75)
        self.generation = generation
```

Add `__all__ = ["CMAESOptimizer", "CMAESPreempted"]`. Add `import signal` and
`import threading` to the module's top-level imports (not scoped inside
`fit()` -- it's referenced later in a `finally` block, and importing at
module level is the standard, unambiguous pattern used elsewhere in this
codebase).

In `fit()`, extend the checkpoint setup block added in Task 4 (right after
building `checkpoint_manager`/`checkpoint_path`/`checkpoint_fingerprint`):

```python
preemption_event = None
previous_handlers: dict[int, Any] = {}
if checkpoint_manager is not None:
    if threading.current_thread() is threading.main_thread():
        preemption_event = threading.Event()

        def _handle_preemption(signum: int, frame: Any) -> None:
            preemption_event.set()

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[sig] = signal.signal(sig, _handle_preemption)
    else:
        # signal.signal() raises ValueError off the main thread. Periodic
        # interval saves (Task 4) remain the crash-safety net when
        # CMAESOptimizer.fit() is called from a worker thread; this is a
        # documented reduction in coverage, not a silent no-op -- log it.
        logger.warning(
            "Checkpointing enabled but fit() was not called on the main "
            "thread: SIGTERM/SIGUSR1 preemption handling is unavailable "
            "here (Python's signal.signal() requires the main thread). "
            "Periodic interval saves still apply.",
        )
```

Change the existing `_run_cmaes(...)` call (already updated in Task 4 to
pass the checkpoint parameters) to also pass `preemption_event=
preemption_event`, and wrap the call in `try`/`finally` to restore
whatever handlers were actually installed:

```python
try:
    best_params_unbounded, best_fitness, generations = self._run_cmaes(
        fitness_fn,
        initial_solution,
        popsize,
        n_params,
        diagnostics,
        checkpoint_manager=checkpoint_manager,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
        preemption_event=preemption_event,
    )
finally:
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
```

(`CMAESPreempted` is a `SystemExit` subclass, so it propagates through this
`finally` and out of `fit()` uncaught, exactly like the test above expects
-- no explicit `except CMAESPreempted` needed here.)

`_run_cmaes` and `_run_cmaes_single` both gain a matching
`preemption_event: threading.Event | None = None` keyword-only parameter,
threaded through the same way as the Task 4 checkpoint parameters (add it
to `_run_cmaes`'s signature and its `_run_cmaes_single(...)` call; add it to
`_run_cmaes_single`'s signature).

Inside the loop, right after the `diagnostics.fitness_history.append(...)`
line and before the existing `if checkpointing and (gen + 1) % ...` save
check, add:

```python
if preemption_event is not None and preemption_event.is_set():
    jax.block_until_ready((state.mean, state.C, state.best_solution))
    _save_checkpoint(gen, key)
    raise CMAESPreempted(gen + 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/global_optimization/test_cmaes_signal_handling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlsq/global_optimization/cmaes_optimizer.py tests/global_optimization/test_cmaes_signal_handling.py
git commit -m "feat(global_optimization): flag-based SIGTERM/SIGUSR1 preemption handling"
```

---

### Task 7: Wire `workflow='hpc'` to pass checkpoint kwargs through for the CMA-ES route

**Files:**
- Modify: `nlsq/core/minpack.py` (`_fit_with_hpc`, currently lines ~1320-1420; `_fit_global_cmaes`)
- Test: `tests/streaming/test_workflow_presets.py` (append)

**Interfaces:**
- Consumes: `CMAESConfig` checkpoint fields (Task 3).
- Produces: (a) `fit()`'s `method` parameter actually forces the CMA-ES/
  multi-start route via `MethodSelector`, for every `workflow='auto_global'`/
  `'hpc'` caller (spec FR9) -- previously it was silently dropped. (b) When
  `workflow='hpc'` resolves to the CMA-ES route (not multistart/chunked/
  streaming), `checkpoint_dir`/`checkpoint_interval` passed to `fit()` are
  forwarded into a `CMAESConfig` instead of being discarded; the existing
  `UserWarning` in `_fit_with_hpc` is narrowed to only fire for the routes
  that still don't support it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/streaming/test_workflow_presets.py
def test_hpc_cmaes_route_actually_checkpoints(tmp_path):
    """workflow='hpc' with bounds narrow enough to select CMA-ES must
    forward checkpoint_dir into CMAESConfig instead of discarding it.

    Must pass cmaes_config with restart_strategy="none" explicitly --
    CMAESConfig defaults to restart_strategy="bipop", and Task 3's
    validation correctly raises NotImplementedError for bipop +
    checkpoint_dir (spec FR3). Omitting this override would make this
    test fail on that guard rather than exercising the checkpoint path.
    """
    from nlsq import fit
    from nlsq.global_optimization.cmaes_config import CMAESConfig

    x = jnp.linspace(0, 5, 100)
    y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = fit(
            model,
            x,
            y,
            p0=[1.0, 0.5],
            workflow="hpc",
            bounds=([0.0, 0.0], [10.0, 10.0]),
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=5,
            run_id="hpc-e2e-test",
            model_id="hpc-e2e-model",
            seed=1,
            method="cmaes",
            cmaes_config=CMAESConfig(restart_strategy="none"),
        )

    assert result is not None
    assert (tmp_path / "hpc-e2e-test.h5").exists()


def test_auto_global_method_cmaes_actually_selects_cmaes():
    """FR9, isolated from checkpointing: workflow='auto_global' with
    method='cmaes' must force the CMA-ES route even when the data's scale
    ratio would otherwise auto-select multi-start (bounds here are
    single-scale, ratio ~1, well under MethodSelector's ~1000x threshold).
    """
    from nlsq import fit

    x = jnp.linspace(0, 5, 100)
    y = 2.5 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.01, 100)

    result = fit(
        model,
        x,
        y,
        p0=[1.0, 0.5],
        workflow="auto_global",
        bounds=([0.0, 0.0], [10.0, 10.0]),
        method="cmaes",
    )

    assert result is not None
    # _fit_global_cmaes sets result["cmaes_diagnostics"] (minpack.py:1518-1519);
    # the multi-start route instead sets result["multistart_diagnostics"]
    # (minpack.py:2132/2755) -- these two keys are mutually exclusive
    # discriminators for which route actually ran.
    assert "cmaes_diagnostics" in result
    assert "multistart_diagnostics" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/streaming/test_workflow_presets.py -v -k "hpc_cmaes_route_actually_checkpoints or auto_global_method_cmaes"`
Expected: FAIL — the first on the still-unconditional `UserWarning`
(`_fit_with_hpc` warns regardless today); the second because `method`
never reaches `MethodSelector`, so multi-start (not CMA-ES) is selected
and no `cmaes_diagnostics` appears on the result.

- [ ] **Step 3: Write the implementation**

**First, fix the `method`-threading gap (spec FR9)** -- without this,
`method="cmaes"` in the test above silently does nothing and the test
would fail on the "route doesn't support checkpointing" warning instead of
actually exercising the checkpoint path (found in code review: `fit()` has
an explicit `method: str | None = None` parameter, but its dispatch calls
to `_fit_with_auto_global`/`_fit_with_hpc` only forward `**kwargs` --
`method` is a named parameter, not part of `kwargs`, so it never reaches
either function; `_fit_with_auto_global` then hardcodes
`requested_method="auto"` to `MethodSelector.select()` regardless).

In `fit()`'s two dispatch call sites, add `method=method,`:

```python
if workflow_lower == "auto_global":
    # Memory-aware global optimization (requires bounds)
    return _fit_with_auto_global(
        f=f,
        xdata=xdata_arr,
        ydata=ydata_arr,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        n_points=n_points,
        n_params=n_params,
        goal=goal_enum,
        method=method,
        **kwargs,
    )

if workflow_lower == "hpc":
    # HPC workflow with checkpointing (wraps auto_global)
    return _fit_with_hpc(
        f=f,
        xdata=xdata_arr,
        ydata=ydata_arr,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        n_points=n_points,
        n_params=n_params,
        goal=goal_enum,
        method=method,
        **kwargs,
    )
```

Add `method: str | None = None` to both `_fit_with_auto_global`'s and
`_fit_with_hpc`'s signatures (same pattern already used by the existing
`_fit_with_config`/`_fit_with_preset` functions in this file). In
`_fit_with_hpc`'s own call into `_fit_with_auto_global`, add `method=method,`.
In `_fit_with_auto_global`, change:

```python
global_method = method_selector.select(
    requested_method="auto",
    lower_bounds=lb,
    upper_bounds=ub,
)
```

to:

```python
global_method = method_selector.select(
    requested_method=method or "auto",
    lower_bounds=lb,
    upper_bounds=ub,
)
```

(`MethodSelector.select()` already handles `requested_method="cmaes"`/
`"multi-start"` explicitly and falls through to scale-ratio auto-selection
for anything else, including `None` -- no change needed there.)

**Then, wire the checkpoint kwargs through.** In `_fit_with_hpc` (`minpack.py`), replace the unconditional warn block:

```python
checkpoint_dir = kwargs.pop("checkpoint_dir", None)
checkpoint_interval = kwargs.pop("checkpoint_interval", 5)
run_id = kwargs.pop("run_id", None)
model_id = kwargs.pop("model_id", None)

# Checkpoint/resume is implemented only for the CMA-ES route
# (restart_strategy="none") -- see
# docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md.
# Whether the CMA-ES route is even selected depends on
# MethodSelector's choice inside _fit_with_auto_global, which this
# function doesn't know yet at this point -- so thread the
# checkpoint kwargs through unconditionally and let
# _fit_global_cmaes wire them into CMAESConfig; every OTHER route
# (multistart/chunked/streaming) still needs the warning, and
# _fit_global_cmaes doesn't see kwargs meant for those routes, so
# the warning must be emitted from wherever the route is actually
# decided. Since that decision lives inside _fit_with_auto_global's
# MethodSelector (not yet reached here), pass a marker through
# kwargs and let _fit_global_cmaes consume it silently while
# _fit_with_auto_global's other route functions keep warning.
if checkpoint_dir is not None:
    kwargs["_hpc_checkpoint_dir"] = checkpoint_dir
    kwargs["_hpc_checkpoint_interval"] = checkpoint_interval
    kwargs["_hpc_run_id"] = run_id
    kwargs["_hpc_model_id"] = model_id
```

In `_fit_global_cmaes` (`minpack.py`, the function that constructs `CMAESOptimizer`), pop the markers and wire them into `cmaes_config`:

```python
checkpoint_dir = kwargs.pop("_hpc_checkpoint_dir", None)
checkpoint_interval = kwargs.pop("_hpc_checkpoint_interval", 5)
run_id = kwargs.pop("_hpc_run_id", None)
model_id = kwargs.pop("_hpc_model_id", None)
seed = kwargs.pop("seed", None)

if checkpoint_dir is not None:
    cmaes_config = dataclasses.replace(
        cmaes_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        run_id=run_id,
        model_id=model_id,
        seed=seed if seed is not None else cmaes_config.seed,
    )
```

(`_fit_global_cmaes` already does `import dataclasses` per the earlier grep of its body.)

For every OTHER route inside `_fit_with_auto_global` that is NOT `_fit_global_cmaes` (the multistart/`curve_fit(multistart=True)`, chunked/`LargeDatasetFitter`, streaming/`AdaptiveHybridStreamingOptimizer` branches), add, right before dispatching to that branch:

```python
if kwargs.pop("_hpc_checkpoint_dir", None) is not None:
    warnings.warn(
        "workflow='hpc': checkpoint_dir was provided, but this route "
        "(non-CMA-ES) does not support checkpoint/crash-recovery yet "
        "-- no checkpoint file will be written. Only the CMA-ES route "
        "(restart_strategy='none') supports checkpointing currently.",
        UserWarning,
        stacklevel=2,
    )
    kwargs.pop("_hpc_checkpoint_interval", None)
    kwargs.pop("_hpc_run_id", None)
    kwargs.pop("_hpc_model_id", None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/streaming/test_workflow_presets.py -v`
Expected: PASS — includes the pre-existing `test_hpc_requires_bounds`, `test_hpc_with_bounds_succeeds`, `test_hpc_accepts_checkpoint_parameters`, `test_hpc_checkpoint_dir_warns_not_implemented` (still passes for non-CMA-ES route selection), and the two new tests `test_hpc_cmaes_route_actually_checkpoints` and `test_auto_global_method_cmaes_actually_selects_cmaes`.

- [ ] **Step 5: Commit**

```bash
git add nlsq/core/minpack.py tests/streaming/test_workflow_presets.py
git commit -m "feat(core): wire workflow='hpc' checkpoint_dir into CMA-ES route"
```

---

### Task 8: Save-latency benchmark + changelog

**Files:**
- Create: `benchmarks/cmaes_checkpoint_latency.py`
- Modify: `CHANGELOG.md`

**Interfaces:** None (measurement + docs only).

- [ ] **Step 1: Write the benchmark script**

```python
# benchmarks/cmaes_checkpoint_latency.py
"""Measure HPCCheckpointManager.save() latency at representative n_params.

Run: python benchmarks/cmaes_checkpoint_latency.py
Records numbers into docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md
NFR2 -- update that file's NFR2 section with the results after running this.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization.checkpoint import (
    CMAESCheckpointState,
    HPCCheckpointManager,
    compute_fingerprint,
)


def _state_for(n_params: int) -> CMAESCheckpointState:
    return CMAESCheckpointState(
        generation_counter=42,
        mean=jnp.zeros(n_params),
        std=jnp.array(0.1),
        p_std=jnp.zeros(n_params),
        p_c=jnp.zeros(n_params),
        C=jnp.eye(n_params),
        B=jnp.eye(n_params),
        D=jnp.ones(n_params),
        best_solution=jnp.zeros(n_params),
        best_fitness=0.001,
        key_data=np.array([1, 2], dtype=np.uint32),
        fitness_history=list(range(100)),
        popsize=int(4 + 3 * np.log(n_params)),
    )


def main() -> None:
    manager = HPCCheckpointManager()
    fp = compute_fingerprint(
        model_id="bench",
        xdata=np.zeros(1000),
        ydata=np.zeros(1000),
        sigma=None,
        bounds=(np.zeros(10), np.ones(10)),
        config_fields={
            "popsize": 8,
            "sigma": 0.5,
            "tol_fun": 1e-8,
            "tol_x": 1e-8,
            "seed": 1,
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        for n_params in (10, 100, 1000):
            state = _state_for(n_params)
            path = Path(tmpdir) / f"bench_{n_params}.h5"
            n_runs = 5
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                manager.save(path, state, fp)
                times.append(time.perf_counter() - start)
            print(
                f"n_params={n_params}: mean={np.mean(times) * 1000:.2f}ms, "
                f"max={np.max(times) * 1000:.2f}ms"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record results**

Run: `python benchmarks/cmaes_checkpoint_latency.py`

Copy the printed `n_params=... mean=...ms` lines into the NFR2 section of
`docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md`, replacing
its "no hard latency requirement is set here pending that data" sentence
with the actual measured numbers and a one-line verdict (e.g. "well under a
generation's compute time even at n_params=1000; no wall-clock-interval
option needed for v1").

- [ ] **Step 3: Update CHANGELOG.md**

Add under `## [Unreleased]` → `### Added`:

```markdown
- `CMAESOptimizer` (with `restart_strategy="none"`) supports checkpoint/
  resume via `CMAESConfig.checkpoint_dir`/`checkpoint_interval`/`run_id`/
  `model_id`, and `workflow='hpc'` now forwards `checkpoint_dir` into it
  instead of discarding it with a warning. BIPOP restarts
  (`restart_strategy="bipop"`, the default) and the multistart/chunked/
  streaming `workflow='hpc'` routes still raise/warn as not-yet-implemented
  -- see `docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md`
  for scope.

### Fixed
- `fit()`'s `method` parameter was silently dropped for `workflow='auto_global'`
  and `workflow='hpc'` -- `method="cmaes"`/`"multi-start"` had no effect and
  route selection always fell back to automatic scale-ratio-based selection.
  `method` now actually reaches `MethodSelector`, for every caller of those
  two workflows, not only when checkpointing is enabled.
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/cmaes_checkpoint_latency.py CHANGELOG.md docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md
git commit -m "docs(global_optimization): checkpoint latency benchmark + changelog entry"
```

---

## Self-Review

**Spec coverage:** FR1 → Task 3 (fields). FR2 → Task 3 (seed validation). FR3 → Task 3 (bipop guard). FR4 → Task 3 (model_id + run_id validation). FR5 → Task 4 (periodic + final save). FR6 → Task 5 (resume + fingerprint check via `HPCCheckpointManager.load`, Task 2). FR7 → Task 6 (signal handling, main-thread-guarded, registered in `fit()`). FR8 → Task 2 (atomic write, directory fsync, mandatory `.bak` fallback, rotation, completion marker). FR9 → Task 7 (`method` threading fix). NFR1 → Task 5's `test_resume_matches_uninterrupted_run`. NFR2 → Task 8. Scope §2's non-goals → no task touches those files; Task 7 explicitly preserves the warning for them.

**Placeholder scan:** every step has real, complete code; no "TODO"/"similar to Task N"/"add validation" left unfilled.

**Type consistency:** `CMAESCheckpointState` fields defined in Task 1 are used identically (same names) in Tasks 2, 4, 5, 8. `HPCCheckpointManager.save(path, state, fingerprint)`/`.load(path, expected_fingerprint)` signatures from Task 2 are called identically in Tasks 4/5. `compute_fingerprint(...)` keyword names match between Task 2's definition and Tasks 4/7's call sites. `checkpoint_manager`/`checkpoint_path`/`checkpoint_fingerprint`/`preemption_event` are threaded identically through `fit()` → `_run_cmaes` → `_run_cmaes_single` across Tasks 4 and 6.

**Resolved during grilling-session review (previously an open limitation):** the separation-of-concerns gap flagged by the three-brain Agy pass -- `_run_cmaes_single` building its own path/fingerprint/signal handlers instead of receiving them pre-built -- is fixed in this revision. `fit()` now owns all construction (`HPCCheckpointManager`, checkpoint path, fingerprint, `threading.Event`, signal registration/restoration); `_run_cmaes`/`_run_cmaes_single` only receive these as parameters and call `.save()`/`.load()`/check the event. No known architectural debt remains from that review round.

**Two independent Codex correctness passes** (after the plan changed between them) found and this revision fixes: the evosax `State.replace()` vs `._replace()` API error (Task 1), a broken end-to-end test missing `restart_strategy="none"` (Task 7), a signal-handler leak on unhandled exceptions (Task 6, now via `fit()`'s own `try`/`finally`), a missing `.bak` load fallback (Task 2), a missing containing-directory fsync (Task 2), a fingerprint format that stored only a bounds hash instead of the spec-mandated raw arrays (Task 2), a racy fixed-`sleep()` in the SIGTERM test (Task 6), a missing main-thread guard on signal registration (Task 6), and the `method`-parameter-never-threaded-through bug that would have made Task 7's own end-to-end test fail (Task 7, new FR9).

**Third review pass (Agy) found and this revision fixes two further real bugs introduced by the second round's own restructuring** -- a sharp reminder that a large refactor needs re-review, not just its originating findings fixed:
- **Critical regression:** rewriting `_run_cmaes_single`'s key initialization accidentally deleted the `seed is None` → OS-entropy fallback, hardcoding `jax.random.key(self.config.seed)` unconditionally. Since `seed` is only *required* when `checkpoint_dir` is set, this would have broken every ordinary (non-checkpointed) CMA-ES fit with `TypeError` the moment it ran, not just checkpoint-enabled ones. Restored the `if seed is not None: ... else: jax.random.key(np.random.randint(...))` branch in Task 4.
- **High:** Task 5's resume-check gated on `checkpoint_path.exists()` alone (the primary file). If a crash removed the primary but the previous successful save's rotated `.bak` survived, resume would silently skip loading anything and start fresh -- bypassing `HPCCheckpointManager.load()`'s own mandatory `.bak` fallback (FR8) before it ever got a chance to run. Now checks for either the primary or `.bak` existing.
- **Minor, also fixed:** `compute_fingerprint`'s `n_params` field was stringified (`str(len(...))`) though the spec's own file-format table documents it as `int`; `import signal` was scoped inside a conditional block in `fit()` though referenced later in an unconditional `finally` -- moved to the module's top-level imports for clarity even though Python's function-level (not block) scoping meant it wasn't a guaranteed crash.

Agy separately suggested splitting Task 4 (it touches `fit()`, `_run_cmaes`, and `_run_cmaes_single`, and is revisited again in Tasks 5/6) into smaller sub-tasks. Not applied: those three functions change together as one coherent unit (the parameter-threading interface between them), and splitting purely by function would scatter one interface change across multiple tasks without making any single task meaningfully smaller to review -- the actual growth driver is Tasks 5/6 layering more behavior onto the same call chain, which is already reflected in them being separate tasks.
