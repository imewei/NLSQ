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
    from evosax.algorithms.distribution_based.cma_es import (  # type: ignore[import-not-found]
        State as EvosaxState,
    )

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
    import jax.numpy as jnp

    return jax.random.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32))


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

    replacements: dict[str, Any] = {
        name: jnp.asarray(d[name]) for name in _EVOSAX_ARRAY_FIELDS
    }
    replacements["generation_counter"] = int(d["generation_counter"])  # type: ignore[typeddict-item]
    return template_state.replace(**replacements)
