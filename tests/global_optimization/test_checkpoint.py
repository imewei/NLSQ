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
