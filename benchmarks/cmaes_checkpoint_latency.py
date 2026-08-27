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
