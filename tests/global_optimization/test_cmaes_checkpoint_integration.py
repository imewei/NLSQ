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


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))
    bounds = ([0.0, 0.0], [10.0, 2.0])
    shared_kwargs = {
        "restart_strategy": "none",
        "seed": 11,
        "model_id": "exponential_decay_v1",
    }

    # Uninterrupted reference run.
    ref_config = CMAESConfig(**shared_kwargs, max_generations=30)
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
        max_generations=30,
        checkpoint_dir=str(part_dir),
        checkpoint_interval=15,
        run_id="resume-test",
    )
    # The `popt` equality alone can't tell a real resume apart from a naive
    # restart-from-scratch-with-the-same-seed: both run 30 generations from
    # the identical key(seed=11) and JAX's PRNG splitting is a pure function
    # of the key, so they'd land on the same result even if the checkpoint
    # were silently ignored. Spy on `HPCCheckpointManager.load` to prove the
    # resume path actually executed, not just that the outputs coincide.
    load_calls = []
    original_load = HPCCheckpointManager.load

    def _spy_load(self, path, expected_fingerprint):
        load_calls.append(path)
        return original_load(self, path, expected_fingerprint)

    monkeypatch.setattr(HPCCheckpointManager, "load", _spy_load)

    resume_optimizer = CMAESOptimizer(config=resume_config)
    resume_result = resume_optimizer.fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    assert len(load_calls) == 1

    np.testing.assert_array_equal(resume_result["popt"], ref_result["popt"])
