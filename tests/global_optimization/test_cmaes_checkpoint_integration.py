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
    # checkpoint_interval=5 over max_generations=20 means at least one
    # interval save landed before the final unconditional save -- pins the
    # every-N-generations cadence (not just "some save happened").
    assert checkpoint_path.with_suffix(".h5.bak").exists()

    with __import__("h5py").File(checkpoint_path, "r") as f:
        assert bool(f.attrs["completion_marker"])
        # Loop ran to max_generations=20 (no early convergence expected
        # this fast) -- final saved generation_counter should be exactly 20.
        assert int(f["state"]["generation_counter"][()]) == 20


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
    part_result = part_optimizer.fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )
    part_history = part_result["cmaes_diagnostics"]["fitness_history"]

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
    # `load_calls == 1` only proves load() was *called*, not that its return
    # value was actually threaded into `state`/`key` -- a regression that
    # called load() then ignored the result would still reproduce ref_result
    # bit-exactly here (same seed, same total generation count). Comparing
    # the interrupted run's own generation-by-generation fitness_history
    # against the resumed run's first 15 entries can only match if the
    # loaded state was genuinely picked up: a fresh restart's trajectory
    # from generation 0 would not coincide with the *partial* run's history
    # the way it coincides with the *complete* reference run's final popt.
    resume_history = resume_result["cmaes_diagnostics"]["fitness_history"]
    assert len(resume_history) == 30
    np.testing.assert_array_equal(resume_history[:15], part_history)


def test_resume_fires_when_primary_missing_but_bak_present(tmp_path, monkeypatch):
    """fit()-level resume gate (`checkpoint_path.exists() or
    checkpoint_bak_path.exists()`) must actually trigger a load() call when
    only `.bak` survives -- gating on the primary alone would silently skip
    resume in exactly the crash-mid-rotation window `.bak` exists to cover.
    Deleting the `or` clause from that gate would leave this test failing
    (load() never called) while every other test in this file stays green."""
    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))
    bounds = ([0.0, 0.0], [10.0, 2.0])
    checkpoint_dir = tmp_path / "bak-resume"

    part_config = CMAESConfig(
        restart_strategy="none",
        seed=13,
        model_id="exponential_decay_v1",
        max_generations=10,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=5,
        run_id="bak-only-test",
    )
    CMAESOptimizer(config=part_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )
    checkpoint_path = checkpoint_dir / "bak-only-test.h5"
    bak_path = checkpoint_path.with_suffix(".h5.bak")
    assert checkpoint_path.exists()
    assert bak_path.exists()

    # Simulate a crash between save()'s two renames: the primary is gone,
    # only the rotated .bak survives.
    checkpoint_path.unlink()

    load_calls = []
    original_load = HPCCheckpointManager.load

    def _spy_load(self, path, expected_fingerprint):
        load_calls.append(path)
        return original_load(self, path, expected_fingerprint)

    monkeypatch.setattr(HPCCheckpointManager, "load", _spy_load)

    resume_config = CMAESConfig(
        restart_strategy="none",
        seed=13,
        model_id="exponential_decay_v1",
        max_generations=20,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=5,
        run_id="bak-only-test",
    )
    CMAESOptimizer(config=resume_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    assert len(load_calls) == 1


def test_resume_from_already_converged_checkpoint_is_a_noop(tmp_path):
    """Resuming with max_generations already equal to the checkpoint's
    saved generation_counter must run zero new generations without raising
    (the `gen = start_gen - 1` / `range(start_gen, max_generations)` empty-
    iteration path) and must not needlessly rewrite an identical
    checkpoint (the `gen >= start_gen` guard on the post-loop save)."""
    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))
    bounds = ([0.0, 0.0], [10.0, 2.0])
    checkpoint_dir = tmp_path / "converged-resume"

    part_config = CMAESConfig(
        restart_strategy="none",
        seed=17,
        model_id="exponential_decay_v1",
        max_generations=10,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=10,
        run_id="converged-test",
    )
    CMAESOptimizer(config=part_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )
    checkpoint_path = checkpoint_dir / "converged-test.h5"
    mtime_before = checkpoint_path.stat().st_mtime_ns

    # Resume with the same max_generations as the checkpoint's own
    # generation_counter -- the loop must run zero iterations.
    resume_config = CMAESConfig(
        restart_strategy="none",
        seed=17,
        model_id="exponential_decay_v1",
        max_generations=10,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=10,
        run_id="converged-test",
    )
    result = CMAESOptimizer(config=resume_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    assert result["cmaes_diagnostics"]["total_generations"] == 10
    assert checkpoint_path.stat().st_mtime_ns == mtime_before
