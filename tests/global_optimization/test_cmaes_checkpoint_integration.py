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
    # NOTE: neither `load_calls == 1` nor the fitness_history comparison
    # below actually discriminates "loaded state genuinely used" from
    # "load() called then ignored" -- CMA-ES's trajectory is a pure
    # function of (seed, popsize, initial_solution), independent of
    # max_generations, so ref/part/resume all sharing seed=11 makes
    # part_history a prefix of ref_history regardless of whether resume
    # actually threads the loaded state through. See
    # test_resume_genuinely_uses_loaded_state_not_a_fresh_restart below
    # for the test that actually proves this via sentinel injection.
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


def test_resume_genuinely_uses_loaded_state_not_a_fresh_restart(tmp_path):
    """Neither `load_calls == 1` (proves load() was called) nor a
    fitness_history-prefix comparison (CMA-ES's trajectory is a pure
    function of seed/popsize/initial_solution, independent of
    max_generations -- confirmed empirically: a fresh restart with the
    same seed reproduces the interrupted run's own history regardless of
    whether resume ever executes) can distinguish "loaded state genuinely
    threaded through" from "load() called, then its return value ignored
    and the optimizer restarted from scratch." This test can: it injects
    an artificial best_fitness into the on-disk checkpoint that is
    strictly better than any value CMA-ES could organically reach on this
    problem in the generations available, then resumes. Only a genuine
    load can produce that exact sentinel in the final result -- a fresh
    restart could never beat it, so it would report its own (much worse,
    organically-reached) best_fitness instead."""
    import h5py

    x = np.linspace(0, 5, 50)
    y = np.asarray(exponential_decay(x, 2.5, 0.5))
    bounds = ([0.0, 0.0], [10.0, 2.0])
    checkpoint_dir = tmp_path / "sentinel-resume"

    part_config = CMAESConfig(
        restart_strategy="none",
        seed=23,
        model_id="exponential_decay_v1",
        max_generations=15,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=15,
        run_id="sentinel-test",
    )
    CMAESOptimizer(config=part_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )
    checkpoint_path = checkpoint_dir / "sentinel-test.h5"
    assert checkpoint_path.exists()

    # exponential_decay's SSR-based fitness is always >= 0 for real data;
    # a negative sentinel is organically unreachable in this problem, so
    # if it survives to the final result, the on-disk value was genuinely
    # loaded and threaded into both the tracked best_solution/best_fitness
    # and evosax's own internal state.best_fitness (best_fitness is one of
    # the fields serialize/deserialize_evosax_state round-trips).
    sentinel_fitness = -1_000_000.0
    with h5py.File(checkpoint_path, "r+") as f:
        f["state"]["best_fitness"][...] = sentinel_fitness

    resume_config = CMAESConfig(
        restart_strategy="none",
        seed=23,
        model_id="exponential_decay_v1",
        max_generations=17,  # a couple more generations past the checkpoint
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=15,
        run_id="sentinel-test",
    )
    result = CMAESOptimizer(config=resume_config).fit(
        exponential_decay,
        x,
        y,
        bounds=bounds,
        refine_with_nlsq=False,
    )

    assert result["cmaes_diagnostics"]["best_fitness"] == sentinel_fitness
