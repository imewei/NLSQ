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
