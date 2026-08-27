import signal
import subprocess
import sys
import textwrap
import time
from typing import Any

import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer


def _exponential_decay(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


def test_signal_handlers_restored_after_normal_fit(tmp_path):
    """fit()'s `with self._preemption_handling(...):` registers SIGTERM/
    SIGUSR1 handlers on entry and must restore whatever was previously
    installed on exit -- including the ordinary, non-preempted completion
    path, not just the CMAESPreempted path. A library silently leaving its
    own handler installed after returning would break the caller's own
    SIGTERM handling for the rest of the process lifetime; this only
    surfaces by comparing disposition before/after, never by output alone."""
    previous_term = signal.getsignal(signal.SIGTERM)
    # SIGUSR1 doesn't exist on Windows -- `_preemption_handling` disables
    # all SIGTERM/SIGUSR1 registration there (see its Windows guard), so
    # there's nothing to compare on that platform.
    previous_usr1: Any = (
        signal.getsignal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else None
    )

    x = np.linspace(0, 5, 50)
    y = np.asarray(_exponential_decay(x, 2.5, 0.5))
    config = CMAESConfig(
        max_generations=5,
        restart_strategy="none",
        seed=19,
        model_id="exponential_decay_v1",
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=5,
        run_id="handler-restore-test",
    )
    CMAESOptimizer(config=config).fit(
        _exponential_decay,
        x,
        y,
        bounds=([0.0, 0.0], [10.0, 2.0]),
        refine_with_nlsq=False,
    )

    assert signal.getsignal(signal.SIGTERM) == previous_term
    if hasattr(signal, "SIGUSR1"):
        assert signal.getsignal(signal.SIGUSR1) == previous_usr1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "SIGUSR1 doesn't exist on Windows, so _preemption_handling disables "
        "SIGTERM/SIGUSR1 registration entirely there (falls back to "
        "periodic interval saves) -- this test's premise, a catchable "
        "SIGTERM producing SystemExit(75), doesn't apply on that platform."
    ),
)
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
