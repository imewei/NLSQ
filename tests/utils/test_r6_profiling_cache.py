"""R6-05/R6-07: profiling temp-dir leak and non-finite cache threshold."""

import glob
import os
import subprocess
import sys
import tempfile

import pytest


def test_profiling_failure_does_not_leak_default_dir(monkeypatch, tmp_path):
    import jax.profiler

    from nlsq.utils.profiling import profile_transfers_runtime

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("profiler unavailable")

        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, *a):  # pragma: no cover
            return False

    monkeypatch.setattr(jax.profiler, "trace", Boom)
    res, stats = profile_transfers_runtime(lambda: 7)
    assert res == 7
    assert stats["profiler_available"] is False
    assert glob.glob(os.path.join(str(tmp_path), "jax-profiling-*")) == []


@pytest.mark.serial
@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "abc"])
def test_bad_min_compile_time_warns_and_uses_default(bad, tmp_path):
    code = (
        "import jax, nlsq\n"
        "print(jax.config.jax_compilation_cache_dir)\n"
        "print(jax.config.jax_persistent_cache_min_compile_time_secs)\n"
    )
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.pop("NLSQ_DISABLE_PERSISTENT_CACHE", None)
    env.update(
        NLSQ_JAX_CACHE_DIR=str(tmp_path / "jc"),
        NLSQ_CACHE_MIN_COMPILE_TIME_SECS=bad,
        JAX_PLATFORMS="cpu",
        NLSQ_SKIP_GPU_CHECK="1",
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-1] == "1.0", out.stdout
    assert "Invalid NLSQ_CACHE_MIN_COMPILE_TIME_SECS" in out.stderr
    # The bad value must not leave the cache half-configured.
    assert "Failed to enable JAX persistent compilation cache" not in out.stderr
    assert str(tmp_path / "jc") in out.stdout
