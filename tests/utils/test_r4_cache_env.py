"""R4-2: NLSQ_CACHE_MIN_COMPILE_TIME_SECS must apply when JAX_COMPILATION_CACHE_DIR is set."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.serial

_CODE = (
    "import nlsq, jax; print(jax.config.jax_compilation_cache_dir, "
    "jax.config.jax_persistent_cache_min_compile_time_secs)"
)


def _run(env_extra):
    env = {**os.environ, "NLSQ_SKIP_GPU_CHECK": "1", **env_extra}
    return subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_min_compile_time_applied_with_jax_cache_dir_env(tmp_path):
    r = _run(
        {
            "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jc"),
            "NLSQ_CACHE_MIN_COMPILE_TIME_SECS": "0",
        }
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(tmp_path / "jc"), "0.0"]


def test_jax_cache_dir_env_without_nlsq_override_keeps_jax_default(tmp_path):
    env = {
        k: v for k, v in os.environ.items() if k != "NLSQ_CACHE_MIN_COMPILE_TIME_SECS"
    }
    env.update(NLSQ_SKIP_GPU_CHECK="1", JAX_COMPILATION_CACHE_DIR=str(tmp_path / "jc"))
    r = subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(tmp_path / "jc"), "1.0"]
