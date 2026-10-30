"""Default logging: a plain import + fit must not print NLSQ log records.

Each test runs a fresh interpreter, because logging state (root handlers,
logger levels) is process-global and set at import time.
"""

import os
import re
import subprocess
import sys

import pytest

_FIT_SCRIPT = """
import jax.numpy as jnp
import numpy as np
import nlsq

x = np.linspace(0.0, 4.0, 200)
y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.05 * np.random.default_rng(0).standard_normal(200)
popt, _ = nlsq.curve_fit(lambda x, a, b, c: a * jnp.exp(-b * x) + c, x, y, p0=[1.0, 1.0, 0.0])
print("POPT", *popt)
"""

_PREFIXES = ("INFO:", "PERFORMANCE:", "DEBUG:")
# Catches other formatters too, e.g. "[INFO] nlsq.trf - ..." or "12:00 [DEBUG] nlsq.x:1"
_NLSQ_RECORD = re.compile(
    r"\b(INFO|PERFORMANCE|DEBUG)\b.*\bnlsq\.|\bnlsq\..*\b(INFO|PERFORMANCE|DEBUG)\b"
)


def _run_fit(tmp_path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in ("NLSQ_DEBUG", "NLSQ_VERBOSE")
    }
    env.update(NLSQ_SKIP_GPU_CHECK="1", NLSQ_LOG_DIR=str(tmp_path), **extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _FIT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "POPT" in proc.stdout
    return proc


def _log_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    lines = (proc.stdout + proc.stderr).splitlines()
    return [ln for ln in lines if ln.startswith(_PREFIXES) or _NLSQ_RECORD.search(ln)]


@pytest.mark.serial
def test_default_fit_writes_no_log_lines(tmp_path):
    proc = _run_fit(tmp_path)
    assert _log_lines(proc) == []


@pytest.mark.serial
def test_nlsq_debug_enables_log_lines(tmp_path):
    proc = _run_fit(tmp_path, NLSQ_DEBUG="1")
    assert _log_lines(proc), proc.stdout + proc.stderr
