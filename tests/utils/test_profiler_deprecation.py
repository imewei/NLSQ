"""Deprecation of the top-level profiler exports and removal of nlsq.facades."""

import importlib
import importlib.util
import subprocess
import sys
import warnings

import pytest

import nlsq

DEPRECATED = [
    "PerformanceProfiler",
    "ProfileMetrics",
    "clear_profiling_data",
    "get_global_profiler",
    "ProfilerVisualization",
    "ProfilingDashboard",
]


@pytest.fixture(autouse=True)
def _reset_seen():
    nlsq._deprecation_seen.clear()
    yield
    nlsq._deprecation_seen.clear()


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecated_names_warn_once(name):
    with pytest.warns(DeprecationWarning, match="next minor release"):
        first = getattr(nlsq, name)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        second = getattr(nlsq, name)
    assert not [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert first is second


@pytest.mark.serial
def test_import_nlsq_emits_no_deprecation_warning():
    r = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import nlsq"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_facades_package_is_gone():
    assert importlib.util.find_spec("nlsq.facades") is None
