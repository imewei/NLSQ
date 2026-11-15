"""Removal of the top-level profiler exports and of nlsq.facades."""

import importlib
import importlib.util
import subprocess
import sys

import pytest

import nlsq

REMOVED = [
    "PerformanceProfiler",
    "ProfileMetrics",
    "clear_profiling_data",
    "get_global_profiler",
    "ProfilerVisualization",
    "ProfilingDashboard",
]

DEFINING_MODULE = {
    "PerformanceProfiler": "nlsq.utils.profiler",
    "ProfileMetrics": "nlsq.utils.profiler",
    "clear_profiling_data": "nlsq.utils.profiler",
    "get_global_profiler": "nlsq.utils.profiler",
    "ProfilerVisualization": "nlsq.utils.profiler_visualization",
    "ProfilingDashboard": "nlsq.utils.profiler_visualization",
}


@pytest.mark.parametrize("name", REMOVED)
def test_removed_names_are_gone_from_top_level(name):
    with pytest.raises(AttributeError):
        getattr(nlsq, name)
    assert name not in dir(nlsq)
    with pytest.raises(ImportError):
        exec(f"from nlsq import {name}")


@pytest.mark.parametrize("name", REMOVED)
def test_names_still_importable_from_defining_module(name):
    assert hasattr(importlib.import_module(DEFINING_MODULE[name]), name)


@pytest.mark.serial
def test_import_nlsq_emits_no_warning():
    r = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import nlsq"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_facades_package_is_gone():
    assert importlib.util.find_spec("nlsq.facades") is None
