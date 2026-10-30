"""The global memory limit (env var / set_memory_limits / memory_context)
must reach strategy selection, and an explicit per-call limit must win."""

import pytest

from nlsq.config import JAXConfig, MemoryConfig, memory_context, set_memory_limits
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

N, P = 1_000_000, 3


@pytest.fixture(autouse=True)
def _restore_global_memory_config():
    """Isolate each test from the process-wide JAXConfig singleton."""
    inst = JAXConfig()
    saved = (inst._memory_config, getattr(inst, "_memory_limit_override", None))
    inst._memory_config = None
    inst._memory_limit_override = None
    yield
    inst._memory_config, inst._memory_limit_override = saved


def _detected() -> float:
    return MemoryBudget._detect_available_memory()


def test_set_memory_limits_reaches_the_budget():
    set_memory_limits(memory_limit_gb=0.5)
    assert MemoryBudget.compute(n_points=N, n_params=P).available_gb == 0.5


def test_per_call_limit_beats_the_global_limit():
    set_memory_limits(memory_limit_gb=0.5)
    budget = MemoryBudget.compute(n_points=N, n_params=P, memory_limit_gb=2.0)
    assert budget.available_gb == 2.0


def test_unconfigured_global_limit_still_uses_detected_memory():
    # The 8 GB MemoryConfig default is a placeholder, not a user choice.
    assert MemoryBudget.compute(n_points=N, n_params=P).available_gb == _detected()


def test_memory_context_applies_then_restores():
    with memory_context(MemoryConfig(memory_limit_gb=3.0)):
        assert MemoryBudget.compute(n_points=N, n_params=P).available_gb == 3.0
    assert MemoryBudget.compute(n_points=N, n_params=P).available_gb == _detected()


def test_env_var_limit_reaches_the_budget(monkeypatch):
    monkeypatch.setenv("NLSQ_MEMORY_LIMIT_GB", "1.5")
    JAXConfig.get_memory_config()  # lazy init reads the environment
    assert MemoryBudget.compute(n_points=N, n_params=P).available_gb == 1.5


def test_tiny_global_limit_changes_the_selected_strategy():
    selector = MemoryBudgetSelector()
    set_memory_limits(memory_limit_gb=0.1)
    strategy, _ = selector.select(n_points=10 * N, n_params=P)  # data alone ~0.15 GB
    assert strategy == "streaming"
