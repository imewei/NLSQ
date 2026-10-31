"""The global memory limit (env var / set_memory_limits / memory_context)
must reach strategy selection, and an explicit per-call limit must win."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.config import (
    JAXConfig,
    MemoryConfig,
    configure_for_large_datasets,
    get_memory_config,
    memory_context,
    set_memory_limits,
)
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig

N, P = 1_000_000, 3
DETECTED_GB = 12.5  # pinned: live free memory drifts between two reads


@pytest.fixture(autouse=True)
def _restore_global_memory_config(monkeypatch):
    """Isolate each test from the process-wide JAXConfig singleton."""
    monkeypatch.setattr(
        MemoryBudget,
        "_detect_available_memory",
        staticmethod(lambda use_gpu=False: DETECTED_GB),
    )
    for name in ("NLSQ_MEMORY_LIMIT_GB", "NLSQ_SAFETY_FACTOR"):
        monkeypatch.delenv(name, raising=False)
    inst = JAXConfig()
    monkeypatch.setattr(inst, "_memory_config", None)
    monkeypatch.setattr(inst, "_large_dataset_config", inst._large_dataset_config)


def _available() -> float:
    return MemoryBudget.compute(n_points=N, n_params=P).available_gb


def test_set_memory_limits_reaches_the_budget():
    set_memory_limits(memory_limit_gb=0.5)
    assert _available() == 0.5


def test_set_memory_limits_none_clears_the_configured_limit():
    set_memory_limits(memory_limit_gb=0.5)
    set_memory_limits(memory_limit_gb=None)
    assert get_memory_config().memory_limit_gb is None
    assert _available() == DETECTED_GB


def test_per_call_limit_beats_the_global_limit():
    set_memory_limits(memory_limit_gb=0.5)
    budget = MemoryBudget.compute(n_points=N, n_params=P, memory_limit_gb=2.0)
    assert budget.available_gb == 2.0


def test_unconfigured_global_limit_still_uses_detected_memory():
    # The 8 GB MemoryConfig default is a placeholder, not a user choice.
    assert get_memory_config().memory_limit_gb is None
    assert _available() == DETECTED_GB


def test_explicit_limit_equal_to_the_old_default_is_honored():
    set_memory_limits(memory_limit_gb=8.0)
    assert _available() == 8.0


def test_memory_context_applies_then_restores():
    with memory_context(MemoryConfig(memory_limit_gb=3.0)):
        assert _available() == 3.0
    assert _available() == DETECTED_GB


def test_memory_context_restores_a_configured_limit():
    set_memory_limits(memory_limit_gb=0.5)
    with memory_context(MemoryConfig(memory_limit_gb=3.0)):
        assert _available() == 3.0
    assert _available() == 0.5


def test_memory_context_restores_after_an_exception():
    set_memory_limits(memory_limit_gb=0.5)
    with pytest.raises(RuntimeError), memory_context(MemoryConfig(memory_limit_gb=3.0)):
        raise RuntimeError("boom")
    assert _available() == 0.5


def test_nested_memory_contexts_restore_the_outer_limit():
    with memory_context(MemoryConfig(memory_limit_gb=3.0)):
        with memory_context(MemoryConfig(memory_limit_gb=1.0)):
            assert _available() == 1.0
        assert _available() == 3.0
    assert _available() == DETECTED_GB


def test_config_that_leaves_the_limit_unset_does_not_pin_the_default():
    with memory_context(MemoryConfig(progress_reporting=False)):
        assert _available() == DETECTED_GB


def test_configure_for_large_datasets_without_a_limit_does_not_pin_the_default():
    configure_for_large_datasets()
    assert _available() == DETECTED_GB


def test_configure_for_large_datasets_with_a_limit_applies_it():
    configure_for_large_datasets(memory_limit_gb=4.0)
    assert _available() == 4.0


def test_env_var_limit_reaches_the_budget(monkeypatch):
    monkeypatch.setenv("NLSQ_MEMORY_LIMIT_GB", "1.5")
    JAXConfig.get_memory_config()  # lazy init reads the environment
    assert _available() == 1.5


@pytest.mark.parametrize("bad", ["0", "nan", "5000"])
def test_out_of_range_env_limit_warns_and_is_ignored(monkeypatch, bad):
    monkeypatch.setenv("NLSQ_MEMORY_LIMIT_GB", bad)
    with pytest.warns(UserWarning, match="NLSQ_MEMORY_LIMIT_GB"):
        JAXConfig.get_memory_config()
    assert _available() == DETECTED_GB


def test_tiny_global_limit_changes_the_selected_strategy():
    selector = MemoryBudgetSelector()
    set_memory_limits(memory_limit_gb=0.1)
    strategy, _ = selector.select(n_points=10 * N, n_params=P)  # data alone ~0.15 GB
    assert strategy == "streaming"


def test_curve_fit_large_leaves_the_users_global_limit_alone(monkeypatch):
    set_memory_limits(memory_limit_gb=32.0)
    seen = []
    original = LargeDatasetFitter.fit

    def spy(self, *args, **kwargs):
        seen.append(get_memory_config().memory_limit_gb)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LargeDatasetFitter, "fit", spy)
    x = np.linspace(0, 5, 20_000)
    y = 2.0 * np.exp(-0.7 * x)
    curve_fit_large(
        lambda x, a, b: a * jnp.exp(-b * x),
        x,
        y,
        p0=[1.0, 1.0],
        auto_size_detection=False,
    )
    assert seen == [32.0]
    assert get_memory_config().memory_limit_gb == 32.0


# --- safety_factor -----------------------------------------------------------
# At N, P the data is ~0.015 GB and the peak ~0.144 GB. A 0.19 GB limit lands on
# "chunked" at the default 0.75 and on "standard" at 0.95, each by >10%.
LIMIT_GB = 0.19


def _threshold(**kwargs) -> float:
    return MemoryBudget.compute(n_points=N, n_params=P, **kwargs).threshold_gb


def test_global_safety_factor_reaches_the_budget():
    set_memory_limits(memory_limit_gb=16.0, safety_factor=0.5)
    assert _threshold() == pytest.approx(8.0)


def test_explicit_default_valued_safety_factor_is_honored():
    # 0.8 was MemoryConfig's old default; asking for it must not read as "unset".
    set_memory_limits(memory_limit_gb=16.0, safety_factor=0.8)
    assert _threshold() == pytest.approx(12.8)


def test_setting_only_the_limit_keeps_the_workflow_default_safety_factor():
    set_memory_limits(memory_limit_gb=16.0)
    assert _threshold() == pytest.approx(12.0)


def test_set_memory_limits_without_a_factor_keeps_the_current_one():
    set_memory_limits(memory_limit_gb=16.0, safety_factor=0.5)
    set_memory_limits(memory_limit_gb=8.0)
    assert _threshold() == pytest.approx(4.0)


def test_per_call_safety_factor_beats_the_global_one():
    set_memory_limits(memory_limit_gb=16.0, safety_factor=0.5)
    assert _threshold(safety_factor=0.25) == pytest.approx(4.0)


def test_memory_context_applies_then_restores_the_safety_factor():
    set_memory_limits(memory_limit_gb=16.0, safety_factor=0.5)
    with memory_context(MemoryConfig(memory_limit_gb=16.0, safety_factor=0.25)):
        assert _threshold() == pytest.approx(4.0)
    assert _threshold() == pytest.approx(8.0)


def test_global_safety_factor_changes_the_selected_strategy():
    selector = MemoryBudgetSelector()
    set_memory_limits(memory_limit_gb=LIMIT_GB)
    assert selector.select(n_points=N, n_params=P)[0] == "chunked"
    set_memory_limits(memory_limit_gb=LIMIT_GB, safety_factor=0.95)
    assert selector.select(n_points=N, n_params=P)[0] == "standard"


def test_constructor_safety_factor_beats_the_global_one():
    set_memory_limits(memory_limit_gb=LIMIT_GB, safety_factor=0.95)
    selector = MemoryBudgetSelector(safety_factor=0.75)
    assert selector.select(n_points=N, n_params=P)[0] == "chunked"


def test_chunked_config_does_not_discount_the_global_safety_factor_twice():
    set_memory_limits(memory_limit_gb=LIMIT_GB, safety_factor=0.5)
    strategy, config = MemoryBudgetSelector().select(n_points=N, n_params=P)
    assert strategy == "chunked"
    assert config.memory_limit_gb == pytest.approx(LIMIT_GB * 0.5)  # once
    assert config.safety_factor == 1.0


def test_env_var_safety_factor_reaches_the_budget(monkeypatch):
    monkeypatch.setenv("NLSQ_MEMORY_LIMIT_GB", "16")
    monkeypatch.setenv("NLSQ_SAFETY_FACTOR", "0.5")
    JAXConfig.get_memory_config()  # lazy init reads the environment
    assert _threshold() == pytest.approx(8.0)


def test_out_of_range_env_safety_factor_warns_and_is_ignored(monkeypatch):
    monkeypatch.setenv("NLSQ_SAFETY_FACTOR", "5.0")
    with pytest.warns(UserWarning, match="NLSQ_SAFETY_FACTOR"):
        JAXConfig.get_memory_config()
    assert get_memory_config().safety_factor is None
    assert _threshold() == pytest.approx(DETECTED_GB * 0.75)


def test_memory_config_safety_factor_is_unset_by_default_and_range_checked():
    assert MemoryConfig().safety_factor is None
    with pytest.raises(ValueError, match="safety_factor"):
        MemoryConfig(safety_factor=0.05)
    with pytest.raises(ValueError, match="safety_factor"):
        MemoryConfig(safety_factor=1.5)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_safety_factor_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError, match="safety_factor"):
        LDMemoryConfig(safety_factor=bad)
    with pytest.raises(ValueError, match="safety_factor"):
        MemoryBudget.compute(n_points=N, n_params=P, safety_factor=bad)
    with pytest.raises(ValueError, match="safety_factor"):
        MemoryBudgetSelector(safety_factor=bad)


def _swap_between_reads(monkeypatch, first, second):
    """get_memory_config returns `first` once, then `second` (a concurrent swap)."""
    configs = iter([first, second])
    monkeypatch.setattr(
        JAXConfig, "get_memory_config", classmethod(lambda cls: next(configs, second))
    )


def test_budget_resolves_limit_and_safety_factor_from_one_snapshot(monkeypatch):
    _swap_between_reads(
        monkeypatch,
        MemoryConfig(memory_limit_gb=10.0, safety_factor=0.1),
        MemoryConfig(memory_limit_gb=1.0, safety_factor=1.0),
    )
    budget = MemoryBudget.compute(n_points=N, n_params=P)
    assert (budget.available_gb, budget.threshold_gb) == (10.0, pytest.approx(1.0))


def test_selector_resolves_limit_and_safety_factor_from_one_snapshot(monkeypatch):
    # Consistent first config -> "chunked"; mixing in the second -> "standard".
    _swap_between_reads(
        monkeypatch,
        MemoryConfig(memory_limit_gb=LIMIT_GB, safety_factor=0.75),
        MemoryConfig(memory_limit_gb=100.0, safety_factor=1.0),
    )
    assert MemoryBudgetSelector().select(n_points=N, n_params=P)[0] == "chunked"
