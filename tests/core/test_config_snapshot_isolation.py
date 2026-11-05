"""Configuration snapshots and caller-owned inputs must not alias globals."""

import pytest

from nlsq.config import JAXConfig, LargeDatasetConfig, MemoryConfig

pytestmark = pytest.mark.serial


def test_large_config_snapshot_owns_its_thresholds():
    with JAXConfig.large_dataset_context(LargeDatasetConfig()):
        snapshot = JAXConfig.get_large_dataset_config()
        expected = snapshot.solver_selection_thresholds["direct"]
        snapshot.solver_selection_thresholds["direct"] = -1
        assert (
            JAXConfig.get_large_dataset_config().solver_selection_thresholds["direct"]
            == expected
        )


def test_large_config_setter_owns_its_input():
    config = LargeDatasetConfig()
    with JAXConfig.large_dataset_context(config):
        expected = config.solver_selection_thresholds["direct"]
        config.solver_selection_thresholds["direct"] = -1
        assert (
            JAXConfig.get_large_dataset_config().solver_selection_thresholds["direct"]
            == expected
        )


def test_memory_config_setter_owns_its_input():
    config = MemoryConfig(memory_limit_gb=2.0)
    with JAXConfig.memory_context(config):
        config.memory_limit_gb = -1.0
        assert JAXConfig.get_memory_config().memory_limit_gb == 2.0
