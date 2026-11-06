"""Infeasible chunk budgets must fail before constructing zero-sized chunks."""

import pytest

from nlsq.caching.memory_manager import MemoryManager


def test_subrow_budget_reports_insufficient_memory():
    with pytest.raises(MemoryError, match="one data point"):
        MemoryManager().estimate_chunking_strategy(1000, 10, memory_limit_gb=1e-9)
