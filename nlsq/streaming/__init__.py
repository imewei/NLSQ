# nlsq/streaming/__init__.py
"""Streaming and large dataset handling modules.

This subpackage contains modules for handling large datasets:
- optimizer: StreamingOptimizer for datasets that exceed memory
- config: StreamingConfig for streaming configuration
- adaptive_hybrid: AdaptiveHybridStreamingOptimizer with defense layers
- hybrid_config: HybridStreamingConfig for hybrid streaming
- large_dataset: LargeDatasetFitter for automatic chunking
"""

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import (
    LargeDatasetFitter,
    LDMemoryConfig,
    estimate_memory_requirements,
    fit_large_dataset,
)
from nlsq.streaming.optimizer import (
    DataGenerator,
    StreamingOptimizer,
    create_hdf5_dataset,
    fit_unlimited_data,
)

__all__ = [
    "AdaptiveHybridStreamingOptimizer",
    "DataGenerator",
    "HybridStreamingConfig",
    "LDMemoryConfig",
    "LargeDatasetFitter",
    "StreamingConfig",
    "StreamingOptimizer",
    "create_hdf5_dataset",
    "estimate_memory_requirements",
    "fit_large_dataset",
    "fit_unlimited_data",
]
