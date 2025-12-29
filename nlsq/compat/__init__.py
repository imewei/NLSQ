# nlsq/compat/__init__.py
"""Backwards compatibility shims for deprecated import paths.

This module provides backwards compatibility for the old flat import structure.
Importing from old paths (e.g., `from nlsq.streaming.optimizer import X`) will
emit a DeprecationWarning while still working correctly.

The compatibility shims use module-level __getattr__ to lazily redirect imports
to their new locations in the reorganized subpackage structure.
"""

import warnings
from typing import Any

# Mapping of old module names to their new locations
_COMPAT_MAPPING: dict[str, str] = {
    # Streaming modules
    "streaming_optimizer": "nlsq.streaming.optimizer",
    "streaming_config": "nlsq.streaming.config",
    "adaptive_hybrid_streaming": "nlsq.streaming.adaptive_hybrid",
    "hybrid_streaming_config": "nlsq.streaming.hybrid_config",
    "large_dataset": "nlsq.streaming.large_dataset",
    # Core modules
    "minpack": "nlsq.core.minpack",
    "least_squares": "nlsq.core.least_squares",
    "trf": "nlsq.core.trf",
    "_optimize": "nlsq.core._optimize",
    "optimizer_base": "nlsq.core.optimizer_base",
    "sparse_jacobian": "nlsq.core.sparse_jacobian",
    "functions": "nlsq.core.functions",
    "loss_functions": "nlsq.core.loss_functions",
    "workflow": "nlsq.core.workflow",
    # Caching modules
    "caching": "nlsq.caching.core",
    "smart_cache": "nlsq.caching.smart_cache",
    "unified_cache": "nlsq.caching.unified_cache",
    "compilation_cache": "nlsq.caching.compilation_cache",
    "memory_manager": "nlsq.caching.memory_manager",
    "memory_pool": "nlsq.caching.memory_pool",
    # Stability modules
    "stability": "nlsq.stability.guard",
    "svd_fallback": "nlsq.stability.svd_fallback",
    "robust_decomposition": "nlsq.stability.robust_decomposition",
    "recovery": "nlsq.stability.recovery",
    "fallback": "nlsq.stability.fallback",
    # Utils modules
    "diagnostics": "nlsq.utils.diagnostics",
    "profiler": "nlsq.utils.profiler",
    "profiler_visualization": "nlsq.utils.profiler_visualization",
    "profiling": "nlsq.utils.profiling",
    "logging": "nlsq.utils.logging",
    "async_logger": "nlsq.utils.async_logger",
    "validators": "nlsq.utils.validators",
    "error_messages": "nlsq.utils.error_messages",
    # Precision modules
    "mixed_precision": "nlsq.precision.mixed_precision",
    "parameter_normalizer": "nlsq.precision.parameter_normalizer",
    "parameter_estimation": "nlsq.precision.parameter_estimation",
    "bound_inference": "nlsq.precision.bound_inference",
    "algorithm_selector": "nlsq.precision.algorithm_selector",
}


def get_deprecated_module(old_name: str) -> Any:
    """Get a module from its deprecated path, emitting a warning."""
    if old_name not in _COMPAT_MAPPING:
        raise ImportError(f"No compatibility mapping for '{old_name}'")

    new_path = _COMPAT_MAPPING[old_name]
    warnings.warn(
        f"Importing from 'nlsq.{old_name}' is deprecated. Use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    import importlib

    return importlib.import_module(new_path)


__all__ = ["_COMPAT_MAPPING", "get_deprecated_module"]
