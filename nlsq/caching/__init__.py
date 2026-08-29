# nlsq/caching/__init__.py
"""Caching and memory management modules.

This subpackage contains caching and memory management:
- unified_cache: Unified JIT compilation cache
- memory_manager: Memory management and tracking
"""

from nlsq.caching.memory_manager import (
    MemoryManager,
    clear_memory_pool,
    get_memory_manager,
    get_memory_stats,
)
from nlsq.caching.unified_cache import (
    UnifiedCache,
    cached_jit,
    clear_cache,
    get_cache_stats,
    get_global_cache,
)

__all__ = [
    "MemoryManager",
    "UnifiedCache",
    "cached_jit",
    "clear_cache",
    "clear_memory_pool",
    "get_cache_stats",
    "get_global_cache",
    "get_memory_manager",
    "get_memory_stats",
]
