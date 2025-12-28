# nlsq/caching/__init__.py
"""Caching and memory management modules.

This subpackage contains caching and memory management:
- core: Basic caching utilities
- smart_cache: SmartCache with intelligent invalidation
- unified_cache: Unified cache management
- compilation_cache: JIT compilation caching
- memory_manager: Memory management and tracking
- memory_pool: Memory pooling for optimization
"""

from nlsq.caching.compilation_cache import (
    CompilationCache,
    cached_jit,
    clear_compilation_cache,
    get_global_compilation_cache,
)
from nlsq.caching.memory_manager import (
    MemoryManager,
    clear_memory_pool,
    get_memory_manager,
    get_memory_stats,
)
from nlsq.caching.memory_pool import (
    MemoryPool,
    TRFMemoryPool,
    clear_global_pool,
    get_global_pool,
)
from nlsq.caching.smart_cache import (
    SmartCache,
    cached_function,
    cached_jacobian,
    clear_all_caches,
    get_global_cache,
    get_jit_cache,
)

__all__ = [
    "CompilationCache",
    "MemoryManager",
    "MemoryPool",
    "SmartCache",
    "TRFMemoryPool",
    "cached_function",
    "cached_jacobian",
    "cached_jit",
    "clear_all_caches",
    "clear_compilation_cache",
    "clear_global_pool",
    "clear_memory_pool",
    "get_global_cache",
    "get_global_compilation_cache",
    "get_global_pool",
    "get_jit_cache",
    "get_memory_manager",
    "get_memory_stats",
]
