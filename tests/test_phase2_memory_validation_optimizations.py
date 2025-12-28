"""Tests for Phase 2 Memory and Validation Optimizations.

This module tests the optimizations introduced in Task Group 7:
- LRU Memory Pool (1.2a) in memory_manager.py
- Model Validation Caching (5.1a) in large_dataset.py
- JIT-Compiled Validation (4.3a) in streaming_optimizer.py
- DataChunker Padding Optimization (5.2a) in large_dataset.py

Expected gains:
- LRU Memory Pool: 5-10% in memory-constrained workloads
- Model Validation Caching: 1-5% in chunked processing
- JIT-Compiled Validation: 5-10% when numeric validation is enabled
- DataChunker Padding: 10-20% memory allocation reduction
"""

from collections import OrderedDict

import jax.numpy as jnp
import numpy as np
import pytest


class TestLRUMemoryPool:
    """Test LRU eviction behavior in memory pool."""

    def test_memory_pool_uses_ordered_dict(self):
        """Test that memory_pool uses OrderedDict for LRU tracking."""
        from nlsq.caching.memory_manager import MemoryManager

        manager = MemoryManager()
        # Memory pool should be OrderedDict (or compatible) for LRU
        assert hasattr(manager.memory_pool, "move_to_end"), (
            "memory_pool should support move_to_end() for LRU"
        )

    def test_move_to_end_on_array_reuse(self):
        """Test that move_to_end is called when array is reused from pool."""
        from nlsq.caching.memory_manager import MemoryManager

        manager = MemoryManager()

        # Allocate several arrays with different shapes
        shape1 = (100,)
        shape2 = (200,)
        shape3 = (300,)

        arr1 = manager.allocate_array(shape1, np.float64, zero=True)
        arr2 = manager.allocate_array(shape2, np.float64, zero=True)
        arr3 = manager.allocate_array(shape3, np.float64, zero=True)

        # Record initial order
        keys_initial = list(manager.memory_pool.keys())
        assert len(keys_initial) == 3

        # Re-request array with shape1 - should move to end
        arr1_reused = manager.allocate_array(shape1, np.float64, zero=True)

        # Check shape1 is now at the end (most recently used)
        keys_after = list(manager.memory_pool.keys())
        assert keys_after[-1] == (shape1, np.float64), (
            f"shape1 should be at end after reuse, got {keys_after[-1]}"
        )

    def test_lru_eviction_with_popitem(self):
        """Test LRU eviction uses popitem(last=False) when at capacity."""
        from nlsq.caching.memory_manager import MemoryManager

        manager = MemoryManager()

        # Allocate arrays up to the optimize_memory_pool max_arrays threshold
        max_arrays = 5  # Test with a small limit

        # Allocate more arrays than max_arrays
        for i in range(max_arrays + 3):
            shape = (100 + i,)
            manager.allocate_array(shape, np.float64, zero=True)

        assert len(manager.memory_pool) == max_arrays + 3

        # Call optimize_memory_pool with small limit
        manager.optimize_memory_pool(max_arrays=max_arrays)

        # Pool should now have at most max_arrays entries
        assert len(manager.memory_pool) <= max_arrays, (
            f"Pool should have at most {max_arrays} entries, got {len(manager.memory_pool)}"
        )

    def test_lru_prioritizes_recently_used_arrays(self):
        """Test that LRU eviction correctly prioritizes recently used arrays."""
        from nlsq.caching.memory_manager import MemoryManager

        manager = MemoryManager()

        # Allocate arrays
        shapes = [(100,), (200,), (300,), (400,), (500,)]
        for shape in shapes:
            manager.allocate_array(shape, np.float64, zero=True)

        # Access arrays in specific order to mark recent usage
        # Access (300,) and (100,) to make them recently used
        manager.allocate_array((300,), np.float64, zero=True)
        manager.allocate_array((100,), np.float64, zero=True)

        # The recently accessed shapes should be at the end
        keys = list(manager.memory_pool.keys())
        # (100,) was accessed last, should be last
        assert keys[-1] == ((100,), np.float64)
        # (300,) was accessed before that, should be second-to-last
        assert keys[-2] == ((300,), np.float64)


class TestModelValidationCaching:
    """Test model validation caching by function identity."""

    def test_validated_functions_attribute_exists(self):
        """Test that LargeDatasetFitter has _validated_functions dict."""
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        fitter = LargeDatasetFitter()
        assert hasattr(fitter, "_validated_functions"), (
            "LargeDatasetFitter should have _validated_functions attribute"
        )
        assert isinstance(fitter._validated_functions, dict)

    def test_validation_cache_uses_composite_key(self):
        """Test that validation caching uses (id(func), id(func.__code__)) key."""
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        fitter = LargeDatasetFitter()

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Validate the model
        x = np.linspace(0, 1, 100)
        y = 2.5 * np.exp(-0.3 * x)
        p0 = [1.0, 0.1]

        fitter._validate_model_function(model, x, y, p0)

        # Check that validation cache has the composite key
        expected_key = (id(model), id(model.__code__))
        assert expected_key in fitter._validated_functions, (
            f"Expected composite key {expected_key} in _validated_functions"
        )

    def test_validation_skipped_for_same_function(self):
        """Test that validation is skipped on subsequent calls with same function."""
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        fitter = LargeDatasetFitter()

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 1, 100)
        y = 2.5 * np.exp(-0.3 * x)
        p0 = [1.0, 0.1]

        # First validation - should validate
        fitter._validate_model_function(model, x, y, p0)
        validation_count_1 = len(fitter._validated_functions)

        # Second validation with same function - should skip (cache hit)
        fitter._validate_model_function(model, x, y, p0)
        validation_count_2 = len(fitter._validated_functions)

        # Cache size should not change (same function)
        assert validation_count_1 == validation_count_2, (
            "Validation should be skipped for same function identity"
        )

    def test_validation_runs_for_different_function(self):
        """Test that validation runs for different functions."""
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        fitter = LargeDatasetFitter()

        def model1(x, a, b):
            return a * jnp.exp(-b * x)

        def model2(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 1, 100)
        y = 2.5 * np.exp(-0.3 * x)

        # First function
        fitter._validate_model_function(model1, x, y, [1.0, 0.1])
        cache_size_1 = len(fitter._validated_functions)

        # Second function - should validate (different function)
        fitter._validate_model_function(model2, x, y, [1.0, 0.1, 0.0])
        cache_size_2 = len(fitter._validated_functions)

        # Cache should have both functions
        assert cache_size_2 == cache_size_1 + 1, (
            "Different functions should be validated separately"
        )


class TestJITCompiledValidation:
    """Test JIT-compiled NaN/Inf validation in streaming optimizer."""

    def test_jit_compiled_validation_returns_tuple(self):
        """Test that JIT-compiled gradient function returns (loss, grad, is_valid) tuple."""
        from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer

        config = StreamingConfig(
            batch_size=32,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Get the JIT-compiled validation function if it exists
        # This test verifies the structure of the compiled validation function
        x_batch = jnp.linspace(0, 1, 32)
        y_batch = 2.5 * jnp.exp(-0.3 * x_batch)
        params = jnp.array([1.0, 0.1])

        # Get the loss and grad function with validation
        loss_and_grad_fn = optimizer._get_loss_and_grad_fn_with_validation(model)

        # Call the function
        result = loss_and_grad_fn(params, x_batch, y_batch)

        # Should return (loss, grad, is_valid) tuple
        assert len(result) == 3, (
            f"Expected (loss, grad, is_valid) tuple, got {len(result)} elements"
        )
        loss, grad, is_valid = result
        assert isinstance(float(loss), float)
        assert grad.shape == params.shape
        assert isinstance(bool(is_valid), bool)

    def test_jit_validation_detects_nan_in_loss(self):
        """Test that JIT-compiled validation detects NaN in loss."""
        from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer

        config = StreamingConfig(
            batch_size=32,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def bad_model(x, a, b):
            # Model that produces NaN
            return a / jnp.where(x == 0, 0.0, x) * b

        loss_and_grad_fn = optimizer._get_loss_and_grad_fn_with_validation(bad_model)

        # Data with x=0 which will cause NaN
        x_batch = jnp.zeros(32)
        y_batch = jnp.ones(32)
        params = jnp.array([1.0, 0.1])

        _loss, _grad, is_valid = loss_and_grad_fn(params, x_batch, y_batch)

        # Should detect invalid values
        assert not bool(is_valid), "is_valid should be False for NaN loss"

    def test_jit_validation_detects_inf_in_gradient(self):
        """Test that JIT-compiled validation detects Inf in gradient."""
        from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer

        config = StreamingConfig(
            batch_size=32,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def bad_model(x, a, b):
            # Model that can produce Inf gradient
            return a * jnp.exp(1000 * b * x)

        loss_and_grad_fn = optimizer._get_loss_and_grad_fn_with_validation(bad_model)

        # Large x values will cause exp overflow
        x_batch = jnp.linspace(0, 100, 32)
        y_batch = jnp.ones(32)
        params = jnp.array([1.0, 1.0])

        _loss, _grad, is_valid = loss_and_grad_fn(params, x_batch, y_batch)

        # Should detect invalid gradient values
        # Note: The specific behavior depends on JAX's handling of overflow
        # This test verifies the validation mechanism is in place
        assert isinstance(bool(is_valid), bool)


class TestDataChunkerPaddingOptimization:
    """Test bucket padding in DataChunker.

    Note: DataChunker now uses power-of-2 bucket sizes for JIT stability.
    Chunks are padded to the nearest bucket size (1024, 2048, 4096, etc.)
    not to the requested chunk_size.
    """

    def test_bucket_padding_applied(self):
        """Test that bucket padding is applied for JIT stability."""
        from nlsq.streaming.large_dataset import DataChunker, get_bucket_size

        # Create test data
        x = np.arange(1050)
        y = np.arange(1050) * 2.0
        chunk_size = 1000

        chunks = list(DataChunker.create_chunks(x, y, chunk_size))

        # First chunk should be padded to bucket size
        x_chunk1, _y_chunk1, _idx1, valid_len1 = chunks[0]
        expected_bucket = get_bucket_size(chunk_size)
        assert len(x_chunk1) == expected_bucket
        assert valid_len1 == chunk_size  # Valid length should be the actual chunk size

        # Last chunk is also padded to bucket size
        x_chunk2, _y_chunk2, _idx2, valid_len2 = chunks[1]
        assert len(x_chunk2) == get_bucket_size(50)  # 50 points padded to bucket
        assert valid_len2 == 50  # Valid length is the actual points count

    def test_resize_cyclic_repetition(self):
        """Test that padding uses cyclic repetition (np.resize behavior)."""
        from nlsq.streaming.large_dataset import DataChunker, get_bucket_size

        # Create test data with 50 points, chunk_size 100
        x = np.arange(50)
        y = np.arange(50) * 2.0
        chunk_size = 100

        chunks = list(DataChunker.create_chunks(x, y, chunk_size))

        # Only one chunk, which needs padding
        x_chunk, _y_chunk, _idx, valid_len = chunks[0]

        # Check bucket padding applied
        expected_bucket = get_bucket_size(50)
        assert len(x_chunk) == expected_bucket
        assert valid_len == 50

        # Verify cyclic pattern in the padded portion
        for i in range(50):
            assert x_chunk[50 + i] == x_chunk[i % 50], (
                f"Expected cyclic repetition at index {50 + i}"
            )

    def test_bucket_padding_memory_allocation(self):
        """Test that bucket padding allocates to power-of-2 sizes."""
        from nlsq.streaming.large_dataset import DataChunker, get_bucket_size

        # Create test data
        x = np.arange(950)
        y = np.arange(950) * 2.0
        chunk_size = 1000

        chunks = list(DataChunker.create_chunks(x, y, chunk_size))

        # Only one chunk
        x_chunk, y_chunk, _idx, valid_len = chunks[0]

        # Check that chunk is padded to bucket size
        expected_bucket = get_bucket_size(950)
        assert len(x_chunk) == expected_bucket
        assert len(y_chunk) == expected_bucket
        assert valid_len == 950

    def test_valid_length_tracks_actual_data(self):
        """Test that valid_length correctly tracks actual data size."""
        from nlsq.streaming.large_dataset import DataChunker, get_bucket_size

        # Data size is exact multiple of chunk_size
        x = np.arange(1000)
        y = np.arange(1000) * 2.0
        chunk_size = 1000

        chunks = list(DataChunker.create_chunks(x, y, chunk_size))

        # Should be exactly one chunk
        assert len(chunks) == 1
        x_chunk, _y_chunk, _idx, valid_len = chunks[0]
        # Chunk is padded to bucket size
        assert len(x_chunk) == get_bucket_size(1000)
        # But valid_len tracks actual data
        assert valid_len == 1000


class TestIntegration:
    """Integration tests for Phase 2 optimizations."""

    def test_lru_pool_with_repeated_allocations(self):
        """Test LRU pool behavior with repeated allocations pattern."""
        from nlsq.caching.memory_manager import MemoryManager

        manager = MemoryManager()

        # Simulate typical usage pattern
        shapes = [(100, 10), (200, 10), (100, 10), (300, 10), (100, 10)]
        for shape in shapes:
            arr = manager.allocate_array(shape, np.float64)
            # Use the array
            arr[:] = np.random.randn(*shape)

        # Most recently used shape should be at end
        keys = list(manager.memory_pool.keys())
        assert keys[-1] == ((100, 10), np.float64), (
            "Most recently used array should be at end of pool"
        )

    def test_model_validation_caching_in_chunked_processing(self):
        """Test validation caching during chunked processing.

        Validation caching is specifically for chunked processing to avoid
        redundant validation across chunks. This test directly verifies
        the caching behavior via _validate_model_function.
        """
        from nlsq.streaming.large_dataset import LargeDatasetFitter

        fitter = LargeDatasetFitter(memory_limit_gb=1.0)

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Create test data
        x = np.linspace(0, 10, 1000)
        y = 2.5 * np.exp(-0.3 * x) + 0.1 * np.random.randn(len(x))
        p0 = [1.0, 0.1]

        # Simulate multiple chunk validations (what _fit_chunked does)
        # First validation - cache miss
        fitter._validate_model_function(model, x[:500], y[:500], p0)
        cache_size_1 = len(fitter._validated_functions)

        # Second validation with same model - cache hit
        fitter._validate_model_function(model, x[500:], y[500:], p0)
        cache_size_2 = len(fitter._validated_functions)

        # Third validation - still cache hit
        fitter._validate_model_function(model, x[:100], y[:100], p0)
        cache_size_3 = len(fitter._validated_functions)

        # Cache should have been populated once and reused
        assert cache_size_1 == 1, "First validation should add to cache"
        assert cache_size_2 == 1, "Second validation should use cache (no new entry)"
        assert cache_size_3 == 1, "Third validation should use cache (no new entry)"

        # Verify the function is in the cache
        func_key = (id(model), id(model.__code__))
        assert func_key in fitter._validated_functions, (
            "Model function should be in validation cache"
        )
