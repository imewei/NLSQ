#!/usr/bin/env python
"""
Test script to verify all README.md examples work with current codebase.
"""

import sys
import traceback

import jax.numpy as jnp
import numpy as np
import pytest

# Check if h5py is available for streaming tests
try:
    pass

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def test_basic_usage():
    """Test Basic Usage example from README."""
    print("Testing Basic Usage example...")
    from nlsq import CurveFit

    # Define your fit function
    def linear(x, m, b):
        return m * x + b

    # Prepare data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    # Perform the fit
    cf = CurveFit()
    popt, _pcov = cf.curve_fit(linear, x, y)
    print(f"Fitted parameters: m={popt[0]:.2f}, b={popt[1]:.2f}")
    assert np.allclose(popt[0], 2.0, atol=0.01)
    print("✅ Basic Usage example passed")


def test_exponential_with_jax():
    """Test exponential example with JAX numpy."""
    print("\nTesting exponential with JAX numpy example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import CurveFit

    # Define exponential fit function using JAX numpy
    def exponential(x, a, b):
        return jnp.exp(a * x) + b

    # Generate synthetic data
    x = np.linspace(0, 4, 50)
    y_true = np.exp(0.5 * x) + 2.0
    y = y_true + 0.1 * np.random.normal(size=len(x))

    # Fit with initial guess
    cf = CurveFit()
    popt, pcov = cf.curve_fit(exponential, x, y, p0=[0.5, 2.0])
    print(f"Fitted: a={popt[0]:.3f}, b={popt[1]:.3f}")

    # Get parameter uncertainties from covariance
    perr = np.sqrt(np.diag(pcov))
    print(f"Uncertainties: σ_a={perr[0]:.3f}, σ_b={perr[1]:.3f}")
    print("✅ Exponential with JAX numpy example passed")


def test_curve_fit_large():
    """Test curve_fit_large example."""
    print("\nTesting curve_fit_large example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import curve_fit_large, estimate_memory_requirements

    # Check memory requirements for your dataset
    n_points = 50_000  # Reduced from 50M for faster testing
    n_params = 3
    stats = estimate_memory_requirements(n_points, n_params)
    print(f"Memory required: {stats.total_memory_estimate_gb:.2f} GB")
    print(f"Recommended chunks: {stats.n_chunks}")

    # Generate large dataset
    x = np.linspace(0, 10, n_points)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n_points)

    # Define fit function using JAX numpy
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Use curve_fit_large for automatic dataset size detection and chunking
    popt, pcov = curve_fit_large(
        exponential,
        x,
        y,
        p0=[2.5, 0.6, 0.2],
        memory_limit_gb=4.0,  # Automatic chunking if needed
        show_progress=False,  # Disabled for testing
    )

    print(f"Fitted parameters: {popt}")
    print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")
    print("✅ curve_fit_large example passed")


def test_advanced_large_dataset():
    """Test advanced large dataset fitting options."""
    print("\nTesting advanced large dataset fitting...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import LargeDatasetFitter, LDMemoryConfig, fit_large_dataset

    n_points = 50_000
    x = np.linspace(0, 10, n_points)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n_points)

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Option 1: Use the convenience function for simple cases
    result = fit_large_dataset(
        exponential,
        x,
        y,
        p0=[2.5, 0.6, 0.2],
        memory_limit_gb=4.0,
        show_progress=False,  # Disabled for testing
    )

    # Option 2: Use LargeDatasetFitter for more control (v0.2.0: sampling removed)
    config = LDMemoryConfig(
        memory_limit_gb=4.0,
        min_chunk_size=10000,
        max_chunk_size=1000000,
    )

    fitter = LargeDatasetFitter(config=config)
    result = fitter.fit_with_progress(
        exponential,
        x,
        y,
        p0=[2.5, 0.6, 0.2],
    )

    print(f"Fitted parameters: {result.popt}")
    # Note: success_rate and n_chunks are only available for multi-chunk fits
    # print(f"Covariance matrix: {result.pcov}")
    print("✅ Advanced large dataset fitting example passed")


def test_sparse_jacobian():
    """Test Sparse Jacobian example."""
    print("\nTesting Sparse Jacobian example...")
    from nlsq import SparseJacobianComputer

    # Create sample data
    n_points = 100
    x_sample = np.linspace(0, 10, n_points)

    def func(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    p0 = np.array([2.0, 0.5, 0.3])

    # Automatically detect and exploit sparsity
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)
    _pattern, sparsity = sparse_computer.detect_sparsity_pattern(func, p0, x_sample)

    if sparsity > 0.1:  # If more than 10% sparse
        print(f"Jacobian is {sparsity:.1%} sparse")
    else:
        print("Jacobian is not sparse for this problem")
    print("✅ Sparse Jacobian example passed")


@pytest.mark.skipif(
    not HAS_H5PY,
    reason="Streaming optimizer requires h5py (install with: pip install nlsq[streaming])",
)
def test_streaming_optimizer():
    """Test Streaming Optimizer example."""
    print("\nTesting Streaming Optimizer example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import StreamingConfig, StreamingOptimizer

    def func(x, a, b):
        return a * jnp.exp(-b * x)

    # Configure streaming optimization
    config = StreamingConfig(batch_size=1000, max_epochs=10, convergence_tol=1e-6)

    optimizer = StreamingOptimizer(config)

    # Create a simple data generator
    def data_generator():
        """Generate batches of data."""
        for _ in range(5):  # Limited batches for testing
            batch_size = 1000
            x_batch = np.random.randn(batch_size) * 10
            y_batch = 2.0 * np.exp(-0.5 * np.abs(x_batch))
            yield x_batch, y_batch

    p0 = np.array([2.0, 0.5])

    # Stream data from generator
    result = optimizer.fit_streaming(data_generator(), func, p0=p0)
    print(f"Streaming optimization result: {result}")
    print("✅ Streaming Optimizer example passed")


def test_memory_management():
    """Test Memory Management example."""
    print("\nTesting Memory Management example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import CurveFit, MemoryConfig, get_memory_config, memory_context

    # Sample data
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100) * 0.1
    p0 = [1.0, 0.0]

    def func(x, a, b):
        return a * x + b

    # Configure memory settings
    config = MemoryConfig(
        memory_limit_gb=8.0,
        enable_mixed_precision_fallback=True,
        safety_factor=0.8,
        progress_reporting=True,
    )

    # Use memory context for temporary settings
    with memory_context(config):
        # Memory-optimized fitting
        cf = CurveFit()
        _popt, _pcov = cf.curve_fit(func, x, y, p0=p0)

    # Check current memory configuration
    current_config = get_memory_config()
    print(f"Memory limit: {current_config.memory_limit_gb} GB")
    print(f"Mixed precision fallback: {current_config.enable_mixed_precision_fallback}")
    print("✅ Memory Management example passed")


def test_algorithm_selection():
    """Test Algorithm Selection example."""
    print("\nTesting Algorithm Selection example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import curve_fit
    from nlsq.precision.algorithm_selector import auto_select_algorithm

    # Sample data
    x = np.linspace(0, 10, 100)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.randn(100) * 0.05

    # Define your model
    def model_nonlinear(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Auto-select best algorithm
    recommendations = auto_select_algorithm(
        f=model_nonlinear, xdata=x, ydata=y, p0=[1.0, 0.5, 0.1]
    )

    # Use recommended algorithm
    method = recommendations.get("algorithm", "trf")
    popt, _pcov = curve_fit(model_nonlinear, x, y, p0=[1.0, 0.5, 0.1], method=method)

    print(f"Selected algorithm: {method}")
    print(f"Fitted parameters: {popt}")
    print("✅ Algorithm Selection example passed")


def test_diagnostics_monitoring():
    """Test Diagnostics & Monitoring example."""
    print("\nTesting Diagnostics & Monitoring example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import ConvergenceMonitor, CurveFit
    from nlsq.utils.diagnostics import OptimizationDiagnostics

    # Sample data
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100) * 0.1
    p0 = [1.0, 0.0]

    def func(x, a, b):
        return a * x + b

    # Create convergence monitor
    monitor = ConvergenceMonitor(window_size=10, sensitivity=1.0)

    # Use CurveFit with stability features
    cf = CurveFit(enable_stability=True, enable_recovery=True)

    # Perform fitting
    popt, _pcov = cf.curve_fit(func, x, y, p0=p0)
    print(f"Fitted parameters: {popt}")

    # For detailed diagnostics, create separate diagnostics object
    OptimizationDiagnostics()
    print("✅ Diagnostics & Monitoring example passed")


def test_caching_system():
    """Test Caching System example."""
    print("\nTesting Caching System example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import SmartCache, curve_fit

    # Sample data
    x1 = np.linspace(0, 10, 50)
    y1 = 2.0 * np.exp(-0.5 * x1) + np.random.randn(50) * 0.05
    x2 = np.linspace(0, 10, 50)
    y2 = 2.2 * np.exp(-0.4 * x2) + np.random.randn(50) * 0.05

    # Configure caching
    cache = SmartCache(max_memory_items=1000, disk_cache_enabled=True)

    # Define fit function (caching happens at the JIT level)
    def exponential(x, a, b):
        return a * jnp.exp(-b * x)

    # First fit - compiles function
    _popt1, _pcov1 = curve_fit(exponential, x1, y1, p0=[1.0, 0.1])

    # Second fit - reuses JIT compilation from first fit
    _popt2, _pcov2 = curve_fit(exponential, x2, y2, p0=[1.2, 0.15])

    # Check cache statistics
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    print("✅ Caching System example passed")


def test_optimization_recovery():
    """Test Optimization Recovery example."""
    print("\nTesting Optimization Recovery example...")
    np.random.seed(42)  # Ensure deterministic test results
    from nlsq import CurveFit, OptimizationRecovery, curve_fit

    # Sample data with some noise
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) * 0.1
    p0_initial = [1.0, 0.0]

    def func(x, a, b):
        return a * x + b

    # CurveFit with built-in recovery enabled
    cf = CurveFit(enable_recovery=True)

    try:
        popt, _pcov = cf.curve_fit(func, x, y, p0=p0_initial)
        print(f"Fitted parameters: {popt}")
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Manual recovery with OptimizationRecovery
        recovery = OptimizationRecovery(max_retries=3, enable_diagnostics=True)
        # Recovery provides automatic fallback strategies
        popt, _pcov = curve_fit(func, x, y, p0=p0_initial)
    print("✅ Optimization Recovery example passed")


def test_input_validation():
    """Test Input Validation example."""
    print("\nTesting Input Validation example...")
    from nlsq import InputValidator, curve_fit

    # Sample data
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.randn(50) * 0.1
    p0 = [1.0, 0.0]

    def func(x, a, b):
        return a * x + b

    # Create validator
    validator = InputValidator(fast_mode=True)

    # Validate inputs before fitting
    _warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
        f=func, xdata=x, ydata=y, p0=p0
    )

    if errors:
        print(f"Validation errors: {errors}")
    else:
        # Use validated data
        popt, _pcov = curve_fit(func, clean_x, clean_y, p0=p0)
        print(f"Fitted parameters: {popt}")
    print("✅ Input Validation example passed")


def main():
    """Run all tests."""
    tests = [
        test_basic_usage,
        test_exponential_with_jax,
        test_curve_fit_large,
        test_advanced_large_dataset,
        test_sparse_jacobian,
        test_streaming_optimizer,
        test_memory_management,
        test_algorithm_selection,
        test_diagnostics_monitoring,
        test_caching_system,
        test_optimization_recovery,
        test_input_validation,
    ]

    failed = []
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed.append((test.__name__, e))
            print(f"❌ {test.__name__} failed:")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed:
        print(f"\n❌ Failed tests ({len(failed)}):")
        for test_name, error in failed:
            print(f"  - {test_name}: {type(error).__name__}: {error}")
        sys.exit(1)
    else:
        print("\n✅ All README examples work correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
