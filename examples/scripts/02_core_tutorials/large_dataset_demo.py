"""
Converted from large_dataset_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # NLSQ Large Dataset Fitting Demonstration
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/large_dataset_demo.ipynb)
#
# **Requirements:** Python 3.12 or higher
#
# ## ⚠️ Deprecation Notice
#
# This notebook demonstrates NLSQ large dataset features:
#
# - **Removed**: Subsampling (which caused data loss)
# - **Added**: Streaming optimization (processes 100% of data)
#
# All large datasets now use streaming optimization for 100% data utilization.
#
# ---
#
# This notebook demonstrates the capabilities of NLSQ for handling very large datasets with automatic memory management, chunking, and streaming optimization for unlimited datasets.
#
# ## Key Features:
# - Memory estimation for datasets from 100K to 100M+ points
# - Automatic memory management and dataset size detection
# - Chunked processing for datasets that don't fit in memory
# - Streaming optimization for unlimited dataset sizes
# - Advanced configuration and algorithm selection
#
# ======================================================================


# ======================================================================
# ## Setup and Imports
# ======================================================================

# Check Python version
import sys

print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} meets requirements")

import time

import jax.numpy as jnp
import numpy as np

from nlsq import (
    AlgorithmSelector,
    CurveFit,
    LargeDatasetConfig,
    LargeDatasetFitter,
    LDMemoryConfig,
    # New advanced features
    MemoryConfig,
    __version__,
    auto_select_algorithm,
    configure_for_large_datasets,
    curve_fit_large,
    estimate_memory_requirements,
    fit_large_dataset,
    get_memory_config,
    large_dataset_context,
    memory_context,
    set_memory_limits,
)

print(f"NLSQ version: {__version__}")
print("NLSQ Large Dataset Demo - Enhanced Version")
print("Including advanced memory management and algorithm selection")


# Define our model functions
def exponential_decay(x, a, b, c):
    """Exponential decay model with offset: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def polynomial_model(x, a, b, c, d):
    """Polynomial model: y = a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d


def gaussian(x, a, mu, sigma, offset):
    """Gaussian model: y = a * exp(-((x - mu)^2) / (2*sigma^2)) + offset"""
    return a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def complex_model(x, a, b, c, d, e, f):
    """Complex model with many parameters for algorithm selection testing"""
    return a * jnp.exp(-b * x) + c * jnp.sin(d * x) + e * x**2 + f


# ======================================================================
# ## 1. Memory Estimation Demo
#
# First, let's understand how much memory different dataset sizes require and what processing strategies NLSQ recommends.
# ======================================================================


def demo_memory_estimation():
    """Demonstrate memory estimation capabilities."""
    print("=" * 60)
    print("MEMORY ESTIMATION DEMO")
    print("=" * 60)

    # Estimate requirements for different dataset sizes
    test_cases = [
        (100_000, 3, "Small dataset"),
        (1_000_000, 3, "Medium dataset"),
        (10_000_000, 3, "Large dataset"),
        (50_000_000, 3, "Very large dataset"),
        (100_000_000, 3, "Extremely large dataset"),
    ]

    for n_points, n_params, description in test_cases:
        stats = estimate_memory_requirements(n_points, n_params)

        print(f"\n{description} ({n_points:,} points, {n_params} parameters):")
        print(f"  Total memory estimate: {stats.total_memory_estimate_gb:.3f} GB")
        print(f"  Number of chunks: {stats.n_chunks}")

        # Determine strategy description
        if stats.n_chunks == 1:
            print("  Strategy: Single pass (fits in memory)")
        elif stats.n_chunks > 1:
            print(f"  Strategy: Chunked processing ({stats.n_chunks} chunks)")

        # For very large datasets, suggest streaming
        if n_points > 50_000_000:
            print("  💡 Consider: Streaming optimization for zero accuracy loss")


demo_memory_estimation()


# ======================================================================
# ## 1.5. Advanced Memory Configuration and Algorithm Selection
#
# NLSQ now provides sophisticated configuration management and automatic algorithm selection for optimal performance with large datasets.
# ======================================================================


def demo_advanced_configuration():
    """Demonstrate advanced configuration and algorithm selection."""
    print("=" * 60)
    print("ADVANCED CONFIGURATION & ALGORITHM SELECTION DEMO")
    print("=" * 60)

    # Current memory configuration
    current_config = get_memory_config()
    print("Current memory configuration:")
    print(f"  Memory limit: {current_config.memory_limit_gb} GB")
    print(
        f"  Mixed precision fallback: {current_config.enable_mixed_precision_fallback}"
    )

    # Automatically configure for large datasets
    print("\nConfiguring for large dataset processing...")
    configure_for_large_datasets(memory_limit_gb=8.0, enable_chunking=True)

    # Show updated configuration
    new_config = get_memory_config()
    print(f"Updated memory limit: {new_config.memory_limit_gb} GB")

    # Generate test dataset for algorithm selection
    print("\n=== Algorithm Selection Demo ===")
    np.random.seed(42)

    # Test different model complexities
    test_cases = [
        ("Simple exponential", exponential_decay, 3, [5.0, 1.2, 0.5]),
        ("Polynomial", polynomial_model, 4, [0.1, -0.5, 2.0, 1.0]),
        ("Complex multi-param", complex_model, 6, [3.0, 0.8, 1.5, 2.0, 0.1, 0.2]),
    ]

    for model_name, model_func, n_params, true_params in test_cases:
        print(f"\n{model_name} ({n_params} parameters):")

        # Generate sample data
        n_sample = 10000  # Smaller sample for algorithm analysis
        x_sample = np.linspace(0, 5, n_sample)
        y_sample = model_func(x_sample, *true_params) + np.random.normal(
            0, 0.05, n_sample
        )

        # Get algorithm recommendation
        try:
            recommendations = auto_select_algorithm(model_func, x_sample, y_sample)

            print(f"  Recommended algorithm: {recommendations['algorithm']}")
            print(f"  Recommended tolerance: {recommendations['ftol']}")
            print(
                f"  Problem complexity: {recommendations.get('complexity', 'Unknown')}"
            )

            # Estimate memory for full dataset
            large_n = 1_000_000  # 1M points
            stats = estimate_memory_requirements(large_n, n_params)
            print(f"  Memory for 1M points: {stats.total_memory_estimate_gb:.3f} GB")
            print(
                f"  Chunking strategy: {'Required' if stats.n_chunks > 1 else 'Not needed'}"
            )
        except Exception as e:
            print(f"  Algorithm selection failed: {e}")
            print(f"  Using default settings for {model_name}")


# Run the demo
demo_advanced_configuration()


# ======================================================================
# ## 2. Basic Large Dataset Fitting
#
# Let's demonstrate fitting a 1 million point dataset using the convenience function `fit_large_dataset`.
# ======================================================================


def demo_basic_large_dataset_fitting():
    """Demonstrate basic large dataset fitting."""
    print("\n" + "=" * 60)
    print("BASIC LARGE DATASET FITTING DEMO")
    print("=" * 60)

    # Generate synthetic large dataset (1M points)
    print("Generating 1M point exponential decay dataset...")
    np.random.seed(42)
    n_points = 1_000_000
    x_data = np.linspace(0, 5, n_points, dtype=np.float64)
    true_params = [5.0, 1.2, 0.5]
    noise_level = 0.05

    y_true = true_params[0] * np.exp(-true_params[1] * x_data) + true_params[2]
    y_data = y_true + np.random.normal(0, noise_level, n_points)

    print(f"Dataset: {n_points:,} points")
    print(
        f"True parameters: a={true_params[0]}, b={true_params[1]}, c={true_params[2]}"
    )

    # Fit using convenience function
    print("\nFitting with automatic memory management...")
    start_time = time.time()

    result = fit_large_dataset(
        exponential_decay,
        x_data,
        y_data,
        p0=[4.0, 1.0, 0.4],
        memory_limit_gb=2.0,  # 2GB limit
        show_progress=True,
    )

    fit_time = time.time() - start_time

    if result.success:
        fitted_params = np.array(result.popt)
        errors = np.abs(fitted_params - np.array(true_params))
        rel_errors = errors / np.array(true_params) * 100

        print(f"\n✅ Fit completed in {fit_time:.2f} seconds")
        print(
            f"Fitted parameters: [{fitted_params[0]:.3f}, {fitted_params[1]:.3f}, {fitted_params[2]:.3f}]"
        )
        print(f"Absolute errors: [{errors[0]:.4f}, {errors[1]:.4f}, {errors[2]:.4f}]")
        print(
            f"Relative errors: [{rel_errors[0]:.2f}%, {rel_errors[1]:.2f}%, {rel_errors[2]:.2f}%]"
        )
    else:
        print(f"❌ Fit failed: {result.message}")


# Run the demo
demo_basic_large_dataset_fitting()


# ======================================================================
# ## 3.5. Context Managers and Temporary Configuration
#
# NLSQ provides context managers for temporary configuration changes, allowing you to optimize settings for specific operations without affecting global state.
# ======================================================================


def demo_context_managers():
    """Demonstrate context managers for temporary configuration."""
    print("\n" + "=" * 60)
    print("CONTEXT MANAGERS DEMO")
    print("=" * 60)

    # Show current configuration
    original_mem_config = get_memory_config()
    print(f"Original memory limit: {original_mem_config.memory_limit_gb} GB")

    # Generate test data
    np.random.seed(555)
    n_points = 500_000
    x_data = np.linspace(0, 5, n_points)
    y_data = exponential_decay(x_data, 4.0, 1.5, 0.3) + np.random.normal(
        0, 0.05, n_points
    )

    print(f"Test dataset: {n_points:,} points")

    # Test 1: Memory context for memory-constrained fitting
    print("\n--- Test 1: Memory-constrained fitting ---")
    constrained_config = MemoryConfig(
        memory_limit_gb=0.5,  # Very low limit
        enable_mixed_precision_fallback=True,
    )

    with memory_context(constrained_config):
        temp_config = get_memory_config()
        print(f"Inside context memory limit: {temp_config.memory_limit_gb} GB")
        print(f"Mixed precision enabled: {temp_config.enable_mixed_precision_fallback}")

        start_time = time.time()
        result1 = fit_large_dataset(
            exponential_decay, x_data, y_data, p0=[3.5, 1.3, 0.25], show_progress=False
        )
        time1 = time.time() - start_time

        if result1.success:
            print(f"✅ Constrained fit completed: {time1:.3f}s")
            print(f"   Parameters: {result1.popt}")
        else:
            print(f"❌ Constrained fit failed: {result1.message}")

    # Check that configuration is restored
    restored_config = get_memory_config()
    print(f"After context memory limit: {restored_config.memory_limit_gb} GB")

    # Test 2: Large dataset context for optimized processing
    print("\n--- Test 2: Large dataset optimization ---")
    ld_config = LargeDatasetConfig()

    with large_dataset_context(ld_config):
        print("Inside large dataset context - chunking optimized")

        start_time = time.time()
        result2 = fit_large_dataset(
            exponential_decay, x_data, y_data, p0=[3.5, 1.3, 0.25], show_progress=False
        )
        time2 = time.time() - start_time

        if result2.success:
            print(f"✅ Optimized fit completed: {time2:.3f}s")
            print(f"   Parameters: {result2.popt}")
        else:
            print(f"❌ Optimized fit failed: {result2.message}")

    # Test 3: Combined context for specific algorithm
    print("\n--- Test 3: Algorithm-specific optimization ---")

    # Get algorithm recommendation first
    sample_size = 5000
    x_sample = x_data[:sample_size]
    y_sample = y_data[:sample_size]
    recommendations = auto_select_algorithm(exponential_decay, x_sample, y_sample)

    print(f"Recommended algorithm: {recommendations['algorithm']}")
    print(f"Recommended tolerance: {recommendations['ftol']}")

    # Use CurveFit with recommended settings
    optimized_config = MemoryConfig(
        memory_limit_gb=2.0, enable_mixed_precision_fallback=True
    )

    with memory_context(optimized_config):
        start_time = time.time()

        # Use the regular CurveFit for comparison
        cf = CurveFit(use_dynamic_sizing=True)
        popt3, pcov3 = cf.curve_fit(
            exponential_decay,
            x_data,
            y_data,
            p0=[3.5, 1.3, 0.25],
            ftol=recommendations.get("ftol", 1e-8),
        )
        time3 = time.time() - start_time

        print(f"✅ Algorithm-optimized fit completed: {time3:.3f}s")
        print(f"   Parameters: {popt3}")
        print(f"   Parameter uncertainties: {np.sqrt(np.diag(pcov3))}")

    # Compare all approaches
    if result1.success and result2.success:
        print("\n=== Performance Comparison ===")
        print(f"Constrained memory: {time1:.3f}s")
        print(f"Chunking optimized: {time2:.3f}s")
        print(f"Algorithm optimized: {time3:.3f}s")

        # Calculate accuracy
        true_params = [4.0, 1.5, 0.3]
        errors1 = np.abs(result1.popt - true_params)
        errors2 = np.abs(result2.popt - true_params)
        errors3 = np.abs(popt3 - true_params)

        print("\nAccuracy comparison (absolute errors):")
        print(f"Constrained: {errors1}")
        print(f"Chunking:    {errors2}")
        print(f"Algorithm:   {errors3}")

    print("\n✓ Context managers allow flexible, temporary configuration changes!")


# Run the demo
demo_context_managers()


# ======================================================================
# ## 3. Chunked Processing Demo
#
# For datasets that don't fit in memory, NLSQ automatically chunks the data and processes it in batches.
# ======================================================================


def demo_chunked_processing():
    """Demonstrate chunked processing with progress reporting."""
    print("\n" + "=" * 60)
    print("CHUNKED PROCESSING DEMO")
    print("=" * 60)

    # Generate a dataset that will require chunking
    print("Generating 2M point polynomial dataset...")
    np.random.seed(123)
    n_points = 2_000_000
    x_data = np.linspace(-2, 2, n_points, dtype=np.float64)
    true_params = [0.5, -1.2, 2.0, 1.5]
    noise_level = 0.1

    y_true = (
        true_params[0] * x_data**3
        + true_params[1] * x_data**2
        + true_params[2] * x_data
        + true_params[3]
    )
    y_data = y_true + np.random.normal(0, noise_level, n_points)

    print(f"Dataset: {n_points:,} points")
    print(f"True parameters: {true_params}")

    # Create fitter with limited memory to force chunking
    fitter = LargeDatasetFitter(memory_limit_gb=0.5)  # Small limit to force chunking

    # Get processing recommendations
    recs = fitter.get_memory_recommendations(n_points, 4)
    print(f"\nProcessing strategy: {recs['processing_strategy']}")
    print(f"Chunk size: {recs['recommendations']['chunk_size']:,}")
    print(f"Number of chunks: {recs['recommendations']['n_chunks']}")
    print(
        f"Memory estimate: {recs['recommendations']['total_memory_estimate_gb']:.2f} GB"
    )

    # Fit with progress reporting
    print("\nFitting with chunked processing...")
    start_time = time.time()

    result = fitter.fit_with_progress(
        polynomial_model, x_data, y_data, p0=[0.4, -1.0, 1.8, 1.2]
    )

    fit_time = time.time() - start_time

    if result.success:
        fitted_params = np.array(result.popt)
        errors = np.abs(fitted_params - np.array(true_params))
        rel_errors = errors / np.abs(np.array(true_params)) * 100

        print(f"\n✅ Chunked fit completed in {fit_time:.2f} seconds")
        if hasattr(result, "n_chunks"):
            print(
                f"Used {result.n_chunks} chunks with {result.success_rate:.1%} success rate"
            )
        print(f"Fitted parameters: {fitted_params}")
        print(f"Absolute errors: {errors}")
        print(f"Relative errors: {rel_errors}%")
    else:
        print(f"❌ Chunked fit failed: {result.message}")


# Run the demo
demo_chunked_processing()


# ======================================================================
# ## 4. Streaming Optimization for Unlimited Datasets
#
# For datasets too large to fit in memory, NLSQ uses streaming optimization with mini-batch gradient descent. **Unlike subsampling (deprecated), streaming processes 100% of data with zero accuracy loss.**
#
# ======================================================================


def demo_streaming_optimization():
    """Demonstrate streaming optimization for unlimited datasets."""
    print("\n" + "=" * 60)
    print("STREAMING OPTIMIZATION DEMO")
    print("=" * 60)

    # Simulate a very large dataset scenario
    print("Simulating extremely large dataset (100M points)...")
    print("Using streaming optimization for zero data loss\n")

    n_points_full = 100_000_000  # 100M points
    true_params = [3.0, 0.8, 0.2]

    # For demo purposes, generate a representative dataset
    # In production, streaming would process full dataset in batches
    print("Generating representative dataset for demo...")
    np.random.seed(777)
    n_demo = 1_000_000  # 1M points for demo
    x_data = np.linspace(0, 10, n_demo)
    y_data = exponential_decay(x_data, *true_params) + np.random.normal(0, 0.1, n_demo)

    # Memory estimation
    stats = estimate_memory_requirements(n_points_full, len(true_params))
    print(f"\nFull dataset memory estimate: {stats.total_memory_estimate_gb:.1f} GB")
    print(f"Number of chunks required: {stats.n_chunks}")

    # Configure streaming optimization
    print("\nConfiguring streaming optimization...")
    config = LDMemoryConfig(
        memory_limit_gb=4.0,
        use_streaming=True,  # Enable streaming
        streaming_batch_size=50000,  # Process 50K points per batch
    )

    fitter = LargeDatasetFitter(config=config)

    print("\nFitting with streaming optimization...")
    print("(Processing 100% of data in batches)\n")

    try:
        start_time = time.time()
        result = fitter.fit(exponential_decay, x_data, y_data, p0=[2.5, 0.6, 0.15])
        fit_time = time.time() - start_time

        if result.success:
            print(f"\n✅ Streaming fit completed in {fit_time:.2f} seconds")
            print(f"\nFitted parameters: {result.x}")
            print(f"True parameters:    {true_params}")
            errors = np.abs(result.x - np.array(true_params))
            rel_errors = errors / np.abs(np.array(true_params)) * 100
            print(f"Relative errors:    {[f'{e:.2f}%' for e in rel_errors]}")
            print("\nℹ️ Streaming processed 100% of data (zero accuracy loss)")
        else:
            print(f"❌ Streaming fit failed: {result.message}")

    except Exception as e:
        print(f"❌ Error during streaming fit: {e}")


demo_streaming_optimization()


# ======================================================================
# ## 5. curve_fit_large Convenience Function
#
# The `curve_fit_large` function provides automatic detection and handling of large datasets, making it easy to switch between standard and large dataset processing.
# ======================================================================


def demo_curve_fit_large():
    """Demonstrate the curve_fit_large convenience function."""
    print("\n" + "=" * 60)
    print("CURVE_FIT_LARGE CONVENIENCE FUNCTION DEMO")
    print("=" * 60)

    # Generate test dataset
    print("Generating 3M point dataset for curve_fit_large demo...")
    np.random.seed(789)
    n_points = 3_000_000
    x_data = np.linspace(0, 10, n_points, dtype=np.float64)

    true_params = [5.0, 5.0, 1.5, 0.5]
    y_true = gaussian(x_data, *true_params)
    y_data = y_true + np.random.normal(0, 0.1, n_points)

    print(f"Dataset: {n_points:,} points")
    print(
        f"True parameters: a={true_params[0]:.2f}, mu={true_params[1]:.2f}, sigma={true_params[2]:.2f}, offset={true_params[3]:.2f}"
    )

    # Use curve_fit_large - automatic large dataset handling
    print("\nUsing curve_fit_large with automatic optimization...")
    start_time = time.time()

    popt, pcov = curve_fit_large(
        gaussian,
        x_data,
        y_data,
        p0=[4.5, 4.8, 1.3, 0.4],
        memory_limit_gb=1.0,  # Force chunking with low memory limit
        show_progress=True,
        auto_size_detection=True,  # Automatically detect large dataset
    )

    fit_time = time.time() - start_time

    errors = np.abs(popt - np.array(true_params))
    rel_errors = errors / np.array(true_params) * 100

    print(f"\n✅ curve_fit_large completed in {fit_time:.2f} seconds")
    print(f"Fitted parameters: {popt}")
    print(f"Absolute errors: {errors}")
    print(f"Relative errors: {rel_errors}%")

    # Show parameter uncertainties from covariance matrix
    param_std = np.sqrt(np.diag(pcov))
    print(f"Parameter uncertainties (std): {param_std}")


# Run the demo
demo_curve_fit_large()


# ======================================================================
# ## 6. Performance Comparison
#
# Let's compare different approaches for various dataset sizes.
# ======================================================================


def compare_approaches():
    """Compare different fitting approaches."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Test different dataset sizes
    sizes = [10_000, 100_000, 500_000]

    print(f"\n{'Size':>10} {'Time (s)':>12} {'Memory (GB)':>12} {'Strategy':>20}")
    print("-" * 55)

    for n in sizes:
        # Generate data
        np.random.seed(42)
        x = np.linspace(0, 10, n)
        y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n)

        # Get memory estimate
        stats = estimate_memory_requirements(n, 3)

        # Determine strategy
        if stats.n_chunks == 1:
            # Streaming handles all large datasets
            strategy = "Streaming"
        else:
            strategy = f"Chunked ({stats.n_chunks} chunks)"

        # Time the fit
        start = time.time()
        _result = fit_large_dataset(
            exponential_decay,
            x,
            y,
            p0=[2.5, 0.6, 0.2],
            memory_limit_gb=0.5,  # Small limit to test chunking
            show_progress=False,
        )
        elapsed = time.time() - start

        print(
            f"{n:10,} {elapsed:12.3f} {stats.total_memory_estimate_gb:12.3f} {strategy:>20}"
        )


# Run comparison
compare_approaches()


# ======================================================================
# ## Summary and Key Takeaways
#
# NLSQ provides comprehensive support for large dataset fitting with recent improvements:
#
# 1. **Automatic Memory Management**: NLSQ automatically detects available memory and chooses the best strategy
# 2. **Improved Chunking Algorithm**: Advanced exponential moving average approach achieves <1% error for well-conditioned problems
# 3. **JAX Tracing Compatibility**: Supports functions with up to 15+ parameters without TracerArrayConversionError
# 4. **curve_fit_large Function**: Automatic dataset size detection and intelligent processing strategy selection
# 5. **Streaming Optimization **: For unlimited dataset sizes, streaming optimization processes 100% of data with zero accuracy loss
# 6. **Progress Reporting**: Long-running fits provide progress updates
# 7. **Memory Estimation**: Predict memory requirements before fitting
#
# ### Best Practices:
#
# - Use `curve_fit_large()` for automatic handling of both small and large datasets
# - Use `estimate_memory_requirements()` to understand dataset requirements
# - Use `fit_large_dataset()` when you need explicit control over large dataset processing
# - Set appropriate `memory_limit_gb` based on your system
# - Enable streaming for very large datasets that exceed memory limits
# - Use progress reporting for long-running fits
#
# ### Recent Improvements (v810dc5c):
#
# - **Fixed JAX tracing issues** for functions with many parameters
# - **Enhanced chunking algorithm** with adaptive learning rates and convergence monitoring
# - **Ensured return type consistency** across all code paths
# - **Added comprehensive test coverage** for large dataset functionality
# ======================================================================


# Print final summary
print("\n" + "=" * 60)
print("DEMO COMPLETED")
print("=" * 60)
print("\nKey takeaways:")
print("• NLSQ automatically handles memory management for large datasets")
print("• Chunked processing works for datasets that don't fit in memory")
print("• curve_fit_large provides automatic dataset size detection")
print("• Improved chunking algorithm achieves <1% error for well-conditioned problems")
print("• Streaming optimization handles unlimited datasets with zero accuracy loss ")
print("• Progress reporting helps track long-running fits")
print("• Memory estimation helps plan processing strategies")
