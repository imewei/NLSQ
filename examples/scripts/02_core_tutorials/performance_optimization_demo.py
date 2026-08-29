"""
Converted from performance_optimization_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # NLSQ Performance Optimization Features
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/performance_optimization_demo.ipynb)
#
# **Requirements:** Python 3.12 or higher
#
# This notebook demonstrates NLSQ's advanced performance optimization features:
#
# - **SparseJacobian**: Exploit sparsity patterns for 10-100x memory reduction
# - **AdaptiveHybridStreamingOptimizer**: Process huge datasets with streaming
#
# These features are essential for:
# - Very large problems (millions of data points)
# - Memory-constrained environments
# - Real-time or low-latency applications
# - Problems with structured sparsity patterns
# ======================================================================
# ======================================================================
# ## Setup and Imports
# ======================================================================
# Install NLSQ if not already installed
# !pip install nlsq  # Uncomment to install in notebook environment
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} meets requirements")

# Import core NLSQ
from nlsq import CurveFit, __version__

print(f"NLSQ version: {__version__}")
print(f"JAX devices: {jax.devices()}")


# Import advanced performance features
from nlsq import (
    AdaptiveHybridStreamingOptimizer,
    HybridStreamingConfig,
    # Sparse Jacobian
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
)

print("✅ Advanced features imported successfully")


# ======================================================================
# ## 1. SparseJacobian - Exploiting Sparsity for Memory Efficiency
#
# Many curve fitting problems have **sparse Jacobians** where each data point only depends on a subset of parameters. The `SparseJacobianComputer` exploits this structure for:
#
# - **10-100x memory reduction**: Store only non-zero elements
# - **Faster computation**: Skip zero elements in matrix operations
# - **Larger problems**: Fit problems that wouldn't fit in memory otherwise
#
# ### When to Use Sparse Jacobians
#
# - **Piecewise models**: Different parameters for different data regions
# - **Multi-component fits**: Independent sub-models
# - **Localized parameters**: Parameters affecting only nearby data points
# - **Very large problems**: Millions of data points
# ======================================================================


def demo_sparse_jacobian_basics():
    """Demonstrate sparse Jacobian detection and usage."""
    print("=" * 70)
    print("SPARSE JACOBIAN BASICS")
    print("=" * 70)

    # Create a piecewise model with sparse Jacobian
    # Each segment only depends on 2 parameters
    def piecewise_linear(x, *params):
        """Piecewise linear model with sparse Jacobian.

        params[0:2] affect x < 0.5
        params[2:4] affect x >= 0.5
        """
        result = jnp.zeros_like(x)
        mask1 = x < 0.5
        mask2 = x >= 0.5

        # Segment 1: y = a1*x + b1
        result = jnp.where(mask1, params[0] * x + params[1], result)

        # Segment 2: y = a2*x + b2
        result = jnp.where(mask2, params[2] * x + params[3], result)

        return result

    # Generate test data
    np.random.seed(42)
    n_points = cap_samples(1000)
    x_data = np.linspace(0, 1, n_points)
    true_params = [2.0, 1.0, -1.0, 2.0]  # a1, b1, a2, b2

    # Detect sparsity pattern
    print("\n--- Detecting Sparsity Pattern ---")
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)

    pattern, sparsity = sparse_computer.detect_sparsity_pattern(
        piecewise_linear, np.array(true_params), x_data, n_samples=30 if QUICK else 100
    )

    print(f"Detected sparsity: {sparsity:.1%} zero elements")
    print(f"Pattern shape: {pattern.shape}")
    print(f"Non-zero elements: {np.sum(pattern):,} / {pattern.size:,}")

    # Visualize sparsity pattern
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pattern.T, aspect="auto", cmap="binary", interpolation="nearest")
    plt.xlabel("Data Point")
    plt.ylabel("Parameter")
    plt.title("Sparsity Pattern (Black = Non-zero)")
    plt.colorbar(label="Non-zero")

    plt.subplot(1, 2, 2)
    # Show which parameters affect which data regions
    param_density = np.sum(pattern, axis=0)
    plt.bar(range(len(true_params)), param_density)
    plt.xlabel("Parameter Index")
    plt.ylabel("Number of Non-zero Entries")
    plt.title("Parameter Activity")
    plt.xticks(range(len(true_params)), ["a1", "b1", "a2", "b2"])

    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "performance_optimization_demo"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ Sparse Jacobian pattern detected successfully")


# Run demo
demo_sparse_jacobian_basics()


# ======================================================================
# ### Memory Savings Analysis
# ======================================================================


def analyze_sparse_jacobian_memory_savings():
    """Analyze memory savings for different problem sizes."""
    print("=" * 70)
    print("SPARSE JACOBIAN MEMORY SAVINGS ANALYSIS")
    print("=" * 70)

    sparse_computer = SparseJacobianComputer()

    # Test different problem sizes
    test_cases = [
        (10_000, 10, 0.90, "Small, highly sparse"),
        (100_000, 20, 0.95, "Medium, highly sparse"),
        (1_000_000, 50, 0.99, "Large, extremely sparse"),
        (10_000_000, 100, 0.995, "Very large, ultra-sparse"),
    ]

    print(
        "\n{:>12} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12}".format(
            "Data Points",
            "Params",
            "Sparsity",
            "Dense (GB)",
            "Sparse (GB)",
            "Savings",
            "Reduction",
        )
    )
    print("-" * 92)

    for n_data, n_params, sparsity, description in test_cases:
        memory_info = sparse_computer.estimate_memory_usage(n_data, n_params, sparsity)

        print(
            "{:12,} {:8} {:9.1%} {:12.2f} {:12.2f} {:11.1f}% {:11.1f}x".format(
                n_data,
                n_params,
                sparsity,
                memory_info["dense_gb"],
                memory_info["sparse_gb"],
                memory_info["savings_percent"],
                memory_info["reduction_factor"],
            )
        )

    print("\n📊 Key Insights:")
    print("  • Sparse representation can reduce memory by 10-100x")
    print("  • Benefits increase dramatically with problem size and sparsity")
    print("  • Enables fitting problems that wouldn't fit in memory otherwise")
    print("  • Overhead is minimal for problems with >90% sparsity")


# Run analysis
analyze_sparse_jacobian_memory_savings()


# ======================================================================
# ### Using Sparse Jacobians in Practice
# ======================================================================


def demo_sparse_jacobian_fitting():
    """Demonstrate fitting with sparse Jacobians."""
    print("=" * 70)
    print("FITTING WITH SPARSE JACOBIANS")
    print("=" * 70)

    # Multi-component Gaussian model (each Gaussian independent)
    def multi_gaussian(x, *params):
        """Sum of N independent Gaussians.

        params = [a1, mu1, sigma1, a2, mu2, sigma2, ...]
        Each Gaussian only affects data near its center (sparse!)
        """
        n_gaussians = len(params) // 3
        result = jnp.zeros_like(x)

        for i in range(n_gaussians):
            a = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            result = result + a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        return result

    # Generate data with 5 Gaussians
    np.random.seed(123)
    n_points = cap_samples(5000)
    x_data = np.linspace(0, 10, n_points)

    # True parameters: 5 Gaussians at different locations
    true_params = [
        1.0,
        2.0,
        0.3,  # Gaussian 1
        0.8,
        4.0,
        0.4,  # Gaussian 2
        1.2,
        6.0,
        0.3,  # Gaussian 3
        0.9,
        7.5,
        0.35,  # Gaussian 4
        1.1,
        9.0,
        0.4,  # Gaussian 5
    ]

    y_data = multi_gaussian(x_data, *true_params) + np.random.normal(0, 0.02, n_points)

    # Detect sparsity
    print("\n--- Analyzing Sparsity ---")
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.001)
    pattern, sparsity = sparse_computer.detect_sparsity_pattern(
        multi_gaussian, np.array(true_params), x_data, n_samples=50 if QUICK else 200
    )

    print(f"Detected sparsity: {sparsity:.1%}")

    # Estimate memory savings
    memory_info = sparse_computer.estimate_memory_usage(
        n_points, len(true_params), sparsity
    )

    print("\nMemory comparison:")
    print(f"  Dense Jacobian: {memory_info['dense_gb'] * 1024:.1f} MB")
    print(f"  Sparse Jacobian: {memory_info['sparse_gb'] * 1024:.1f} MB")
    print(f"  Savings: {memory_info['savings_percent']:.1f}%")
    print(f"  Reduction factor: {memory_info['reduction_factor']:.1f}x")

    # Visualize the data and sparsity
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot data and true fit
    axes[0].plot(x_data, y_data, "b.", alpha=0.3, markersize=1, label="Data")
    axes[0].plot(
        x_data,
        multi_gaussian(x_data, *true_params),
        "r-",
        linewidth=2,
        label="True Model",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Multi-Gaussian Data (5 components)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot sparsity pattern
    sample_pattern = pattern[::25]  # Sample for visualization
    im = axes[1].imshow(
        sample_pattern.T, aspect="auto", cmap="binary", interpolation="nearest"
    )
    axes[1].set_xlabel("Data Point (sampled)")
    axes[1].set_ylabel("Parameter")
    axes[1].set_title(f"Jacobian Sparsity Pattern ({sparsity:.1%} zeros)")

    # Add parameter labels
    param_labels = []
    for i in range(5):
        param_labels.extend([f"a{i + 1}", f"μ{i + 1}", f"σ{i + 1}"])
    axes[1].set_yticks(range(len(true_params)))
    axes[1].set_yticklabels(param_labels, fontsize=8)

    plt.colorbar(im, ax=axes[1], label="Non-zero")
    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "performance_optimization_demo"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ Sparse Jacobian analysis complete")
    print("\n💡 Notice how each Gaussian parameter only affects a localized region!")


demo_sparse_jacobian_fitting()


# ======================================================================
# ## 2. Adaptive Hybrid Streaming - Huge Dataset Size
#
# The adaptive hybrid streaming optimizer processes data in chunks while
# providing L-BFGS warmup and Gauss-Newton refinement.
#
# - **Huge dataset size**: Process datasets larger than RAM with bounded memory
# - **Accurate covariance**: Exact J^T J accumulation
# - **Defense layers**: Protects warmup from divergence
# ======================================================================


def demo_adaptive_hybrid_streaming():
    """Demonstrate adaptive hybrid streaming usage."""
    print("=" * 70)
    print("ADAPTIVE HYBRID STREAMING")
    print("=" * 70)

    def exponential_model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Simulate large dataset (reduced in quick mode for faster CI)
    total_data_size = 5_000 if QUICK else 50_000
    true_params = [5.0, 1.2, 0.5]

    x_data = np.linspace(0, 10, total_data_size)
    y_data = exponential_model(x_data, *true_params)
    y_data += np.random.normal(0, 0.05, total_data_size)

    config = HybridStreamingConfig(
        chunk_size=1000,
        gauss_newton_max_iterations=10,
    )
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    p0 = np.array([4.0, 1.0, 0.4])

    result = optimizer.fit((x_data, y_data), exponential_model, p0=p0, verbose=0)

    print(f"Total dataset size: {total_data_size:,} points")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Initial guess: {list(p0)}")
    print(f"Fitted params: {result['x']}")
    print("\n✅ Adaptive hybrid streaming demo complete")


if QUICK:
    print(
        "\n⏩ Quick mode: skipping adaptive hybrid streaming and combined optimization."
    )
else:
    demo_adaptive_hybrid_streaming()


# ======================================================================
# ## 3. Combined Example: All Features Together
#
# Let's demonstrate how to combine sparse Jacobians with streaming for maximum performance on a large, sparse problem.
# ======================================================================


def demo_combined_optimization():
    """Combine SparseJacobian and smart chunking."""
    print("=" * 70)
    print("COMBINED OPTIMIZATION DEMO")
    print("=" * 70)

    # Large, sparse problem: Multi-peak Gaussian
    def multi_peak_model(x, *params):
        """Multiple Gaussians - sparse Jacobian."""
        n_peaks = len(params) // 3
        result = jnp.zeros_like(x)

        for i in range(n_peaks):
            a = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            result = result + a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        return result

    # Problem size (reduced in quick mode for faster CI)
    n_points = 5_000 if QUICK else 50_000
    n_peaks = 4 if QUICK else 10
    n_params = n_peaks * 3  # 30 parameters

    print("\nProblem size:")
    print(f"  Data points: {n_points:,}")
    print(f"  Parameters: {n_params}")
    print(f"  Peaks: {n_peaks}")

    # Generate data
    np.random.seed(456)
    x_data = np.linspace(0, 20, n_points)

    # True parameters: peaks at regular intervals
    true_params = []
    for i in range(n_peaks):
        a = 0.8 + 0.4 * np.random.random()
        mu = (i + 0.5) * 20 / n_peaks
        sigma = 0.3 + 0.2 * np.random.random()
        true_params.extend([a, mu, sigma])

    y_data = multi_peak_model(x_data, *true_params) + np.random.normal(
        0, 0.02, n_points
    )

    # Step 1: Analyze sparsity
    print("\n--- Step 1: Sparsity Analysis ---")
    sparse_comp = SparseJacobianComputer(sparsity_threshold=0.001)
    pattern, sparsity = sparse_comp.detect_sparsity_pattern(
        multi_peak_model,
        np.array(true_params),
        x_data,
        n_samples=100 if QUICK else 500,
    )

    print(f"Sparsity: {sparsity:.1%}")

    memory_info = sparse_comp.estimate_memory_usage(n_points, n_params, sparsity)
    print("Memory savings:")
    print(f"  Dense: {memory_info['dense_gb'] * 1024:.1f} MB")
    print(f"  Sparse: {memory_info['sparse_gb'] * 1024:.1f} MB")
    print(f"  Reduction: {memory_info['reduction_factor']:.1f}x")

    # Step 2: Fit with standard NLSQ (for comparison)
    print("\n--- Step 2: Fitting ---")
    print("Using standard NLSQ curve fitting...")

    # Initial guess (slightly off)
    p0 = [p * np.random.uniform(0.9, 1.1) for p in true_params]

    start_time = time.time()
    cf = CurveFit()
    popt, pcov = cf.curve_fit(multi_peak_model, x_data, y_data, p0=p0)
    fit_time = time.time() - start_time

    # Calculate errors
    errors = np.abs(popt - np.array(true_params))
    rel_errors = errors / np.abs(np.array(true_params))

    print(f"\n✅ Fit completed in {fit_time:.2f}s")
    print(f"Max relative error: {np.max(rel_errors):.4f}")
    print(f"Mean relative error: {np.mean(rel_errors):.4f}")

    # Visualize results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot fit
    sample_idx = np.arange(0, len(x_data), 10)  # Sample for speed
    axes[0].plot(
        x_data[sample_idx],
        y_data[sample_idx],
        "b.",
        alpha=0.5,
        markersize=1,
        label="Data",
    )
    axes[0].plot(
        x_data,
        multi_peak_model(x_data, *true_params),
        "g-",
        linewidth=2,
        label="True",
        alpha=0.7,
    )
    axes[0].plot(
        x_data, multi_peak_model(x_data, *popt), "r--", linewidth=2, label="Fitted"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Multi-Peak Fit ({n_peaks} peaks, {n_points:,} points)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot parameter errors
    param_groups = ["a", "μ", "σ"] * n_peaks
    colors = [
        "red" if pg == "a" else "blue" if pg == "μ" else "green" for pg in param_groups
    ]

    axes[1].bar(range(n_params), rel_errors, color=colors, alpha=0.6)
    axes[1].set_xlabel("Parameter Index")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Parameter Recovery Accuracy")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Legend for parameter types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="Amplitude"),
        Patch(facecolor="blue", alpha=0.6, label="Center"),
        Patch(facecolor="green", alpha=0.6, label="Width"),
    ]
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "performance_optimization_demo"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("COMBINED OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(
        f"\n✓ Sparse Jacobian: {sparsity:.1%} sparsity, {memory_info['reduction_factor']:.1f}x memory reduction"
    )
    print(f"✓ Fit Quality: {np.max(rel_errors):.2%} max error on {n_params} parameters")
    print(f"✓ Performance: {fit_time:.2f}s for {n_points:,} points")
    print("\n🎯 All optimizations working together for maximum performance!")


if not QUICK:
    demo_combined_optimization()


# ======================================================================
# ## 4. Best Practices and Recommendations
#
# ### When to Use Each Feature
#
# #### SparseJacobian
# ✅ **Use when:**
# - Jacobian has >90% sparsity
# - Very large problems (millions of points)
# - Piecewise or multi-component models
# - Memory-constrained
#
# ❌ **Don't use when:**
# - Dense Jacobians (<50% sparsity)
# - Small problems (<10K points)
# - Simple global models
#
# #### AdaptiveHybridStreamingOptimizer
# ✅ **Use when:**
# - Dataset >10GB or doesn't fit in memory
# - Multi-scale parameters need normalization
# - You need covariance estimates at scale
# - Long-running jobs need checkpointing
#
# ❌ **Don't use when:**
# - Data fits in memory
# - Batch methods are fast enough
# - Need exact convergence guarantees
#
# ### Performance Tips
#
# 1. **Profile first**: Identify actual bottlenecks before optimizing
# 2. **Combine techniques**: Use multiple features together
# 3. **Benchmark**: Measure performance improvements
# 4. **Start simple**: Add optimizations incrementally
# 5. **Monitor memory**: Track memory usage during development
#
# ### Typical Workflows
#
# **Small problems (<10K points):**
# ```python
# from nlsq import CurveFit
# cf = CurveFit()
# popt, pcov = cf.curve_fit(func, x, y, p0)
# ```
#
# **Very large sparse problems:**
# ```python
# from nlsq import SparseJacobianComputer, SparseOptimizer
# sparse_comp = SparseJacobianComputer()
# pattern, sparsity = sparse_comp.detect_sparsity_pattern(func, x0, x_sample)
# # Use sparse-aware optimization
# ```
#
# **Huge datasets:**
# ```python
# from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
# config = HybridStreamingConfig(chunk_size=10000)
# optimizer = AdaptiveHybridStreamingOptimizer(config)
# result = optimizer.fit((x, y), func, p0=p0)
# ```
# ======================================================================


# ======================================================================
# ## Summary
#
# This notebook demonstrated NLSQ's advanced performance optimization features:
#
# ✅ **SparseJacobian**
# - Exploit sparsity patterns in Jacobian matrices
# - Memory reduction: 10-100x for sparse problems
# - Enables problems that wouldn't fit in memory
#
# ✅ **AdaptiveHybridStreamingOptimizer**
# - Process huge dataset sizes with bounded memory
# - L-BFGS warmup + streaming Gauss-Newton refinement
# - Suitable for >10GB datasets or long-running pipelines
#
# ### Key Takeaways
#
# 1. **Profile before optimizing**: Identify real bottlenecks
# 2. **Combine techniques**: Multiple features work together
# 3. **Match method to problem**: Choose appropriate optimization for your use case
# 4. **Measure improvements**: Benchmark before and after
# 5. **Start simple**: Add complexity only when needed
#
# These features enable NLSQ to handle problems ranging from small datasets on laptops to massive datasets on clusters, all while maintaining high performance and numerical accuracy.
#
# ---
#
# ## Additional Resources
#
# - **NLSQ Documentation**: [https://nlsq.readthedocs.io](https://nlsq.readthedocs.io)
# - **GitHub Repository**: [https://github.com/imewei/NLSQ](https://github.com/imewei/NLSQ)
# - **Other Examples**:
#   - NLSQ Quickstart
#   - Advanced Features Demo
#   - Large Dataset Demo
#   - 2D Gaussian Demo
#
# *This notebook demonstrates advanced performance optimization features. Requires Python 3.12+ and NLSQ >= 0.1.0.*
# ======================================================================
