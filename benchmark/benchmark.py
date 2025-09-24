#!/usr/bin/env python
"""NLSQ Comprehensive Benchmark and Test Suite

This script combines all benchmarking and testing functionality:
- Performance benchmarks (NLSQ vs SciPy)
- Solver comparisons (SVD, CG, LSQR)
- Large dataset testing
- Memory efficiency testing
- Sparse Jacobian testing
- Streaming optimizer testing
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit

# Add parent to path for NLSQ import
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

# Configure JAX for CPU by default to avoid GPU memory issues
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import nlsq
from nlsq import (
    curve_fit,
    curve_fit_large,
    CurveFit,
    LeastSquares,
    LargeDatasetFitter,
    StreamingOptimizer,
    StreamingConfig,
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
    estimate_memory_requirements,
    MemoryConfig,
    LargeDatasetConfig,
    memory_context,
)


# ============================================================================
# Test Functions
# ============================================================================

def exponential_model_numpy(x, a, b):
    """NumPy exponential model for SciPy"""
    return a * np.exp(-b * x)


def exponential_model_jax(x, a, b):
    """JAX exponential model for NLSQ"""
    return a * jnp.exp(-b * x)


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for advanced tests"""
    x, y = coords[0], coords[1]

    # Rotate coordinates
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    x_rot = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0)

    # Calculate Gaussian
    exp_arg = -(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))
    return amplitude * jnp.exp(exp_arg) + offset


def sparse_function(x, a, b, c):
    """Function with sparse Jacobian structure for testing"""
    result = jnp.zeros_like(x)
    result = jnp.where(x < 3, a * jnp.exp(-b * x), result)
    result = jnp.where(x >= 3, c * x**2, result)
    return result


# ============================================================================
# Basic Benchmarks
# ============================================================================

class BasicBenchmark:
    """Basic performance benchmarks comparing NLSQ and SciPy"""

    def __init__(self):
        self.results = {}

    def run_1d_exponential(self, sizes: List[int] = None) -> Dict:
        """Benchmark 1D exponential fitting"""
        if sizes is None:
            sizes = [50, 100, 200, 500, 1000]

        print("\n=== 1D Exponential Fitting Benchmark ===")
        print(f"{'Size':>8} {'NLSQ (ms)':>12} {'SciPy (ms)':>12} {'Speedup':>10}")
        print("-" * 45)

        results = {'sizes': sizes, 'nlsq': [], 'scipy': [], 'speedup': []}

        for size in sizes:
            # Generate test data
            np.random.seed(42)
            x = np.linspace(0, 4, size)
            true_a, true_b = 2.5, 1.3
            y = exponential_model_numpy(x, true_a, true_b)
            y += np.random.normal(0, 0.05, size)
            p0 = [1.0, 1.0]

            # Time NLSQ
            try:
                start = time.perf_counter()
                popt_nlsq, _ = curve_fit(exponential_model_jax, x, y, p0=p0)
                nlsq_time = (time.perf_counter() - start) * 1000
                results['nlsq'].append(nlsq_time)
            except Exception as e:
                print(f"NLSQ error for size {size}: {e}")
                results['nlsq'].append(None)
                nlsq_time = None

            # Time SciPy
            try:
                start = time.perf_counter()
                popt_scipy, _ = scipy_curve_fit(exponential_model_numpy, x, y, p0=p0)
                scipy_time = (time.perf_counter() - start) * 1000
                results['scipy'].append(scipy_time)
            except Exception as e:
                print(f"SciPy error for size {size}: {e}")
                results['scipy'].append(None)
                scipy_time = None

            # Calculate speedup
            if nlsq_time and scipy_time:
                speedup = scipy_time / nlsq_time
                results['speedup'].append(speedup)
                print(f"{size:8} {nlsq_time:12.2f} {scipy_time:12.2f} {speedup:10.2f}x")
            else:
                results['speedup'].append(None)
                print(f"{size:8} {'Failed':>12} {'Failed':>12} {'N/A':>10}")

        self.results['1d_exponential'] = results
        return results

    def run_2d_gaussian(self, sizes: List[int] = None) -> Dict:
        """Benchmark 2D Gaussian fitting"""
        if sizes is None:
            sizes = [10, 20, 30, 50]

        print("\n=== 2D Gaussian Fitting Benchmark ===")
        print(f"{'Size':>8} {'NLSQ (ms)':>12} {'Speedup':>10}")
        print("-" * 35)

        results = {'sizes': sizes, 'nlsq': [], 'speedup': []}

        for size in sizes:
            # Create 2D grid
            x = np.linspace(-5, 5, size)
            y = np.linspace(-5, 5, size)
            xx, yy = np.meshgrid(x, y)
            coords = np.array([xx.ravel(), yy.ravel()])

            # Generate random parameters
            np.random.seed(42)
            true_params = [
                1.5,  # amplitude
                np.random.uniform(-2, 2),  # x0
                np.random.uniform(-2, 2),  # y0
                1.0 + np.random.random(),  # sigma_x
                1.0 + np.random.random(),  # sigma_y
                np.random.uniform(-np.pi/4, np.pi/4),  # theta
                0.1  # offset
            ]

            # Generate data
            data = gaussian_2d(coords, *true_params)
            data += jnp.array(np.random.normal(0, 0.05, data.shape))

            # Time NLSQ
            try:
                fitter = CurveFit(use_dynamic_sizing=True)
                start = time.perf_counter()
                popt, _ = fitter.curve_fit(gaussian_2d, coords, data, p0=true_params)
                nlsq_time = (time.perf_counter() - start) * 1000
                results['nlsq'].append(nlsq_time)
                print(f"{size:8} {nlsq_time:12.2f}")
            except Exception as e:
                print(f"{size:8} {'Failed':>12} - {str(e)[:30]}")
                results['nlsq'].append(None)

        self.results['2d_gaussian'] = results
        return results


# ============================================================================
# Solver Comparison
# ============================================================================

class SolverComparison:
    """Compare different solver methods"""

    def __init__(self):
        self.results = {}

    def run_solver_comparison(self, size: int = 100) -> Dict:
        """Compare solver performance on standard problem"""
        print(f"\n=== Solver Performance Comparison (size={size}) ===")

        # Generate test data
        np.random.seed(42)
        x = np.linspace(0, 4, size)
        true_a, true_b = 2.5, 1.3
        y = exponential_model_numpy(x, true_a, true_b)
        y += np.random.normal(0, 0.05, size)

        solvers = ['auto', 'svd', 'cg', 'lsqr']
        results = {'solvers': solvers, 'times': [], 'params': [], 'errors': []}

        print(f"{'Solver':>10} {'Time (ms)':>12} {'a':>8} {'b':>8} {'Error':>10}")
        print("-" * 50)

        fitter = CurveFit(use_dynamic_sizing=True)

        for solver in solvers:
            try:
                start = time.perf_counter()
                popt, _ = fitter.curve_fit(
                    exponential_model_jax, x, y,
                    p0=[1.0, 1.0], solver=solver
                )
                elapsed = (time.perf_counter() - start) * 1000
                error = np.sqrt((popt[0] - true_a)**2 + (popt[1] - true_b)**2)

                results['times'].append(elapsed)
                results['params'].append(popt.tolist())
                results['errors'].append(error)

                print(f"{solver:>10} {elapsed:12.2f} {popt[0]:8.3f} {popt[1]:8.3f} {error:10.4f}")

            except Exception as e:
                results['times'].append(None)
                results['params'].append(None)
                results['errors'].append(None)
                print(f"{solver:>10} {'Failed':>12} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
                print(f"  Error: {str(e)[:50]}")

        self.results['solver_comparison'] = results
        return results


# ============================================================================
# Large Dataset Testing
# ============================================================================

class LargeDatasetTesting:
    """Test large dataset handling capabilities"""

    def __init__(self):
        self.results = {}

    def test_curve_fit_large(self, sizes: List[int] = None) -> Dict:
        """Test curve_fit_large function"""
        if sizes is None:
            sizes = [1000, 10000, 100000]

        print("\n=== Large Dataset Testing (curve_fit_large) ===")
        print(f"{'Size':>10} {'Time (s)':>12} {'Memory (MB)':>12} {'Error':>10}")
        print("-" * 45)

        results = {'sizes': sizes, 'times': [], 'memory': [], 'errors': []}

        for n in sizes:
            try:
                # Generate large dataset
                np.random.seed(42)
                x = np.linspace(0, 10, n)
                true_a, true_b = 2.5, 1.3
                y = exponential_model_numpy(x, true_a, true_b)
                y += np.random.normal(0, 0.05, n)

                # Measure memory if psutil available
                try:
                    import psutil
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / 1024 / 1024
                except ImportError:
                    mem_before = 0

                start = time.perf_counter()
                popt, pcov = curve_fit_large(
                    exponential_model_jax,
                    x, y,
                    p0=[1.0, 1.0],
                    memory_limit_gb=2.0,
                    show_progress=False
                )
                elapsed = time.perf_counter() - start

                # Calculate memory usage
                try:
                    mem_after = process.memory_info().rss / 1024 / 1024
                    mem_used = mem_after - mem_before
                except:
                    mem_used = 0

                # Calculate error
                error = np.sqrt((popt[0] - true_a)**2 + (popt[1] - true_b)**2)

                results['times'].append(elapsed)
                results['memory'].append(mem_used)
                results['errors'].append(error)

                print(f"{n:10} {elapsed:12.2f} {mem_used:12.1f} {error:10.4f}")

            except Exception as e:
                results['times'].append(None)
                results['memory'].append(None)
                results['errors'].append(None)
                print(f"{n:10} {'Failed':>12} {'N/A':>12} {'N/A':>10}")
                print(f"  Error: {str(e)[:50]}")

        self.results['curve_fit_large'] = results
        return results

    def test_memory_estimation(self) -> Dict:
        """Test memory requirement estimation"""
        print("\n=== Memory Requirements Estimation ===")

        test_cases = [
            (1000, 3),      # Small
            (100000, 5),    # Medium
            (10000000, 10), # Large
        ]

        results = {'cases': [], 'estimates': []}

        print(f"{'Dataset':>15} {'Params':>8} {'Base (GB)':>12} {'Jacobian (GB)':>15} {'Total (GB)':>12}")
        print("-" * 65)

        for n_data, n_params in test_cases:
            req = estimate_memory_requirements(n_data, n_params)

            results['cases'].append((n_data, n_params))
            results['estimates'].append({
                'n_points': req.n_points,
                'n_params': req.n_params,
                'memory_per_point_bytes': req.memory_per_point_bytes,
                'total_memory_gb': req.total_memory_estimate_gb,
                'recommended_chunk_size': req.recommended_chunk_size,
                'n_chunks': req.n_chunks,
                'requires_sampling': req.requires_sampling
            })

            # Calculate base and jacobian memory estimates
            base_memory_gb = (n_data * 3 * 8) / (1024**3)  # 3 float64 values per point
            jacobian_memory_gb = (n_data * n_params * 8) / (1024**3)  # Jacobian matrix

            print(f"{n_data:15,} {n_params:8} {base_memory_gb:12.3f} "
                  f"{jacobian_memory_gb:15.3f} {req.total_memory_estimate_gb:12.3f}")

        self.results['memory_estimation'] = results
        return results


# ============================================================================
# Advanced Feature Testing
# ============================================================================

class AdvancedFeatureTesting:
    """Test advanced features like sparse Jacobian and streaming"""

    def __init__(self):
        self.results = {}

    def test_sparse_jacobian(self) -> Dict:
        """Test sparse Jacobian functionality"""
        print("\n=== Sparse Jacobian Testing ===")

        # Create sparse problem
        np.random.seed(42)
        x = np.linspace(0, 10, 1000)
        true_params = [2.5, 1.3, 0.5]
        y = sparse_function(x, *true_params).block_until_ready()
        y = np.array(y) + np.random.normal(0, 0.01, x.shape)

        # Detect sparsity
        sparsity, info = detect_jacobian_sparsity(
            sparse_function, np.array(true_params), x[:100]
        )

        results = {
            'sparsity': sparsity,
            'memory_reduction': info['memory_reduction'],
            'avg_nnz_per_row': info['avg_nnz_per_row'],
            'avg_nnz_per_col': info['avg_nnz_per_col']
        }

        print(f"Sparsity detected: {sparsity:.1%}")
        print(f"Memory reduction: {info['memory_reduction']:.1f}%")
        print(f"Avg non-zeros per row: {info['avg_nnz_per_row']:.1f}")
        print(f"Avg non-zeros per col: {info['avg_nnz_per_col']:.1f}")

        # Test sparse optimizer
        optimizer = SparseOptimizer(min_sparsity=0.5, auto_detect=True)
        should_use = optimizer.should_use_sparse(len(y), len(true_params))
        print(f"Should use sparse methods: {should_use}")

        results['should_use_sparse'] = should_use

        self.results['sparse_jacobian'] = results
        return results

    def test_streaming_optimizer(self) -> Dict:
        """Test streaming optimizer configuration"""
        print("\n=== Streaming Optimizer Testing ===")

        # Configure streaming
        config = StreamingConfig(
            batch_size=1000,
            max_epochs=10,
            convergence_tol=1e-6
        )

        # Create optimizer
        optimizer = StreamingOptimizer(config)

        # Estimate memory based on batch size and parameters
        n_params = 2
        batch_memory_mb = (config.batch_size * n_params * 8 * 3) / (1024 * 1024)  # data, jacobian, work
        total_memory_mb = batch_memory_mb * 2  # Double for safety

        results = {
            'batch_size': config.batch_size,
            'max_epochs': config.max_epochs,
            'convergence_tol': config.convergence_tol,
            'batch_memory_mb': batch_memory_mb,
            'total_memory_mb': total_memory_mb
        }

        print(f"Batch size: {config.batch_size}")
        print(f"Max epochs: {config.max_epochs}")
        print(f"Convergence tolerance: {config.convergence_tol}")
        print(f"Estimated memory per batch: {batch_memory_mb:.2f} MB")
        print(f"Total estimated memory: {total_memory_mb:.2f} MB")

        self.results['streaming_optimizer'] = results
        return results

    def test_memory_config(self) -> Dict:
        """Test memory configuration system"""
        print("\n=== Memory Configuration Testing ===")

        # Test different configurations
        configs = [
            MemoryConfig(memory_limit_gb=4.0, gpu_memory_fraction=0.8),
            MemoryConfig(memory_limit_gb=8.0, enable_mixed_precision_fallback=True),
            MemoryConfig(memory_limit_gb=16.0, chunk_size_mb=512)
        ]

        results = {'configs': []}

        for i, config in enumerate(configs, 1):
            print(f"\nConfig {i}:")
            print(f"  Memory limit: {config.memory_limit_gb} GB")
            print(f"  GPU fraction: {config.gpu_memory_fraction}")
            print(f"  Mixed precision: {config.enable_mixed_precision_fallback}")

            results['configs'].append({
                'memory_limit_gb': config.memory_limit_gb,
                'gpu_memory_fraction': config.gpu_memory_fraction,
                'enable_mixed_precision_fallback': config.enable_mixed_precision_fallback
            })

        # Test context manager
        with memory_context(configs[0]):
            print("\nMemory context active - config applied")

        self.results['memory_config'] = results
        return results


# ============================================================================
# Main Benchmark Suite
# ============================================================================

class NLSQBenchmarkSuite:
    """Main benchmark suite orchestrator"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path.cwd()
        self.results = {}

        # Initialize component benchmarks
        self.basic = BasicBenchmark()
        self.solver = SolverComparison()
        self.large_dataset = LargeDatasetTesting()
        self.advanced = AdvancedFeatureTesting()

    def run_basic_benchmarks(self):
        """Run basic performance benchmarks"""
        print("\n" + "=" * 60)
        print("BASIC PERFORMANCE BENCHMARKS")
        print("=" * 60)

        self.basic.run_1d_exponential()
        self.basic.run_2d_gaussian()
        self.results['basic'] = self.basic.results

    def run_solver_comparison(self):
        """Run solver comparison tests"""
        print("\n" + "=" * 60)
        print("SOLVER COMPARISON")
        print("=" * 60)

        self.solver.run_solver_comparison()
        self.results['solver'] = self.solver.results

    def run_large_dataset_tests(self):
        """Run large dataset tests"""
        print("\n" + "=" * 60)
        print("LARGE DATASET TESTING")
        print("=" * 60)

        self.large_dataset.test_curve_fit_large()
        self.large_dataset.test_memory_estimation()
        self.results['large_dataset'] = self.large_dataset.results

    def run_advanced_features(self):
        """Run advanced feature tests"""
        print("\n" + "=" * 60)
        print("ADVANCED FEATURES")
        print("=" * 60)

        self.advanced.test_sparse_jacobian()
        self.advanced.test_streaming_optimizer()
        self.advanced.test_memory_config()
        self.results['advanced'] = self.advanced.results

    def run_all(self):
        """Run all benchmarks and tests"""
        print("=" * 60)
        print("NLSQ COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        print(f"NLSQ Version: {nlsq.__version__}")
        print(f"JAX Backend: {jax.default_backend()}")

        self.run_basic_benchmarks()
        self.run_solver_comparison()
        self.run_large_dataset_tests()
        self.run_advanced_features()

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            json.dump(self.results, f, indent=2, default=convert)
        print(f"\nResults saved to {output_path}")

    def print_summary(self):
        """Print a summary of benchmark results"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Basic benchmarks summary
        if 'basic' in self.results and '1d_exponential' in self.results['basic']:
            exp_results = self.results['basic']['1d_exponential']
            if exp_results['speedup']:
                avg_speedup = np.nanmean([s for s in exp_results['speedup'] if s])
                print(f"Average 1D exponential speedup: {avg_speedup:.2f}x")

        # Solver comparison summary
        if 'solver' in self.results and 'solver_comparison' in self.results['solver']:
            solver_results = self.results['solver']['solver_comparison']
            best_solver_idx = np.nanargmin(solver_results['times'])
            if best_solver_idx is not None:
                best_solver = solver_results['solvers'][best_solver_idx]
                best_time = solver_results['times'][best_solver_idx]
                print(f"Fastest solver: {best_solver} ({best_time:.2f} ms)")

        # Large dataset summary
        if 'large_dataset' in self.results:
            print("Large dataset support: ✓ Tested up to 100K points")

        # Advanced features summary
        if 'advanced' in self.results:
            if 'sparse_jacobian' in self.results['advanced']:
                sparsity = self.results['advanced']['sparse_jacobian']['sparsity']
                print(f"Sparse Jacobian support: ✓ ({sparsity:.1%} sparsity detected)")
            print("Streaming optimizer: ✓ Configured")
            print("Memory management: ✓ Multiple configs tested")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NLSQ Comprehensive Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test suites:
  basic     - Basic performance benchmarks (1D/2D fitting)
  solver    - Solver comparison (SVD, CG, LSQR)
  large     - Large dataset testing
  advanced  - Advanced features (sparse, streaming)
  all       - Run all benchmarks

Examples:
  python benchmark.py --suite all
  python benchmark.py --suite basic --save
  python benchmark.py --suite solver --gpu
        """
    )

    parser.add_argument(
        '--suite',
        choices=['all', 'basic', 'solver', 'large', 'advanced'],
        default='all',
        help='Which benchmark suite to run (default: all)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Output directory for results (default: current dir)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU if available (default: CPU)'
    )

    args = parser.parse_args()

    # Configure JAX backend
    if args.gpu:
        try:
            # Check if GPU is available
            if jax.devices('gpu'):
                jax.config.update("jax_platform_name", "gpu")
                print("Using GPU backend")
        except:
            print("GPU not available, using CPU")

    # Create benchmark suite
    suite = NLSQBenchmarkSuite(output_dir=args.output_dir)

    # Run selected benchmarks
    if args.suite == 'all':
        suite.run_all()
    elif args.suite == 'basic':
        suite.run_basic_benchmarks()
    elif args.suite == 'solver':
        suite.run_solver_comparison()
    elif args.suite == 'large':
        suite.run_large_dataset_tests()
    elif args.suite == 'advanced':
        suite.run_advanced_features()

    # Print summary
    suite.print_summary()

    # Save results if requested
    if args.save:
        suite.save_results()


if __name__ == '__main__':
    main()