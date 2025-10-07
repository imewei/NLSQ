#!/usr/bin/env python3
"""Comprehensive performance benchmark for NLSQ optimizations.

This script benchmarks Sprint 2 performance improvements including:
- Function signature caching
- Memory pool optimization
- NumPy/JAX array handling

Compares performance before and after optimizations.
"""

import time
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple
import sys

try:
    from nlsq import curve_fit
    from nlsq.func_cache import get_function_cache, cached_jit
    from nlsq.trf_memory_pool import get_trf_memory_pool
except ImportError:
    print("Error: NLSQ not installed. Run 'pip install -e .' first.")
    sys.exit(1)


def exponential_model(x, a, b, c):
    """Exponential model for benchmarking."""
    return a * jnp.exp(-b * x) + c


def gaussian_model(x, amp, mu, sigma):
    """Gaussian model for benchmarking."""
    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def polynomial_model(x, *coeffs):
    """Polynomial model for benchmarking."""
    result = jnp.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


class PerformanceBenchmark:
    """Performance benchmark suite for NLSQ."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
        
    def benchmark_function_caching(self, n_runs: int = 50) -> Dict[str, float]:
        """Benchmark function signature caching performance.
        
        Parameters
        ----------
        n_runs : int
            Number of benchmark runs
            
        Returns
        -------
        metrics : Dict[str, float]
            Timing metrics (first_run, cached_runs, speedup)
        """
        print(f"\nBenchmarking function caching ({n_runs} runs)...")
        
        # Generate test data
        np.random.seed(42)
        x = np.linspace(0, 10, 1000)
        y_true = 2.0 * np.exp(-0.5 * x) + 0.3
        y = y_true + 0.05 * np.random.randn(len(x))
        p0 = [2.0, 0.5, 0.3]
        
        # First run (includes JIT compilation)
        start = time.perf_counter()
        popt, _ = curve_fit(exponential_model, x, y, p0=p0)
        first_run_time = time.perf_counter() - start
        
        # Cached runs (should be faster)
        cached_times = []
        for i in range(n_runs):
            # Generate slightly different data (same shape/dtype)
            y_noisy = y_true + 0.05 * np.random.randn(len(x))
            
            start = time.perf_counter()
            popt, _ = curve_fit(exponential_model, x, y_noisy, p0=p0)
            cached_times.append(time.perf_counter() - start)
        
        avg_cached = np.mean(cached_times)
        speedup = first_run_time / avg_cached
        
        # Get cache statistics
        cache = get_function_cache()
        stats = cache.get_stats()
        
        print(f"  First run (JIT):    {first_run_time*1000:.2f} ms")
        print(f"  Avg cached run:     {avg_cached*1000:.2f} ms")
        print(f"  Speedup:            {speedup:.2f}x")
        print(f"  Cache hit rate:     {stats['hit_rate']*100:.1f}%")
        
        return {
            'first_run_ms': first_run_time * 1000,
            'avg_cached_ms': avg_cached * 1000,
            'speedup': speedup,
            'cache_hit_rate': stats['hit_rate'],
        }
    
    def benchmark_dataset_sizes(self) -> Dict[str, List[float]]:
        """Benchmark performance across different dataset sizes.
        
        Returns
        -------
        results : Dict[str, List[float]]
            Timing results for each size
        """
        print("\nBenchmarking dataset sizes...")
        
        sizes = [100, 500, 1000, 5000, 10000]
        results = {'sizes': sizes, 'times_ms': []}
        
        np.random.seed(42)
        
        for size in sizes:
            x = np.linspace(0, 10, size)
            y_true = 2.0 * np.exp(-0.5 * x) + 0.3
            y = y_true + 0.05 * np.random.randn(size)
            p0 = [2.0, 0.5, 0.3]
            
            # Warmup
            curve_fit(exponential_model, x, y, p0=p0)
            
            # Benchmark
            start = time.perf_counter()
            popt, _ = curve_fit(exponential_model, x, y, p0=p0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            results['times_ms'].append(elapsed_ms)
            print(f"  Size {size:5d}: {elapsed_ms:7.2f} ms")
        
        return results
    
    def benchmark_memory_pool(self, n_runs: int = 20) -> Dict[str, float]:
        """Benchmark memory pool performance.
        
        Parameters
        ----------
        n_runs : int
            Number of benchmark runs
            
        Returns
        -------
        metrics : Dict[str, float]
            Memory pool metrics
        """
        print(f"\nBenchmarking memory pool ({n_runs} runs)...")
        
        pool = get_trf_memory_pool()
        stats_before = pool.get_stats()
        
        # Run fits
        np.random.seed(42)
        x = np.linspace(0, 10, 1000)
        
        times = []
        for i in range(n_runs):
            y_true = 5.0 * np.exp(-((x - 5.0) ** 2) / (2 * 2.0**2))
            y = y_true + 0.1 * np.random.randn(len(x))
            p0 = [5.0, 5.0, 2.0]
            
            start = time.perf_counter()
            popt, _ = curve_fit(gaussian_model, x, y, p0=p0)
            times.append(time.perf_counter() - start)
        
        stats_after = pool.get_stats()
        avg_time = np.mean(times) * 1000
        
        print(f"  Avg time per fit:   {avg_time:.2f} ms")
        print(f"  Pool size:          {stats_after['num_shapes']} shapes")
        print(f"  Total memory:       {stats_after['total_memory_mb']:.2f} MB")
        print(f"  Pool enabled:       {stats_after['enabled']}")
        
        return {
            'avg_time_ms': avg_time,
            'pool_size': stats_after['num_shapes'],
            'memory_mb': stats_after['total_memory_mb'],
        }
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("NLSQ Performance Benchmark Suite")
        print("Sprint 2: Performance Optimization")
        print("=" * 60)
        
        # Function caching benchmark
        cache_results = self.benchmark_function_caching(n_runs=50)
        
        # Dataset size scaling
        size_results = self.benchmark_dataset_sizes()
        
        # Memory pool
        pool_results = self.benchmark_memory_pool(n_runs=20)
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Function Caching Speedup:    {cache_results['speedup']:.2f}x")
        print(f"Cache Hit Rate:              {cache_results['cache_hit_rate']*100:.1f}%")
        print(f"Memory Pool Efficiency:      {pool_results['memory_mb']:.2f} MB pooled")
        print()
        print("Scaling (size → time):")
        for size, time_ms in zip(size_results['sizes'], size_results['times_ms']):
            print(f"  {size:5d} points → {time_ms:7.2f} ms")
        
        return {
            'function_caching': cache_results,
            'dataset_scaling': size_results,
            'memory_pool': pool_results,
        }


if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n✅ Benchmark complete!")
    print("\nTo save results: python benchmark_sprint2.py > benchmark_results.txt")
