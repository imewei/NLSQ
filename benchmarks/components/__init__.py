"""Component-specific benchmark scripts.

Standalone scripts for benchmarking individual NLSQ components:
- cache_benchmark.py: JIT compilation cache performance
- jacobian_benchmark.py: Jacobian mode auto-switching (jacfwd vs jacrev)
- memory_benchmark.py: Memory pool and allocation optimization
- sparse_benchmark.py: Sparse Jacobian detection and solvers
- stability_benchmark.py: Numerical stability guard overhead
- transfer_benchmark.py: Host-device transfer profiling

Run individual benchmarks:
    python benchmarks/components/cache_benchmark.py
    python benchmarks/components/jacobian_benchmark.py --mode=direct
"""
