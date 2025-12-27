"""Benchmark tests for NLSQ performance optimization.

This package contains benchmark tests for Phase 1-3 optimizations:
- benchmark_streaming.py: Streaming throughput with different padding strategies
- benchmark_checkpoints.py: Sync vs async checkpoint I/O latency
- benchmark_memory.py: Memory manager psutil call frequency vs TTL

Benchmarks use pytest-benchmark for consistent measurement methodology.
"""
