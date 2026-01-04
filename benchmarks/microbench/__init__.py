"""Micro-benchmarks for specific operations.

pytest-benchmark based tests for measuring specific operations:
- test_import.py: Import time (SC-001)
- test_condition.py: Condition number estimation (SC-010)
- test_sparse.py: Sparse Jacobian construction (SC-002)
- test_optimizations.py: Overall optimization validation

Run with pytest-benchmark:
    pytest benchmarks/microbench/ --benchmark-only
    pytest benchmarks/microbench/ --benchmark-json=results.json
"""
