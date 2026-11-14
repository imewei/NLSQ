#!/usr/bin/env python3
"""Benchmark script for User Story 1: Large Dataset Fitting Performance.

This script validates the following success criteria:
- SC-001: No JIT recompilation on a bucket-padded final chunk (XLA compile count)
- SC-005: Padding the final chunk removes the recompilation an unpadded final
  chunk triggers (compile counts; wall times are informational)

The key insight: SC-001 and SC-005 are about chunked processing where the
final chunk might have a different size than earlier chunks. With power-of-2
bucket padding, the final chunk should be padded to a bucket that was already
compiled, avoiding recompilation.

Usage:
    python scripts/benchmarks/benchmark_us1.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class _CompileCounter(logging.Handler):
    """Count XLA compilations reported by ``jax.log_compiles``."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Compiling"):
            self.messages.append(record.getMessage())


def _count_compiles(fn: Callable[[], Any], shape_token: str) -> tuple[int, int, float]:
    """Run ``fn``; return (all compiles, compiles mentioning ``shape_token``, seconds).

    Total compile counts include a few small data-dependent kernels from the
    optimizer trajectory, so shape-specific recompilation is counted separately.
    """
    counter = _CompileCounter()
    jax_logger = logging.getLogger("jax")
    jax_logger.addHandler(counter)
    try:
        with jax.log_compiles(True):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
    finally:
        jax_logger.removeHandler(counter)
    n_shape = sum(shape_token in m for m in counter.messages)
    return len(counter.messages), n_shape, elapsed


def test_chunked_processing() -> dict[str, Any]:
    """Test that bucket padding of a final chunk reuses a compiled kernel.

    - SC-001: the final chunk, padded by ``ChunkBuffer.from_array`` (the same
      bucket padding ``DataChunker.create_chunks`` applies), triggers no XLA
      compilation for the unpadded tail shape once its bucket has been warmed.
    - SC-005: the same final chunk left unpadded does compile for the tail
      shape, and the padded run compiles less overall, so padding removes real
      recompilation work. Counts are compared, not cold vs
      warm wall time; times are reported for information only.
    """
    import nlsq
    from nlsq.streaming.large_dataset import (
        CHUNK_BUCKETS,
        ChunkBuffer,
        get_bucket_size,
    )

    rng = np.random.default_rng(42)

    def model(x: jax.Array, a: float, b: float) -> jax.Array:
        return a * jnp.exp(-b * x)

    # Step 1: Warmup with power-of-2 bucket sizes
    print("    Warming up JIT caches for bucket sizes...")
    for bucket in CHUNK_BUCKETS[:4]:  # 1024, 2048, 4096, 8192
        x_warm = np.linspace(0, 10, bucket)
        y_warm = 2.5 * np.exp(-0.3 * x_warm) + rng.normal(0, 0.01, bucket)
        nlsq.curve_fit(model, x_warm, y_warm, p0=[1.0, 1.0])

    # Step 2: final chunk of 127 points, unpadded vs bucket-padded
    print("    Testing A/B: unpadded vs padded final chunk...")
    odd_size = 127  # Simulates a final chunk with leftover data
    x_odd = np.linspace(0, 10, odd_size)
    y_odd = 2.5 * np.exp(-0.3 * x_odd) + rng.normal(0, 0.01, odd_size)

    # B first: padded via the library's own bucket padding. Its bucket (1024) is
    # already compiled, and it must run before A so A cannot warm anything for B.
    x_buf = ChunkBuffer.from_array(x_odd)
    y_buf = ChunkBuffer.from_array(y_odd)
    bucket_size = get_bucket_size(odd_size)
    tail_token = f"[{odd_size}]"  # shape fragment in JAX compile log lines
    padding_applied = bool(
        x_buf.bucket_size == bucket_size
        and len(x_buf.data) == bucket_size
        and x_buf.valid_length == odd_size
        and np.array_equal(x_buf.data[:odd_size], x_odd)
        and int(x_buf.mask.sum()) == odd_size
    )
    padded_compiles, padded_tail_compiles, padded_time = _count_compiles(
        lambda: nlsq.curve_fit(model, x_buf.data, y_buf.data, p0=[1.0, 1.0]),
        tail_token,
    )

    # A: unpadded 127-point shape, never seen before
    unpadded_compiles, unpadded_tail_compiles, unpadded_time = _count_compiles(
        lambda: nlsq.curve_fit(model, x_odd, y_odd, p0=[1.0, 1.0]),
        tail_token,
    )

    # Step 3: SC-001 supporting evidence - warm timings in one bucket are stable
    # (the 1024 bucket was warmed in step 1 with the same shape).
    print("    Testing bucket consistency...")
    bucket_1024_times = []
    for _ in range(5):
        x_test = np.linspace(0, 10, 1024)
        y_test = 2.5 * np.exp(-0.3 * x_test) + rng.normal(0, 0.01, 1024)
        start = time.perf_counter()
        nlsq.curve_fit(model, x_test, y_test, p0=[1.0, 1.0])
        elapsed = time.perf_counter() - start
        bucket_1024_times.append(elapsed)

    cv = np.std(bucket_1024_times) / np.mean(bucket_1024_times)

    sc_001_passed = padding_applied and padded_tail_compiles == 0
    # unpadded_tail_compiles == 0 would mean the counter is blind: fail loudly.
    sc_005_passed = unpadded_tail_compiles > 0 and padded_compiles < unpadded_compiles

    return {
        "odd_size": odd_size,
        "bucket_size": bucket_size,
        "padding_applied": padding_applied,
        "unpadded_compiles": unpadded_compiles,
        "unpadded_tail_shape_compiles": unpadded_tail_compiles,
        "padded_compiles": padded_compiles,
        "padded_tail_shape_compiles": padded_tail_compiles,
        "unpadded_time": float(unpadded_time),
        "padded_time": float(padded_time),
        "bucket_1024_times": bucket_1024_times,
        "bucket_1024_cv": float(cv),
        "sc_001_passed": bool(sc_001_passed),
        "sc_005_passed": bool(sc_005_passed),
    }


def test_bucket_consistency() -> dict[str, Any]:
    """Test that the same bucket sizes reuse compiled kernels.

    After warmup, repeated calls with the same bucket size should have
    consistent timing (no recompilation).
    """
    import nlsq
    from nlsq.streaming.large_dataset import CHUNK_BUCKETS

    def model(x: jax.Array, a: float, b: float) -> jax.Array:
        return a * jnp.exp(-b * x)

    rng = np.random.default_rng(42)
    results = {}

    for bucket in CHUNK_BUCKETS[:4]:
        times = []
        for i in range(5):
            x = np.linspace(0, 10, bucket)
            y = 2.5 * np.exp(-0.3 * x) + rng.normal(0, 0.01, bucket)

            start = time.perf_counter()
            nlsq.curve_fit(model, x, y, p0=[1.0, 1.0])
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # After warmup (run 0), times should be consistent
        post_warmup_times = times[1:]
        variance = np.std(post_warmup_times) / np.mean(post_warmup_times)

        results[bucket] = {
            "times": times,
            "warmup_time": times[0],
            "post_warmup_mean": float(np.mean(post_warmup_times)),
            "post_warmup_std": float(np.std(post_warmup_times)),
            "coefficient_of_variation": float(variance),
            "consistent": bool(variance < 0.3),  # <30% variation is consistent
        }

    all_consistent = all(r["consistent"] for r in results.values())

    return {
        "bucket_results": {str(k): v for k, v in results.items()},
        "all_consistent": bool(all_consistent),
    }


def compare_with_baseline() -> dict[str, Any]:
    """Compare current LargeDatasetFitter performance with baseline."""
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "specs"
        / "002-performance-optimizations"
        / "baseline.json"
    )

    if not baseline_path.exists():
        return {"error": f"Baseline not found at {baseline_path}"}

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_time = baseline.get("large_dataset_fit", {}).get("elapsed_seconds", 0)

    return {
        "baseline_seconds": baseline_time,
        "note": "Baseline comparison is informational only for US1",
    }


def main() -> None:
    """Run User Story 1 benchmarks."""
    print("=" * 60)
    print("User Story 1: Large Dataset Fitting Performance")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jax_backend": jax.default_backend(),
    }

    print("1/3 Testing bucket consistency (JIT cache reuse)...")
    results["bucket_consistency"] = test_bucket_consistency()
    if results["bucket_consistency"]["all_consistent"]:
        print("    Bucket timings consistent - JIT cache working")
    else:
        print("    Warning: Some bucket timings inconsistent")

    print()
    print("2/3 Testing chunked processing (SC-001, SC-005)...")
    results["chunked_processing"] = test_chunked_processing()

    cp = results["chunked_processing"]
    print(
        f"    Unpadded ({cp['odd_size']} pts): {cp['unpadded_compiles']} compiles "
        f"({cp['unpadded_tail_shape_compiles']} for the tail shape), "
        f"{cp['unpadded_time']:.4f}s"
    )
    print(
        f"    Padded ({cp['odd_size']} -> {cp['bucket_size']}): "
        f"{cp['padded_compiles']} compiles "
        f"({cp['padded_tail_shape_compiles']} for the tail shape), {cp['padded_time']:.4f}s"
    )
    print(f"    Bucket padding applied by ChunkBuffer: {cp['padding_applied']}")
    print(f"    Bucket 1024 warm CV: {cp['bucket_1024_cv']:.2f}")

    if cp["sc_001_passed"]:
        print("    SC-001: PASSED - No recompilation on bucket-padded final chunk")
    else:
        print("    SC-001: FAILED - Padded final chunk recompiled or was not padded")

    if cp["sc_005_passed"]:
        print(
            f"    SC-005: PASSED - padding avoided {cp['unpadded_compiles']} "
            "compilations the unpadded chunk triggered"
        )
    else:
        print(
            f"    SC-005: FAILED - unpadded {cp['unpadded_compiles']} vs "
            f"padded {cp['padded_compiles']} compiles"
        )

    print()
    print("3/3 Baseline comparison (informational)...")
    results["baseline"] = compare_with_baseline()
    if "error" not in results["baseline"]:
        print(f"    Original baseline: {results['baseline']['baseline_seconds']:.3f}s")

    # Save results
    output_path = (
        Path(__file__).resolve().parents[2]
        / "specs"
        / "002-performance-optimizations"
        / "us1_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    sc_001 = cp["sc_001_passed"]
    sc_005 = cp["sc_005_passed"]
    print(
        f"SC-001 (No recompilation on final chunk): {'PASSED' if sc_001 else 'FAILED'}"
    )
    print(
        f"SC-005 (Padding avoids unpadded recompiles): "
        f"{'PASSED' if sc_005 else 'FAILED'}"
    )

    if sc_001 and sc_005:
        print()
        print("User Story 1 COMPLETE - All success criteria met!")
        sys.exit(0)
    else:
        print()
        print("User Story 1 INCOMPLETE - Some criteria not met")
        sys.exit(1)


if __name__ == "__main__":
    main()
