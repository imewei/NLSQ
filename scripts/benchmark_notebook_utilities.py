#!/usr/bin/env python3
"""
Performance benchmarks for notebook transformation utilities.

Benchmarks various configurations to validate performance claims:
- Sequential vs parallel processing
- Incremental vs full processing
- Different worker counts
- Scaling with notebook count

Usage:
    python scripts/benchmark_notebook_utilities.py
    python scripts/benchmark_notebook_utilities.py --synthetic 100
    python scripts/benchmark_notebook_utilities.py --workers 2 4 8
"""

import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
from notebook_utils.pipeline import TransformationPipeline
from notebook_utils.tracking import ProcessingTracker
from notebook_utils.transformations import (
    IPythonDisplayImportTransformer,
    MatplotlibInlineTransformer,
    PltShowReplacementTransformer,
)


def create_synthetic_notebook(
    name: str, has_matplotlib: bool = True, has_plt_show: bool = True
) -> dict:
    """Create a synthetic notebook for benchmarking.

    Args:
        name: Notebook name
        has_matplotlib: Include matplotlib code
        has_plt_show: Include plt.show() calls

    Returns:
        Notebook dictionary
    """
    cells = [
        {
            "cell_type": "markdown",
            "source": [f"# {name}"],
            "metadata": {},
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import numpy as np"],
        },
    ]

    if has_matplotlib:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import matplotlib.pyplot as plt"],
            }
        )

    if has_plt_show:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots()\n",
                    "ax.plot([1, 2, 3])\n",
                    "plt.show()",
                ],
            }
        )

    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def create_synthetic_dataset(output_dir: Path, num_notebooks: int) -> list[Path]:
    """Create synthetic dataset for benchmarking.

    Args:
        output_dir: Directory to create notebooks in
        num_notebooks: Number of notebooks to create

    Returns:
        List of notebook paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    notebooks = []

    for i in range(num_notebooks):
        notebook_path = output_dir / f"benchmark_notebook_{i:04d}.ipynb"
        notebook = create_synthetic_notebook(
            f"Benchmark Notebook {i}",
            has_matplotlib=True,
            has_plt_show=(i % 2 == 0),  # Half have plt.show()
        )

        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        notebooks.append(notebook_path)

    return notebooks


def benchmark_sequential(
    notebooks: list[Path], pipeline: TransformationPipeline
) -> dict:
    """Benchmark sequential processing.

    Args:
        notebooks: List of notebook paths
        pipeline: Transformation pipeline

    Returns:
        Benchmark results
    """
    start_time = time.perf_counter()

    for notebook_path in notebooks:
        pipeline.run(notebook_path, dry_run=True)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return {
        "mode": "sequential",
        "notebooks": len(notebooks),
        "time_seconds": elapsed,
        "notebooks_per_second": len(notebooks) / elapsed,
    }


def process_notebook(args):
    """Process single notebook (for parallel execution)."""
    notebook_path, transformers = args

    pipeline = TransformationPipeline(transformers)
    return pipeline.run(notebook_path, dry_run=True)


def benchmark_parallel(
    notebooks: list[Path], pipeline: TransformationPipeline, workers: int
) -> dict:
    """Benchmark parallel processing.

    Args:
        notebooks: List of notebook paths
        pipeline: Transformation pipeline
        workers: Number of parallel workers

    Returns:
        Benchmark results
    """
    start_time = time.perf_counter()

    # Prepare arguments for parallel execution
    transformers = pipeline.get_transformers()
    args_list = [(nb_path, transformers) for nb_path in notebooks]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_notebook, args_list))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return {
        "mode": f"parallel_{workers}_workers",
        "notebooks": len(notebooks),
        "workers": workers,
        "time_seconds": elapsed,
        "notebooks_per_second": len(notebooks) / elapsed,
    }


def benchmark_incremental(
    notebooks: list[Path],
    pipeline: TransformationPipeline,
    tracker: ProcessingTracker,
) -> dict:
    """Benchmark incremental processing.

    Args:
        notebooks: List of notebook paths
        pipeline: Transformation pipeline
        tracker: Processing tracker

    Returns:
        Benchmark results for first and second runs
    """
    transform_names = [t.name() for t in pipeline.get_transformers()]

    # First run - processes all notebooks
    start_time = time.perf_counter()
    processed_first = 0

    for notebook_path in notebooks:
        if tracker.needs_processing(notebook_path, transform_names):
            stats = pipeline.run(notebook_path, dry_run=True)
            tracker.mark_processed(notebook_path, transform_names, stats)
            processed_first += 1

    first_elapsed = time.perf_counter() - start_time

    # Second run - should skip all unchanged notebooks
    start_time = time.perf_counter()
    processed_second = 0

    for notebook_path in notebooks:
        if tracker.needs_processing(notebook_path, transform_names):
            stats = pipeline.run(notebook_path, dry_run=True)
            tracker.mark_processed(notebook_path, transform_names, stats)
            processed_second += 1

    second_elapsed = time.perf_counter() - start_time

    return {
        "mode": "incremental",
        "notebooks": len(notebooks),
        "first_run": {
            "time_seconds": first_elapsed,
            "notebooks_processed": processed_first,
            "notebooks_per_second": processed_first / first_elapsed
            if first_elapsed > 0
            else 0,
        },
        "second_run": {
            "time_seconds": second_elapsed,
            "notebooks_processed": processed_second,
            "notebooks_per_second": processed_second / second_elapsed
            if second_elapsed > 0
            else 0,
        },
        "speedup": first_elapsed / second_elapsed
        if second_elapsed > 0
        else float("inf"),
    }


@click.command()
@click.option(
    "--synthetic",
    type=int,
    default=None,
    help="Number of synthetic notebooks to create for benchmarking",
)
@click.option(
    "--dir",
    "notebook_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing real notebooks to benchmark (default: examples/notebooks)",
)
@click.option(
    "--workers",
    multiple=True,
    type=int,
    default=[1, 2, 4, 8],
    help="Worker counts to benchmark for parallel processing",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="benchmark_results.json",
    help="Output file for results",
)
def main(
    synthetic: int | None, notebook_dir: Path | None, workers: tuple[int], output: Path
):
    """Run performance benchmarks for notebook utilities."""
    click.echo("=" * 70)
    click.echo("Notebook Transformation Utilities - Performance Benchmarks")
    click.echo("=" * 70)
    click.echo()

    # Create pipeline
    pipeline = TransformationPipeline(
        [
            MatplotlibInlineTransformer(),
            IPythonDisplayImportTransformer(),
            PltShowReplacementTransformer(),
        ]
    )

    # Determine notebooks to use
    if synthetic:
        click.echo(f"🔧 Creating {synthetic} synthetic notebooks...")
        temp_dir = Path(tempfile.mkdtemp())
        notebooks = create_synthetic_dataset(temp_dir, synthetic)
        click.echo(f"✅ Created in {temp_dir}")
    elif notebook_dir:
        click.echo(f"📂 Using notebooks from {notebook_dir}")
        notebooks = sorted(notebook_dir.rglob("*.ipynb"))
    else:
        click.echo("📂 Using notebooks from examples/notebooks")
        repo_root = Path(__file__).parent.parent
        notebook_dir = repo_root / "examples" / "notebooks"
        notebooks = sorted(notebook_dir.rglob("*.ipynb"))

    if not notebooks:
        click.echo("❌ No notebooks found!", err=True)
        return

    click.echo(f"📊 Benchmarking with {len(notebooks)} notebooks")
    click.echo()

    results = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_notebooks": len(notebooks),
        "benchmarks": [],
    }

    # Benchmark 1: Sequential processing
    click.echo("⏱️  Benchmark 1: Sequential Processing")
    click.echo("-" * 70)
    result = benchmark_sequential(notebooks, pipeline)
    results["benchmarks"].append(result)
    click.echo(f"  Time: {result['time_seconds']:.3f}s")
    click.echo(f"  Throughput: {result['notebooks_per_second']:.2f} notebooks/sec")
    click.echo()

    # Benchmark 2: Parallel processing with different worker counts
    click.echo("⚡ Benchmark 2: Parallel Processing")
    click.echo("-" * 70)
    baseline = result["time_seconds"]

    for worker_count in workers:
        result = benchmark_parallel(notebooks, pipeline, worker_count)
        results["benchmarks"].append(result)

        speedup = baseline / result["time_seconds"]
        efficiency = speedup / worker_count * 100

        click.echo(f"  Workers: {worker_count}")
        click.echo(f"    Time: {result['time_seconds']:.3f}s")
        click.echo(
            f"    Throughput: {result['notebooks_per_second']:.2f} notebooks/sec"
        )
        click.echo(f"    Speedup: {speedup:.2f}× (Efficiency: {efficiency:.1f}%)")

    click.echo()

    # Benchmark 3: Incremental processing
    click.echo("🔄 Benchmark 3: Incremental Processing")
    click.echo("-" * 70)

    # Create fresh tracker for this benchmark
    if synthetic:
        tracker = ProcessingTracker(temp_dir / ".notebook_transforms.json")
    else:
        tracker = ProcessingTracker()

    result = benchmark_incremental(notebooks, pipeline, tracker)
    results["benchmarks"].append(result)

    click.echo("  First run (full processing):")
    click.echo(f"    Time: {result['first_run']['time_seconds']:.3f}s")
    click.echo(f"    Processed: {result['first_run']['notebooks_processed']} notebooks")
    click.echo(
        f"    Throughput: {result['first_run']['notebooks_per_second']:.2f} notebooks/sec"
    )
    click.echo()
    click.echo("  Second run (incremental - unchanged notebooks):")
    click.echo(f"    Time: {result['second_run']['time_seconds']:.3f}s")
    click.echo(
        f"    Processed: {result['second_run']['notebooks_processed']} notebooks"
    )
    if result["second_run"]["notebooks_processed"] > 0:
        click.echo(
            f"    Throughput: {result['second_run']['notebooks_per_second']:.2f} notebooks/sec"
        )
    else:
        click.echo("    Throughput: N/A (no notebooks processed)")
    click.echo()
    click.echo(f"  Speedup: {result['speedup']:.2f}× faster (second run)")
    click.echo()

    # Summary
    click.echo("=" * 70)
    click.echo("📈 Summary")
    click.echo("=" * 70)

    # Find best parallel result
    parallel_results = [
        r for r in results["benchmarks"] if r["mode"].startswith("parallel_")
    ]
    best_parallel = max(parallel_results, key=lambda x: x["notebooks_per_second"])
    best_speedup = baseline / best_parallel["time_seconds"]

    click.echo(f"Sequential:  {baseline:.3f}s baseline")
    click.echo(
        f"Parallel:    {best_parallel['time_seconds']:.3f}s "
        f"({best_speedup:.2f}× speedup with {best_parallel['workers']} workers)"
    )
    click.echo(f"Incremental: {result['speedup']:.2f}× speedup on unchanged notebooks")
    click.echo()

    # Save results
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"💾 Results saved to {output}")

    # Cleanup synthetic dataset
    if synthetic:
        shutil.rmtree(temp_dir)
        click.echo(f"🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
