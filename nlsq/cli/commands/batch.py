"""Batch command handler for NLSQ CLI.

This module provides the 'nlsq batch' command for executing parallel batch
fitting from multiple YAML workflow configuration files.

Example Usage
-------------
From command line:
    $ nlsq batch workflow1.yaml workflow2.yaml
    $ nlsq batch configs/*.yaml --workers 4
    $ nlsq batch configs/*.yaml --summary batch_summary.json

From Python:
    >>> from nlsq.cli.commands.batch import run_batch
    >>> results = run_batch(["w1.yaml", "w2.yaml"])
    >>> results = run_batch(["w1.yaml", "w2.yaml"], summary_file="summary.json")

Note
----
Uses ThreadPoolExecutor by default for JAX compatibility. JAX does not work well
with ProcessPoolExecutor due to fork safety issues with GPU devices.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from nlsq.cli.errors import CLIError, setup_logging


def _run_single_workflow(
    workflow_path: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a single workflow.

    This function is designed to be called by ThreadPoolExecutor.

    Parameters
    ----------
    workflow_path : str
        Path to the workflow YAML configuration file.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    dict
        Result dictionary with status, workflow_path, and result or error.
    """
    from nlsq.cli.commands.fit import run_fit
    from nlsq.cli.errors import CLIError

    try:
        result = run_fit(
            workflow_path=workflow_path,
            verbose=verbose,
        )

        return {
            "status": "success",
            "workflow_path": workflow_path,
            "result": result,
            "error": None,
        }

    except CLIError as e:
        return {
            "status": "failed",
            "workflow_path": workflow_path,
            "result": None,
            "error": {
                "type": type(e).__name__,
                "message": e.message,
                "context": e.context,
                "suggestion": e.suggestion,
            },
        }

    except Exception as e:
        return {
            "status": "failed",
            "workflow_path": workflow_path,
            "result": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "context": {},
                "suggestion": None,
            },
        }


# =============================================================================
# Batch Processing Helper Functions (extracted for complexity reduction)
# =============================================================================


def _validate_batch_inputs(
    workflow_paths: list[str],
    continue_on_error: bool,
) -> list[str]:
    """Validate batch inputs and return list of missing files.

    Parameters
    ----------
    workflow_paths : list[str]
        List of workflow file paths.
    continue_on_error : bool
        If True, return missing files without raising.

    Returns
    -------
    missing_files : list[str]
        List of missing file paths (empty if all exist).

    Raises
    ------
    CLIError
        If no workflow files specified, or if files missing and continue_on_error=False.
    """
    if not workflow_paths:
        raise CLIError(
            "No workflow files specified",
            suggestion="Provide at least one YAML workflow file path",
        )

    missing_files = [path for path in workflow_paths if not Path(path).exists()]

    if missing_files and not continue_on_error:
        raise CLIError(
            f"Workflow files not found: {missing_files}",
            context={"missing_files": missing_files},
            suggestion="Check that all file paths are correct",
        )

    return missing_files


def _calculate_optimal_workers(
    n_workflows: int,
    max_workers: int | None,
) -> int:
    """Calculate optimal number of workers for batch processing.

    Parameters
    ----------
    n_workflows : int
        Number of workflows to process.
    max_workers : int or None
        User-specified max workers. If None, auto-detect.

    Returns
    -------
    int
        Number of workers to use.
    """
    if max_workers is not None:
        return max_workers

    # Auto-detect: use minimum of CPU count and number of workflows
    # For threads, limit to 4 to avoid overwhelming JAX with concurrent compilations
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count, n_workflows)


def _collect_batch_results(
    executor: ThreadPoolExecutor,
    workflow_paths: list[str],
    verbose: bool,
    logger: Any,
) -> list[dict[str, Any]]:
    """Collect results from submitted workflow futures.

    Parameters
    ----------
    executor : ThreadPoolExecutor
        The executor running workflows.
    workflow_paths : list[str]
        List of workflow paths.
    verbose : bool
        Enable verbose logging.
    logger : Logger
        Logger instance.

    Returns
    -------
    list[dict]
        List of result dictionaries.
    """
    results: list[dict[str, Any]] = []

    # Submit all workflows
    futures = {
        executor.submit(_run_single_workflow, path, verbose): path
        for path in workflow_paths
    }

    # Collect results as they complete
    for future in as_completed(futures):
        workflow_path = futures[future]

        try:
            result = future.result()
            results.append(result)

            # Log status
            if result["status"] == "success":
                if verbose:
                    logger.info(f"SUCCESS: {workflow_path}")
            else:
                error_info = result.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                logger.warning(f"FAILED: {workflow_path} - {error_msg}")

        except Exception as e:
            # Handle executor-level errors
            results.append(
                {
                    "status": "failed",
                    "workflow_path": workflow_path,
                    "result": None,
                    "error": {
                        "type": "ExecutorError",
                        "message": str(e),
                        "context": {},
                        "suggestion": None,
                    },
                }
            )
            logger.warning(f"FAILED: {workflow_path} - Executor error: {e}")

    return results


def _build_batch_summary(
    results: list[dict[str, Any]],
    n_workflows: int,
    max_workers: int,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    """Build summary dictionary from batch results.

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries.
    n_workflows : int
        Total number of workflows.
    max_workers : int
        Number of workers used.
    start_time : datetime
        Batch start time.
    end_time : datetime
        Batch end time.

    Returns
    -------
    dict
        Summary dictionary with stats and details.
    """
    duration = (end_time - start_time).total_seconds()
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "total": n_workflows,
        "succeeded": succeeded,
        "failed": failed,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "max_workers": max_workers,
        "failures": [
            {
                "workflow_path": r["workflow_path"],
                "error": r.get("error"),
            }
            for r in results
            if r["status"] == "failed"
        ],
        "successes": [
            {
                "workflow_path": r["workflow_path"],
            }
            for r in results
            if r["status"] == "success"
        ],
    }


def _write_batch_summary(
    summary: dict[str, Any],
    summary_file: str,
    verbose: bool,
    logger: Any,
) -> None:
    """Write batch summary to file.

    Parameters
    ----------
    summary : dict
        Summary dictionary.
    summary_file : str
        Output file path.
    verbose : bool
        Enable verbose logging.
    logger : Logger
        Logger instance.
    """
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        logger.info(f"Summary written to: {summary_file}")


def _print_batch_summary(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Print batch summary to console.

    Parameters
    ----------
    summary : dict
        Summary dictionary.
    results : list[dict]
        List of result dictionaries.
    """
    print("\nBatch Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  Succeeded: {summary['succeeded']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Duration: {summary['duration_seconds']:.2f}s")

    if summary["failed"] > 0:
        print("\nFailed workflows:")
        for r in results:
            if r["status"] == "failed":
                error_info = r.get("error", {})
                print(
                    f"  - {r['workflow_path']}: {error_info.get('message', 'Unknown error')}"
                )


def run_batch(
    workflow_paths: list[str],
    summary_file: str | None = None,
    max_workers: int | None = None,
    continue_on_error: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Execute multiple workflows in parallel using threads.

    Uses ThreadPoolExecutor for JAX compatibility. JAX does not work well
    with ProcessPoolExecutor due to fork safety issues with GPU devices.

    Parameters
    ----------
    workflow_paths : list[str]
        List of paths to workflow YAML configuration files.
    summary_file : str, optional
        Path for aggregate summary file. If None, no summary is written.
    max_workers : int, optional
        Maximum number of parallel workers. If None, auto-detects based on
        CPU count and number of workflows.
    continue_on_error : bool
        If True, continue processing on individual workflow failures.
        Failures are collected and reported at the end.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    list[dict]
        List of result dictionaries, one per workflow.
        Each contains: status, workflow_path, result, error.
    """
    # Set up logging
    logger = setup_logging(
        verbosity=2 if verbose else 1,
        console=True,
    )

    # Step 1: Validate inputs
    _validate_batch_inputs(workflow_paths, continue_on_error)

    # Step 2: Calculate optimal workers
    n_workflows = len(workflow_paths)
    num_workers = _calculate_optimal_workers(n_workflows, max_workers)

    if verbose:
        logger.info(f"Starting batch processing of {n_workflows} workflows")
        logger.info(f"Using {num_workers} parallel workers (threads)")

    # Step 3: Execute workflows and collect results
    start_time = datetime.now()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = _collect_batch_results(executor, workflow_paths, verbose, logger)
    end_time = datetime.now()

    # Step 4: Build summary
    summary = _build_batch_summary(
        results=results,
        n_workflows=n_workflows,
        max_workers=num_workers,
        start_time=start_time,
        end_time=end_time,
    )

    # Step 5: Write summary file if requested
    if summary_file is not None:
        _write_batch_summary(summary, summary_file, verbose, logger)

    # Step 6: Print summary to console
    _print_batch_summary(summary, results)

    return results
