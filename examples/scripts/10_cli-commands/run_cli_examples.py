#!/usr/bin/env python
"""NLSQ CLI Examples - Python Demonstration.

This script demonstrates how to use the NLSQ CLI from Python,
including subprocess calls, result parsing, and batch processing.

Usage:
    python run_cli_examples.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Change to script directory
SCRIPT_DIR = Path(__file__).parent
WORKFLOWS_DIR = SCRIPT_DIR / "workflows"
OUTPUT_DIR = SCRIPT_DIR / "output"


def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_command(cmd, capture_output=True, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        check=check,
        cwd=SCRIPT_DIR,
    )
    return result


def generate_data():
    """Generate sample data files."""
    print_header("Step 1: Generate Sample Data")
    result = run_command([sys.executable, "generate_data.py"])
    print(result.stdout)


def show_system_info():
    """Display NLSQ system information."""
    print_header("Step 2: System Information")
    result = run_command(["nlsq", "info"])
    print(result.stdout)


def run_single_fit(workflow_name):
    """Run a single fitting workflow."""
    print_header(f"Single Fit: {workflow_name}")

    workflow_path = WORKFLOWS_DIR / f"{workflow_name}.yaml"
    result = run_command(["nlsq", "fit", str(workflow_path)])
    print(result.stdout)

    return result


def run_fit_with_stdout(workflow_name):
    """Run a fit and capture JSON output."""
    print_header(f"Fit with stdout: {workflow_name}")

    workflow_path = WORKFLOWS_DIR / f"{workflow_name}.yaml"
    result = run_command(["nlsq", "fit", str(workflow_path), "--stdout"])

    # Parse JSON output
    fit_result = json.loads(result.stdout)

    print(f"Fitted parameters (popt): {fit_result.get('popt', 'N/A')}")
    print(f"Uncertainties: {fit_result.get('uncertainties', 'N/A')}")
    print(f"R-squared: {fit_result.get('r_squared', 'N/A')}")
    print(f"RMSE: {fit_result.get('rmse', 'N/A')}")
    print(f"Success: {fit_result.get('success', 'N/A')}")

    return fit_result


def run_batch_fitting():
    """Run batch fitting on all workflows."""
    print_header("Batch Fitting")

    # Get all workflow files
    workflows = sorted(WORKFLOWS_DIR.glob("*.yaml"))
    workflow_paths = [str(w) for w in workflows]

    summary_file = OUTPUT_DIR / "batch_summary.json"

    cmd = ["nlsq", "batch"] + workflow_paths + ["--summary", str(summary_file)]
    result = run_command(cmd)
    print(result.stdout)

    # Read and display summary
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        print("\nBatch Summary:")
        print(f"  Total workflows: {summary.get('total', 'N/A')}")
        print(f"  Succeeded: {summary.get('succeeded', 'N/A')}")
        print(f"  Failed: {summary.get('failed', 'N/A')}")
        print(f"  Duration: {summary.get('duration_seconds', 'N/A'):.2f} seconds")

    return summary


def compare_results():
    """Compare fitting results across workflows."""
    print_header("Results Comparison")

    results = {}
    result_files = OUTPUT_DIR.glob("*_results.json")

    for result_file in sorted(result_files):
        with open(result_file) as f:
            data = json.load(f)

        name = result_file.stem.replace("_results", "")
        results[name] = {
            "popt": data.get("popt", []),
            "r_squared": data.get("r_squared", None),
            "rmse": data.get("rmse", None),
            "nfev": data.get("nfev", None),
        }

    # Display comparison table
    print(f"{'Workflow':<25} {'R-squared':>12} {'RMSE':>12} {'nfev':>8}")
    print("-" * 60)

    for name, data in results.items():
        r2 = data["r_squared"]
        rmse = data["rmse"]
        nfev = data["nfev"]

        r2_str = f"{r2:.6f}" if r2 is not None else "N/A"
        rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
        nfev_str = str(nfev) if nfev is not None else "N/A"

        print(f"{name:<25} {r2_str:>12} {rmse_str:>12} {nfev_str:>8}")


def programmatic_workflow():
    """Demonstrate programmatic CLI usage patterns."""
    print_header("Programmatic CLI Usage Patterns")

    print("\n1. Check if NLSQ is installed:")
    try:
        result = run_command(["nlsq", "--version"])
        print(f"   NLSQ version: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("   NLSQ not found. Install with: pip install nlsq")
        return

    print("\n2. Get available models programmatically:")
    result = run_command(["nlsq", "info"])
    # Parse builtin models from output
    if "Builtin Models:" in result.stdout:
        lines = result.stdout.split("\n")
        in_models = False
        models = []
        for line in lines:
            if "Builtin Models:" in line:
                in_models = True
                continue
            if in_models:
                if line.strip().startswith("-"):
                    model = line.strip().split("(")[0].replace("-", "").strip()
                    models.append(model)
                elif not line.strip():
                    break
        print(f"   Available models: {', '.join(models[:5])}...")

    print("\n3. Run fit and process results:")
    workflow_path = WORKFLOWS_DIR / "01_radioactive_decay.yaml"
    result = run_command(["nlsq", "fit", str(workflow_path), "--stdout"])
    fit_data = json.loads(result.stdout)

    popt = fit_data["popt"]
    print(f"   Fitted parameters: {popt}")

    # For exponential_decay: a*exp(-b*x) + c
    # popt[0] = initial activity (N0)
    # popt[1] = decay constant (lambda)
    # popt[2] = offset
    if len(popt) >= 2:
        import math

        half_life = math.log(2) / popt[1]
        print(f"   Derived half-life: {half_life:.0f} years")

    print("\n4. Error handling example:")
    try:
        # Try to run a non-existent workflow
        run_command(["nlsq", "fit", "nonexistent.yaml"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"   Expected error caught: {type(e).__name__}")
        print("   (This is normal - workflow file doesn't exist)")


def main():
    """Run all CLI examples."""
    print_header("NLSQ CLI Examples - Python Demonstration")
    print("This script demonstrates using NLSQ CLI from Python.")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Generate data
    generate_data()

    # Step 2: Show system info
    show_system_info()

    # Step 3: Run individual fits
    run_single_fit("01_radioactive_decay")
    run_single_fit("03_reaction_kinetics")

    # Step 4: Run fit with stdout capture
    run_fit_with_stdout("01_radioactive_decay")

    # Step 5: Run batch fitting
    run_batch_fitting()

    # Step 6: Compare results
    compare_results()

    # Step 7: Programmatic patterns
    programmatic_workflow()

    print_header("Complete!")
    print("All CLI examples completed successfully.")
    print(f"\nOutput files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
