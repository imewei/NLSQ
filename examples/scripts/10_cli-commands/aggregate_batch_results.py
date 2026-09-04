"""
Aggregate `nlsq batch` Results Into One Parameter Table

`nlsq batch --summary` writes counts and failures, but not fitted parameters -
those go to each workflow's own `export.results_file`. This script collects
those per-workflow JSON files into a single table (and optionally a CSV), which
is what you usually want after a batch run.

Run this example:
    cd examples/scripts/10_cli-commands
    nlsq batch workflows/batch_example/*.yaml --summary output/batch_summary.json
    python aggregate_batch_results.py

    # Other output directory, plus a machine-readable copy
    python aggregate_batch_results.py output --csv output/parameters.csv

If no result files are found the script prints how to produce them and exits 0,
so it is safe to run before the batch.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=Path(__file__).parent / "output",
        type=Path,
        help="Directory holding batch_results_*.json (default: ./output)",
    )
    parser.add_argument(
        "--pattern",
        default="batch_results_*.json",
        help="Glob for per-workflow result files",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Batch summary JSON; failures from it are reported alongside the table",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write the table to this CSV file",
    )
    return parser.parse_args()


def parameter_names(model_id: str, n_params: int) -> list[str]:
    """Recover parameter names from the model signature, or fall back to p0..pN.

    The result JSON stores popt as a bare list, so names come from the model the
    metadata points at. Custom models are not resolvable here (their path lives
    in the workflow, not the result), hence the fallback.
    """
    try:
        from nlsq.cli.model_registry import ModelRegistry

        model = ModelRegistry().get_model(
            model_id, {"type": "builtin", "name": model_id}
        )
        # First argument is the independent variable.
        names = list(inspect.signature(model).parameters)[1 : n_params + 1]
        if len(names) == n_params:
            return names
    except Exception:
        pass
    return [f"p{i}" for i in range(n_params)]


def load_results(output_dir: Path, pattern: str) -> list[dict[str, Any]]:
    """Load per-workflow result files, sorted by filename for a stable table."""
    rows = []
    for path in sorted(output_dir.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "file": path.name,
                "name": data.get("metadata", {}).get("workflow_name", path.stem),
                "model": data.get("metadata", {}).get("model_id", "unknown"),
                "popt": data["popt"],
                "uncertainties": data.get(
                    "uncertainties", [float("nan")] * len(data["popt"])
                ),
                "r_squared": data.get("statistics", {}).get("r_squared", float("nan")),
                "status": data.get("convergence", {}).get("status", "unknown"),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> list[str]:
    """Print one row per workflow; returns the parameter names used."""
    n_params = max(len(r["popt"]) for r in rows)
    names = parameter_names(rows[0]["model"], n_params)

    header = f"{'workflow':<22}{'status':<10}"
    header += "".join(f"{n:>18}" for n in names)
    header += f"{'R^2':>8}"
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['name']:<22}{row['status']:<10}"
        for value, sigma in zip(row["popt"], row["uncertainties"], strict=False):
            line += f"{value:>10.4g} +-{sigma:<6.3g}"
        line += f"{row['r_squared']:>8.3f}"
        print(line)

    return names


def write_csv(rows: list[dict[str, Any]], names: list[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["workflow", "model", "status", "r_squared"]
    for name in names:
        fieldnames += [name, f"{name}_stderr"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "workflow": row["name"],
                "model": row["model"],
                "status": row["status"],
                "r_squared": row["r_squared"],
            }
            for name, value, sigma in zip(
                names, row["popt"], row["uncertainties"], strict=False
            ):
                record[name] = value
                record[f"{name}_stderr"] = sigma
            writer.writerow(record)

    print(f"\nWrote {csv_path}")


def report_failures(summary_path: Path) -> None:
    """Successful fits leave a result file; failed ones only appear in the summary."""
    if not summary_path.exists():
        print(f"\nSummary not found: {summary_path}")
        return

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    print(
        f"\nBatch summary: {summary['succeeded']}/{summary['total']} succeeded "
        f"in {summary['duration_seconds']:.2f}s "
        f"({summary['max_workers']} workers)"
    )
    for failure in summary.get("failures", []):
        error = failure.get("error") or {}
        print(f"  FAILED {failure['workflow_path']}: {error.get('message', 'unknown')}")


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Aggregating nlsq batch results")
    print("=" * 70)
    print(f"Directory: {args.output_dir}")
    print()

    rows = load_results(args.output_dir, args.pattern)
    if not rows:
        print(f"No files matching '{args.pattern}' in {args.output_dir}.")
        print("Run the batch first, from examples/scripts/10_cli-commands:")
        print(
            "  nlsq batch workflows/batch_example/*.yaml"
            " --summary output/batch_summary.json"
        )
        return

    names = print_table(rows)

    if args.csv is not None:
        write_csv(rows, names, args.csv)

    if args.summary is not None:
        report_failures(args.summary)


if __name__ == "__main__":
    main()
