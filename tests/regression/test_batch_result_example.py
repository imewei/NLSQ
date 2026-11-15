"""Regression checks for the batch result aggregation example."""

import csv
import json
import runpy
from pathlib import Path


def test_mixed_models_use_positional_parameter_names(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/scripts/10_cli-commands/aggregate_batch_results.py"
    )
    example = runpy.run_path(str(script))
    rows = [
        {
            "model": model,
            "name": model,
            "status": "ok",
            "popt": values,
            "uncertainties": [0.0] * 3,
            "r_squared": 1.0,
        }
        for model, values in [("exponential_decay", [1, 2, 3]), ("gaussian", [4, 5, 6])]
    ]
    names = example["print_table"](rows)
    example["write_csv"](rows, names, tmp_path / "parameters.csv")
    with (tmp_path / "parameters.csv").open() as file:
        output = list(csv.DictReader(file))
    assert names == ["p0", "p1", "p2"]
    assert output[1]["p1"] == "5"


def test_missing_uncertainties_keep_fitted_values(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/scripts/10_cli-commands/aggregate_batch_results.py"
    )
    example = runpy.run_path(str(script))
    (tmp_path / "batch_results_1.json").write_text(
        json.dumps({"popt": [1, 2], "uncertainties": []})
    )
    rows = example["load_results"](tmp_path, "batch_results_*.json")
    names = example["print_table"](rows)
    example["write_csv"](rows, names, tmp_path / "parameters.csv")
    with (tmp_path / "parameters.csv").open() as file:
        output = list(csv.DictReader(file))
    assert output[0]["p0"] == "1"
    assert output[0]["p1"] == "2"
