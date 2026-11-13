import pytest
import yaml

from nlsq.cli.commands.batch import run_batch


@pytest.mark.serial
@pytest.mark.parametrize(
    ("section", "invalid"),
    [
        ("export", ["result.json"]),
        ("export", {"bad": "path"}),
        ("export", "\x00"),
        ("data", "\x00"),
        ("model", "\x00"),
        ("visualization", "\x00"),
    ],
)
def test_invalid_path_does_not_stop_other_workflows(tmp_path, section, invalid):
    data = tmp_path / "data.txt"
    data.write_text("0 1\n1 3\n2 5\n")
    config = {
        "data": {
            "input_file": str(data),
            "format": "ascii",
            "columns": {"x": 0, "y": 1},
        },
        "model": {"type": "builtin", "name": "linear"},
        "fitting": {"p0": [1.0, 1.0]},
        "export": {"results_file": str(tmp_path / "result.json")},
    }
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(config))
    if section == "model":
        config[section] = {"type": "custom", "path": invalid}
    elif section == "visualization":
        config[section] = {"enabled": True, "output_dir": invalid}
    else:
        key = "input_file" if section == "data" else "results_file"
        config[section][key] = invalid
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config))

    results = run_batch([str(bad), str(good)], max_workers=1)

    assert [result["status"] for result in results] == ["failed", "success"]
    assert (tmp_path / "result.json").is_file()
