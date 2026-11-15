import ast
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("kind", "gallery", "suffix"),
    [
        ("scripts", "04_gallery", "py"),
        ("scripts", "09_gallery_advanced", "py"),
        ("notebooks", "09_gallery_advanced", "ipynb"),
    ],
)
def test_second_order_half_life_uncertainty_includes_covariance(kind, gallery, suffix):
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / kind
        / gallery
        / f"chemistry/reaction_kinetics.{suffix}"
    )
    source = path.read_text()
    if suffix == "ipynb":
        source = "\n".join(
            "".join(cell["source"])
            for cell in json.loads(source)["cells"]
            if cell["cell_type"] == "code"
            and "t_half_2nd_err =" in "".join(cell["source"])
        )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "t_half_2nd_err"
            for target in node.targets
        )
    )
    values = {
        "np": np,
        "t_half_2nd": 50.0,
        "C0_2nd_fit": 2.0,
        "k_2nd_fit": 0.01,
        "C0_2nd_err": 0.2,
        "k_2nd_err": 0.001,
        "pcov_2nd": np.array([[0.04, -0.0001], [-0.0001, 0.000001]]),
    }
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
        values,
    )
    assert values["t_half_2nd_err"] == pytest.approx(5.0)
