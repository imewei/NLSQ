"""Regression tests for the 2026-11-18 three-brain CLI findings (fixer F5)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from nlsq.cli.commands import config as config_module
from nlsq.cli.commands.batch import _warn_output_collisions
from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import CLIError, DataLoadError
from nlsq.cli.model_validation import validate_model
from nlsq.cli.result_exporter import ResultExporter
from nlsq.cli.visualization import FitVisualizer

_HEAD = "def model(x, a):\n    return a * x\n"

# One entry per gadget class: code-string compiler, stdlib re-export, private
# _src re-export, Python 3.14 pathlib mutators, and the value-alias variants that
# defeat import-alias tracking.
_BLOCKED = {
    "numpy_testing_measure": "import numpy as np\nnp.testing.measure('1', 1)\n",
    "from_numpy_testing": "from numpy.testing import measure\n",
    "from_numpy_import_testing": "from numpy import testing\n",
    "value_alias_testing": "import numpy as np\nm = [np][0]\nm.testing.measure('1', 1)\n",
    "numpy_lib_format": "import numpy as np\nnp.lib.format.open_memmap('f', mode='w+')\n",
    "jax_debug": "import jax\njax.debug.callback(print)\n",
    "jax_tools_to_ir": "import jax.tools.jax_to_ir\n",
    "from_jax_tools": "from jax.tools import jax_to_ir\n",
    "jax_experimental_logging": (
        "import jax\n"
        "jax.experimental.mosaic.gpu.utils.logging.FileHandler('f', mode='w')\n"
    ),
    "from_jax_experimental_mosaic": "from jax.experimental.mosaic.gpu import utils\n",
    "value_alias_logging": (
        "import jax\nm = [jax][0]\nm.experimental.mosaic.gpu.utils.logging.FileHandler('f')\n"
    ),
    "jax_experimental_xb": (
        "import jax\njax.experimental.topologies.xb.register_plugin('p', library_path='x')\n"
    ),
    "value_alias_xla_bridge": (
        "import jax\nm = [jax][0]\nm.experimental.multihost_utils.xla_bridge.x\n"
    ),
    "path_copy": "import pathlib\npathlib.Path('a').copy('b')\n",
    "path_copy_kw_target": "import pathlib\npathlib.Path('a').copy(target='b')\n",
    "path_copy_kw_order_and_target": (
        "import pathlib\npathlib.Path('a').copy('C', target='b')\n"
    ),
    "path_copy_double_star": (
        "import pathlib\npathlib.Path('a').copy(**{'target': 'b'})\n"
    ),
    "path_copy_star_args": "import pathlib\npathlib.Path('a').copy(*['b'])\n",
    "path_copy_two_positional": "import pathlib\npathlib.Path('a').copy('C', 'F')\n",
    "path_copy_bound_alias": ("import pathlib\nc = pathlib.Path('a').copy\nc('b')\n"),
    "copy_nonliteral_order_documented_limit": (
        "def f(x):\n    order = 'C'\n    return x.copy(order)\n"
    ),
    "reexport_jax_version_pathlib_copy": (
        "import jax\njax.version.pathlib.Path.copy(jax.version.pathlib.Path('a'), 'b')\n"
    ),
    "reexport_jax_version_pathlib_copy_kw": (
        "import jax\n"
        "jax.version.pathlib.Path.copy(jax.version.pathlib.Path('a'), target='b')\n"
    ),
    "reexport_bound_copy": (
        "import jax\nc = jax.version.pathlib.Path.copy\n"
        "c(jax.version.pathlib.Path('a'), 'b')\n"
    ),
    "reexport_from_jax_version_pathlib": (
        "from jax.version import pathlib\npathlib.Path.copy(pathlib.Path('a'), 'b')\n"
    ),
    "reexport_numpy_conftest_path": (
        "import numpy.conftest as c\nc.Path.copy(c.Path('a'), 'b')\n"
    ),
    "reexport_value_alias_conftest": (
        "import numpy as np\nm = [np][0]\nm.conftest.Path('a')\n"
    ),
    "reexport_value_alias_pathlib_copy": (
        "import jax\nm = [jax][0]\n"
        "m.version.pathlib.Path.copy(m.version.pathlib.Path('a'), 'b')\n"
    ),
    "reexport_codecs": "import numpy as np\nnp.f2py.crackfortran.codecs\n",
    "reexport_numpy_test_package": "import numpy as np\nnp.lib.tests\n",
    "path_copy_into": "import pathlib\npathlib.Path('a').copy_into('b')\n",
    "path_move": "import pathlib\npathlib.Path('a').move('b')\n",
    "path_move_into": "import pathlib\npathlib.Path('a').move_into('b')\n",
    "path_lchmod": "import pathlib\npathlib.Path('a').lchmod(0)\n",
}

_ALLOWED = {
    "experimental_ode_sparse": (
        "import jax.numpy as jnp\nfrom jax.experimental import ode, sparse\n"
        "import jax.experimental.sparse as sp\n"
    ),
    "numpy_copy": "import numpy as np\nimport jax.numpy as jnp\nnp.copy([1])\njnp.copy([1])\n",
    "array_copy_methods": "def f(x):\n    return x.copy(), x.copy('C'), x.copy(order='F')\n",
    "dunder_version": "import numpy as np\nnp.__version__\n",
    "array_copy_kw_literal_order": "def f(x):\n    return x.copy(order='K'), x.copy('A')\n",
    "numpy_copy_value_use": "import numpy as np\nlist(map(np.copy, [[1]]))\nnp.copy([1], order='C')\n",
    "numeric_copy_forms": (
        "import jax.numpy as jnp\nimport numpy as np\n"
        "np.copy(1)\njnp.copy(1)\nnp.ma.copy(1)\nnp.ndarray.copy(np.ones(2))\n"
        "np.version.version\n"
    ),
    "dict_copy": "def f(d):\n    return d.copy()\n",
}


def _violations(tmp_path, body, strict=False):
    path = tmp_path / "m.py"
    path.write_text(body + _HEAD)
    return validate_model(path, strict=strict).violations


@pytest.mark.parametrize("name", sorted(_BLOCKED))
def test_validator_blocks_gadget(tmp_path, name):
    assert _violations(tmp_path, _BLOCKED[name]), name


@pytest.mark.parametrize("name", sorted(_ALLOWED))
def test_validator_keeps_legitimate_numeric_use(tmp_path, name):
    assert _violations(tmp_path, _ALLOWED[name]) == [], name


def _export_config(tmp_path, **flags):
    return {"export": {"results_file": str(tmp_path / "out.json"), **flags}}


@pytest.mark.parametrize("key", ["stdout", "skip_file_on_stdout"])
def test_quoted_export_flag_rejected_not_truthy(tmp_path, key):
    result = {"popt": [1.0], "pcov": [[1.0]], "success": True, "message": "ok"}
    cfg = _export_config(tmp_path, **{key: "false"})
    with pytest.raises(CLIError, match=key):
        ResultExporter().export(result, cfg)
    assert not (tmp_path / "out.json").exists()


def test_fit_rejects_quoted_export_flag_before_fitting(tmp_path):
    from nlsq.cli.commands.fit import run_fit

    data = tmp_path / "d.txt"
    data.write_text("0 1\n1 3\n2 5\n")
    cfg = {
        "data": {"input_file": str(data), "format": "ascii"},
        "model": {"type": "builtin", "name": "linear"},
        "export": {
            "results_file": str(tmp_path / "out.json"),
            "stdout": True,
            "skip_file_on_stdout": "false",
        },
    }
    wf = tmp_path / "w.yaml"
    wf.write_text(yaml.safe_dump(cfg))
    with pytest.raises(CLIError, match="skip_file_on_stdout"):
        run_fit(str(wf))
    assert not (tmp_path / "out.json").exists()


def test_batch_preflight_sees_results_file_vs_figure_collision(tmp_path, caplog):
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump({"export": {"results_file": str(tmp_path / "fit_combined.png")}})
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "export": {"results_file": str(tmp_path / "b.json")},
                "visualization": {
                    "enabled": True,
                    "filename_prefix": "fit",
                    "output_dir": str(tmp_path),
                    "formats": ["png"],
                },
            }
        )
    )
    logger = logging.getLogger("f5-collision")
    with caplog.at_level(logging.WARNING, logger="f5-collision"):
        _warn_output_collisions(
            [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")], logger
        )
    assert "collision" in caplog.text.lower()


def _csv(tmp_path, header_cfg):
    path = tmp_path / "y.csv"
    path.write_text("2024,2025\n1,2\n3,4\n")
    cfg = {"data": {"columns": {"x": 0, "y": 1}, "csv": header_cfg}}
    return DataLoader().load(str(path), cfg["data"])


def test_explicit_numeric_header_is_honoured(tmp_path):
    x, y, _ = _csv(tmp_path, {"header": True})[:3]
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [2.0, 4.0])


def test_default_numeric_header_still_treated_as_data(tmp_path):
    x = _csv(tmp_path, {})[0]
    np.testing.assert_array_equal(x, [2024.0, 1.0, 3.0])


def test_ascii_complex_dtype_rejected(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("1+2j 3+4j\n2+5j 6+7j\n")
    cfg = {"columns": {"x": 0, "y": 1}, "ascii": {"dtype": "complex128"}}
    with pytest.raises(DataLoadError, match="complex"):
        DataLoader().load(str(path), {"format": "ascii", **cfg})


def test_config_command_partial_failure_leaves_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom_model.py").write_text("keep")
    with pytest.raises(FileExistsError):
        config_module.run_config()
    assert not (tmp_path / "workflow_config.yaml").exists()
    assert (tmp_path / "custom_model.py").read_text() == "keep"


def test_failed_plot_construction_closes_figure(tmp_path):
    plt.close("all")

    def scalar_model(x, a):
        return a * np.ones(3)  # plot() rejects 3 values against 500 grid points

    cfg = {"visualization": {"output_dir": str(tmp_path), "formats": ["png"]}}
    data = {"xdata": np.array([0.0, 1.0]), "ydata": np.array([1.0, 1.0])}
    with pytest.raises(Exception):  # noqa: B017 - any plot failure
        FitVisualizer().generate({"popt": [1.0]}, data, scalar_model, cfg)
    assert plt.get_fignums() == []


def test_explicit_numeric_header_selectable_by_name(tmp_path):
    path = tmp_path / "n.csv"
    path.write_text("2024,2025\n1,2\n3,4\n")
    x, y, _ = DataLoader().load(
        str(path),
        {"columns": {"x": "2024", "y": "2025"}, "csv": {"header": True}},
    )[:3]
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [2.0, 4.0])


def test_shipped_template_leaves_csv_header_unset():
    from nlsq.cli.templates import get_workflow_template

    cfg = yaml.safe_load(get_workflow_template().read_text())
    assert "header" not in cfg["data"]["csv"]


def test_copy_violation_message_names_the_remedy(tmp_path):
    msgs = _violations(tmp_path, "def f(x, order):\n    return x.copy(order)\n")
    assert any("literal order" in m and "trusted" in m for m in msgs)


def test_shipped_templates_still_validate():
    from nlsq.cli.templates import get_custom_model_template

    assert validate_model(get_custom_model_template()).violations == []


def test_config_command_second_copy_failure_is_all_or_nothing(tmp_path, monkeypatch):
    import shutil

    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom_model.py").write_text("old")
    real_copy = shutil.copy
    calls = []

    def failing_copy(src, dst, *a, **k):
        calls.append(dst)
        if len(calls) == 2:  # the model template, after preflight passed
            raise OSError("disk full")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(config_module.shutil, "copy", failing_copy)
    with pytest.raises(OSError, match="disk full"):
        config_module.run_config(force=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["custom_model.py"]
    assert (tmp_path / "custom_model.py").read_text() == "old"


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="needs POSIX FIFO")
def test_config_command_force_replaces_special_file(tmp_path, monkeypatch):
    import os
    import stat

    monkeypatch.chdir(tmp_path)
    os.mkfifo(tmp_path / "custom_model.py")
    config_module.run_config(force=True)
    assert stat.S_ISREG((tmp_path / "custom_model.py").stat().st_mode)
    assert (tmp_path / "workflow_config.yaml").is_file()


def _names(path):
    return sorted(p.name for p in path.iterdir())


@pytest.mark.parametrize("race", ["directory", "regular_file"])
def test_config_command_second_publish_failure_rolls_back_first(
    tmp_path, monkeypatch, race
):
    import os

    monkeypatch.chdir(tmp_path)
    real_link = os.link
    calls = []

    def racing_link(src, dst, *a, **k):
        calls.append(dst)
        if len(calls) == 2:  # appears after preflight, before the second publish
            if race == "directory":
                os.mkdir(dst)
            else:
                Path(dst).write_text("mine")
        return real_link(src, dst, *a, **k)

    monkeypatch.setattr(config_module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        config_module.run_config()
    expected = ["custom_model.py"]  # only what the racer created remains
    assert _names(tmp_path) == expected
    if race == "regular_file":
        assert (tmp_path / "custom_model.py").read_text() == "mine"


def test_config_command_force_second_install_failure_restores_originals(
    tmp_path, monkeypatch
):
    import os

    monkeypatch.chdir(tmp_path)
    (tmp_path / "workflow_config.yaml").write_text("old-wf")
    (tmp_path / "custom_model.py").write_text("old-model")
    real_replace = os.replace

    def failing_replace(src, dst, *a, **k):
        if os.path.basename(dst) == "custom_model.py" and ".nlsq-" in str(src):
            if str(src).endswith(".tmp"):
                raise OSError("boom")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(config_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="boom"):
        config_module.run_config(force=True)
    assert _names(tmp_path) == ["custom_model.py", "workflow_config.yaml"]
    assert (tmp_path / "workflow_config.yaml").read_text() == "old-wf"
    assert (tmp_path / "custom_model.py").read_text() == "old-model"


def test_config_command_force_success_removes_backups(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workflow_config.yaml").write_text("old")
    config_module.run_config(force=True)
    assert _names(tmp_path) == ["custom_model.py", "workflow_config.yaml"]
    assert (tmp_path / "workflow_config.yaml").read_text() != "old"


def test_config_command_accepts_251_character_output_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "a" * 246 + ".yaml"
    assert len(name) == 251
    config_module.run_config(workflow=True, output=name)
    assert _names(tmp_path) == [name]
