"""Regression tests for the 2026-11-18 round-2 three-brain CLI findings (fixer F5)."""

from __future__ import annotations

import types

import pytest

from nlsq.cli.model_validation import validate_model

_HEAD = "def model(x, a):\n    return a * x\n"


def _violations(tmp_path, body, strict=False):
    path = tmp_path / "m.py"
    path.write_text(body + _HEAD)
    return validate_model(path, strict=strict).violations


_BLOCKED = {
    "evosax_pickle": (
        "from evosax.learned_evolution.les_tools import load_pkl_object\n"
        "load_pkl_object(b'x', pkg_load=True)\n"
    ),
    "import_evosax": "import evosax\n",
    "numpy_test": "import numpy as np\nnp.test(tests=['p.py'])\n",
    "scipy_test": "import scipy\nscipy.test()\n",
    "from_numpy_test": "from numpy import test as t\n",
    "numpy_linalg_test": "import numpy as np\nnp.linalg.test()\n",
    "import_pytest": "import pytest\npytest.main(['p.py'])\n",
    "import_psutil": "import psutil\npsutil.Process().kill()\n",
    "import_tqdm_contrib": "from tqdm.contrib.telegram import tqdm\n",
    "jit_compiler_options": (
        "import jax\nimport jax.numpy as jnp\n"
        "f = jax.jit(jnp.sin, compiler_options={'x': 1})\n"
    ),
    "xla_dump_string": "OPTS = {'xla_dump_to': 'out'}\n",
}


@pytest.mark.parametrize("name", sorted(_BLOCKED))
@pytest.mark.parametrize("strict", [False, True])
def test_round2_validator_blocks_gadget(tmp_path, name, strict):
    assert _violations(tmp_path, _BLOCKED[name], strict=strict), name


def test_round2_local_keyword_named_updated_is_allowed(tmp_path):
    body = (
        "def scale(x, *, updated):\n    return x * updated\n"
        "def helper(x, a):\n    return scale(x, updated=a)\n"
    )
    assert _violations(tmp_path, body) == []


@pytest.mark.parametrize(
    "body",
    [
        "import functools\nfunctools.wraps(len, updated=())\n",
        "def scale(x, *, updated):\n    return x\nscale = __import__\nscale(x, updated=1)\n",
        "w = len\nw(1, assigned=('__doc__',))\n",
    ],
)
def test_round2_hook_keyword_still_blocked_off_local_function(tmp_path, body):
    assert _violations(tmp_path, body)


def test_round2_top_level_return_is_not_valid(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("return 1\n" + _HEAD)
    result = validate_model(path)
    assert not result.is_valid
    assert any("Syntax error" in v for v in result.violations)


@pytest.mark.serial
def test_round2_module_level_jax_compile_does_not_abort_loader(tmp_path):
    """RLIMIT_AS baseline must include XLA's runtime or the loader SIGABRTs."""
    import subprocess
    import sys

    model = tmp_path / "jit_model.py"
    model.write_text(
        "import jax\nimport jax.numpy as jnp\n"
        "P0 = jnp.array([1.0, 0.5])\n"
        "_v = jax.jit(jnp.sin)(jnp.ones(3))\n"
        "def model(x, a, b):\n    return a * jnp.exp(-b * x)\n"
    )
    code = (
        "import sys\nfrom nlsq.cli.model_registry import ModelRegistry\n"
        "m = ModelRegistry().get_model(sys.argv[1], "
        "{'type': 'custom', 'path': sys.argv[1], 'function': 'model'})\n"
        "print('LOADED', callable(m))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(model)],
        cwd=tmp_path,  # model files must live under the working directory
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "LOADED True" in proc.stdout


# ---------------------------------------------------------------- data loaders


def _load_csv(tmp_path, text, **csv_cfg):
    from nlsq.cli.data_loaders import DataLoader

    path = tmp_path / "d.csv"
    path.write_text(text, newline="")
    return DataLoader().load(path, {"columns": {"x": 0, "y": 1}, "csv": csv_cfg})


def test_round2_missing_marker_row_is_not_taken_for_header(tmp_path):
    from nlsq.cli.errors import DataLoadError

    # the first record "0,NA" is data: it must not vanish before the finiteness check
    with pytest.raises(DataLoadError, match="non-finite"):
        _load_csv(tmp_path, "0,NA\n1,2\n2,3\n")


def test_round2_multiline_quoted_field_keeps_numeric_columns(tmp_path):
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "signal", "note"])
    w.writerow([0, 1, "a\nb"])
    w.writerow([1, 2, "c"])
    x, y, _ = _load_csv(tmp_path, buf.getvalue())
    assert x.tolist() == [0.0, 1.0]
    assert y.tolist() == [1.0, 2.0]


def test_round2_duplicate_header_name_is_rejected(tmp_path):
    from nlsq.cli.data_loaders import DataLoader
    from nlsq.cli.errors import DataLoadError

    path = tmp_path / "d.csv"
    path.write_text("time,signal,signal\n0,1,10\n1,2,20\n")
    cfg = {"columns": {"x": "time", "y": "signal"}}
    with pytest.raises(DataLoadError, match="appears 2 times"):
        DataLoader().load(path, cfg)
    cfg["columns"]["y"] = 2  # explicit index stays usable
    _, y, _ = DataLoader().load(path, cfg)
    assert y.tolist() == [10.0, 20.0]


# --------------------------------------------------------------------- logging


def test_round2_failed_reconfigure_keeps_previous_logger(tmp_path):
    import logging

    from nlsq.cli.errors import setup_logging

    log = tmp_path / "a.log"
    logger = setup_logging(log_file=log, console=False, verbosity=1)
    with pytest.raises(OSError):
        setup_logging(log_file=tmp_path, console=False)  # a directory
    logger.warning("still-alive")
    for h in logging.getLogger("nlsq.cli").handlers:
        h.flush()
    assert "still-alive" in log.read_text()


def test_round2_file_gets_debug_regardless_of_console_verbosity(tmp_path):
    import logging

    from nlsq.cli.errors import setup_logging

    log = tmp_path / "b.log"
    logger = setup_logging(log_file=log, console=False, verbosity=1)
    logger.debug("dbg-msg")
    logger.info("info-msg")
    for h in logging.getLogger("nlsq.cli").handlers:
        h.flush()
    text = log.read_text()
    assert "dbg-msg" in text and "info-msg" in text
    setup_logging(console=False)  # drop the file handler again


def test_round2_standard_extra_mapping_is_merged(tmp_path):
    import json
    import logging

    from nlsq.cli.errors import setup_logging

    log = tmp_path / "c.jsonl"
    logger = setup_logging(log_file=log, console=False, structured=True)
    logger.warning("sample", extra={"context": {"id": 42}})
    for h in logging.getLogger("nlsq.cli").handlers:
        h.flush()
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["context"] == {"id": 42}
    setup_logging(console=False)


# -------------------------------------------------------------- model registry


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # model files must live under the working directory


def _get_model(path, function="model", **cfg):
    from nlsq.cli.model_registry import ModelRegistry

    config = {"type": "custom", "path": str(path), "function": function, **cfg}
    return ModelRegistry().get_model(str(path), config)


def test_round2_dataclass_with_postponed_annotations_loads(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Settings:\n    factor: float = 2.0\n"
        "def model(x, a):\n    return a * x * Settings().factor\n"
    )
    assert _get_model(p)(2.0, 3.0) == 12.0


def test_round2_jitted_first_model_owns_bare_helpers(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(
        "import jax\n"
        "@jax.jit\ndef first(x, a):\n    return a * x\n"
        "def second(x, a, b):\n    return a * x + b\n"
        "def estimate_p0(x, y):\n    return [1.0]\n"
        "def bounds():\n    return ([0.0], [1.0])\n"
    )
    first = _get_model(p, "first")
    second = _get_model(p, "second")
    assert hasattr(first, "estimate_p0") and hasattr(first, "bounds")
    assert not hasattr(second, "estimate_p0") and not hasattr(second, "bounds")


def test_round2_required_keyword_only_parameter_is_rejected(tmp_path):
    from nlsq.cli.errors import ModelError

    p = tmp_path / "m.py"
    p.write_text("def model(x, *, a):\n    return a * x\n")
    with pytest.raises(ModelError, match=r"keyword-only.*'a'"):
        _get_model(p)
    p.write_text("def model(x, a, *, scale=1.0):\n    return a * x * scale\n")
    assert callable(_get_model(p))


# --------------------------------------------------------------- visualization


def _viz(tmp_path, model, popt, y, vis_extra=None):
    import numpy as np

    from nlsq.cli.visualization import FitVisualizer

    x = np.arange(1.0, 9.0)
    config = {
        "visualization": {
            "enabled": True,
            "output_dir": str(tmp_path),
            "formats": ["png"],
            "histogram": {"enabled": True, "bins": "sqrt"},
            **(vis_extra or {}),
        }
    }
    result = {"popt": np.array(popt), "pcov": np.eye(len(popt)) * 0.01}
    return FitVisualizer().generate(
        result, {"xdata": x, "ydata": np.asarray(y, dtype=float)}, model, config
    )


def test_round2_constant_model_scalar_prediction_is_plotted(tmp_path):
    paths = _viz(tmp_path, lambda x, c: c, [3.0], [3.0] * 8)
    assert any("combined" in str(p) for p in paths)


def test_round2_empty_finite_residuals_histogram_does_not_crash(tmp_path):
    import numpy as np

    for bins in ("sqrt", "sturges", "auto"):
        paths = _viz(
            tmp_path,
            lambda x, a: x * np.nan,
            [1.0],
            [1.0] * 8,
            {"histogram": {"enabled": True, "bins": bins}},
        )
        assert any("histogram" in str(p) for p in paths), bins


def test_round2_bare_bool_histogram_is_disabled_not_a_crash(tmp_path):
    paths = _viz(tmp_path, lambda x, a: a * x, [1.0], [1.0] * 8, {"histogram": True})
    assert paths and not any("histogram" in str(p) for p in paths)


# ------------------------------------------------------ commands and exporter


def test_round2_numeric_results_file_is_a_config_error(tmp_path):
    import yaml

    from nlsq.cli.commands.fit import run_fit
    from nlsq.cli.errors import ConfigError

    data = tmp_path / "d.txt"
    data.write_text("0 1\n1 3\n2 5\n")
    cfg = {
        "data": {"input_file": str(data), "format": "ascii"},
        "model": {"type": "builtin", "name": "linear"},
        "export": {"results_file": 42},
    }
    wf = tmp_path / "w.yaml"
    wf.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match=r"export\.results_file"):
        run_fit(str(wf))


def test_round2_config_next_steps_name_the_published_file(
    tmp_path, monkeypatch, capsys
):
    from nlsq.cli.commands.config import run_config

    monkeypatch.chdir(tmp_path)
    run_config(workflow=True, output="experiment.yaml")
    out = capsys.readouterr().out
    assert "nlsq fit experiment.yaml" in out
    assert "workflow_config.yaml" not in out


def test_round2_config_long_filename_is_a_handled_error(tmp_path, monkeypatch, capsys):
    from nlsq.cli.main import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "--workflow", "-o", "a" * 300 + ".yaml"]) == 1
    assert "Error" in capsys.readouterr().err


def test_round2_exporter_widens_integer_residuals_before_squaring(tmp_path):
    import json

    import numpy as np

    from nlsq.cli.result_exporter import ResultExporter

    out = tmp_path / "r.json"
    ResultExporter().export(
        {"popt": np.array([0.0]), "success": True, "fun": np.array([50000], np.int32)},
        {"export": {"results_file": str(out), "format": "json"}},
    )
    stats = json.loads(out.read_text())["statistics"]
    assert stats["chi_squared"] == 2.5e9
    assert stats["rmse"] == 50000.0


def test_round2_hdf5_null_dataspace_is_a_load_error(tmp_path):
    import h5py
    import numpy as np

    from nlsq.cli.data_loaders import DataLoader
    from nlsq.cli.errors import DataLoadError

    p = tmp_path / "e.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=h5py.Empty("f8"))
        f.create_dataset("y", data=np.arange(3.0))
    with pytest.raises(DataLoadError, match="'x' is empty"):
        DataLoader().load(p, {"format": "hdf5", "hdf5": {"x_path": "x", "y_path": "y"}})


def test_round2_interrupt_with_nothing_queued_exits_130(tmp_path, monkeypatch):
    import time

    from nlsq.cli.commands import batch
    from nlsq.cli.main import handle_batch

    wf = tmp_path / "w.yaml"
    wf.write_text("data: {}\n")

    def slow_fit(path, verbose):
        time.sleep(0.2)
        return {
            "status": "succeeded",
            "workflow_path": path,
            "result": {},
            "error": None,
        }

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_run_single_workflow", slow_fit)
    monkeypatch.setattr(batch, "_drain_futures", interrupted)
    monkeypatch.setattr(batch, "_validate_batch_inputs", lambda *a, **k: None)

    args = types.SimpleNamespace(
        workflows=[str(wf)],
        summary=None,
        workers=1,
        continue_on_error=True,
        verbose=False,
    )
    assert handle_batch(args) == 130


# ----------------------------------------------------------------------- wave 3


def test_wave3_literal_quote_in_unquoted_field_keeps_all_rows(tmp_path):
    x, y, _ = _load_csv(tmp_path, 'x,y,note\n1,2,5" tube\n2,3,ordinary\n3,4,6" tube\n')
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [2.0, 3.0, 4.0]


def test_wave3_multiline_header_record_is_skipped_as_one_record(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    path = tmp_path / "h.csv"
    path.write_text('x,"label\nwith space"\n1,2\n3,4\n')
    cfg = {"columns": {"x": "x", "y": "label\nwith space"}}
    x, y, _ = DataLoader().load(path, cfg)
    assert x.tolist() == [1.0, 3.0]
    assert y.tolist() == [2.0, 4.0]


def test_wave3_all_marker_first_row_is_reported_not_dropped(tmp_path):
    from nlsq.cli.errors import DataLoadError

    with pytest.raises(DataLoadError, match="non-finite"):
        _load_csv(tmp_path, "NA,NA\n1,2\n2,3\n")


def test_wave3_custom_model_loads_do_not_accumulate_modules(tmp_path):
    import sys

    p = tmp_path / "m.py"
    p.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass S:\n    k: float = 2.0\n"
        "def model(x, a):\n    return a * x * S().k\n"
    )
    before = {n for n in sys.modules if n.startswith("nlsq_custom_model_")}
    for _ in range(4):
        assert _get_model(p)(1.0, 3.0) == 6.0
    after = {n for n in sys.modules if n.startswith("nlsq_custom_model_")}
    assert after == before


def test_wave3_data_length_model_plots_at_observed_points(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np

    from nlsq.cli.visualization import FitVisualizer

    x = np.array([0.0, 1.0, 3.0, 4.0])
    popt = np.array([2.0, 3.0])

    def model(xx, a, b):
        return a * xx + b * np.arange(4.0)  # only valid on 4 points

    fig, ax = plt.subplots()
    FitVisualizer()._plot_main(
        ax, x, model(x, *popt), None, popt, None, model, {}, {}, {}
    )
    fit = next(ln for ln in ax.lines if len(ln.get_xdata()) == 4)
    np.testing.assert_array_equal(fit.get_xdata(), x)
    np.testing.assert_array_equal(fit.get_ydata(), [0.0, 5.0, 12.0, 17.0])
    plt.close(fig)


_NLSQ_BLOCKED = {
    "registry": "from nlsq.cli.model_registry import ModelRegistry\n",
    "exporter": "from nlsq.cli.result_exporter import ResultExporter\n",
    "import_root": "import nlsq\n",
    "import_root_alias_cli": "import nlsq as n\nn.cli\n",
    "from_nlsq_cli": "from nlsq import cli\n",
    "from_nlsq_curve_fit": "from nlsq import curve_fit\n",
    "import_submodule_binds_root": "import nlsq.core.functions\n",
    "import_cli_as": "import nlsq.cli.model_registry as r\n",
    "wildcard": "from nlsq.core.functions import *\n",
    "jit_star_star": "import jax\nf = jax.jit(lambda x: x, **{'a': 1})\n",
    "jit_star_star_alias": "from jax import jit\nf = jit(lambda x: x, **{'a': 1})\n",
}

_NLSQ_ALLOWED_IMPORTS = {
    "from_core_functions": "from nlsq.core.functions import gaussian\n",
    "from_core_import_functions": "from nlsq.core import functions\n",
    "from_nlsq_functions": "from nlsq import functions\n",
    "import_as": "import nlsq.core.functions as f\nf.gaussian\n",
    "jit_plain": "import jax\nf = jax.jit(lambda x: x)\n",
}


@pytest.mark.parametrize("name", sorted(_NLSQ_BLOCKED))
@pytest.mark.parametrize("strict", [False, True])
def test_wave3_model_file_cannot_reach_nlsq_internals(tmp_path, name, strict):
    assert _violations(tmp_path, _NLSQ_BLOCKED[name], strict=strict), name


@pytest.mark.parametrize("name", sorted(_NLSQ_ALLOWED_IMPORTS))
def test_wave3_nlsq_function_reexports_stay_allowed(tmp_path, name):
    assert _violations(tmp_path, _NLSQ_ALLOWED_IMPORTS[name]) == [], name


def test_wave3_shipped_models_still_validate():
    from pathlib import Path

    from nlsq.cli.templates import get_custom_model_template

    root = Path(__file__).resolve().parents[2]
    files = [
        get_custom_model_template(),
        *sorted((root / "examples/scripts/10_cli-commands/models").glob("*.py")),
        *sorted((root / "tests/cli/fixtures").glob("*model*.py")),
    ]
    assert len(files) >= 7
    for f in files:
        assert validate_model(Path(f)).violations == [], f


def test_wave3_config_display_path_survives_symlinked_prefix(
    tmp_path, monkeypatch, capsys
):
    import os

    from nlsq.cli.commands.config import run_config

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    monkeypatch.chdir(real)
    # absolute -o through a symlinked prefix that resolves inside the CWD
    run_config(workflow=True, output=str(link / "w.yaml"))
    assert (real / "w.yaml").exists()
    assert "w.yaml" in capsys.readouterr().out


# ----------------------------------------------------------------------- wave 4

_PKG_BLOCKED = {
    "pkg_nlsq": '__package__ = "nlsq"\nfrom .cli.model_registry import ModelRegistry\n',
    "pkg_nlsq_core": '__package__ = "nlsq.core"\nfrom . import minpack\n',
    "pkg_parent_dots": '__package__ = "nlsq.core"\nfrom ..cli import model_registry\n',
    "pkg_os": '__package__ = "os"\nfrom . import getcwd as gc\n',
    "pkg_subprocess": '__package__ = "subprocess"\nfrom . import run\n',
    "plain_relative": "from . import helper\n",
    "annassign": '__package__: str = "os"\n',
    "augassign": '__package__ = "o"\n__package__ += "s"\n',
    "walrus": '(__package__ := "os")\n',
    "for_target": 'for __package__ in ["os"]:\n    pass\n',
    "with_target": "import contextlib\nwith contextlib.nullcontext() as __package__:\n    pass\n",
    "tuple_target": '__package__, y = "os", 1\n',
    "star_target": "a, *__package__ = 1, 2\n",
    "import_as": "import math as __package__\n",
    "from_import_as": "from math import pi as __package__\n",
    "def_name": "def __package__():\n    pass\n",
    "spec": "__spec__ = None\n",
    "path": '__path__ = ["/tmp"]\n',
    "loader": "__loader__ = None\n",
    "name": '__name__ = "os"\n',
    "attr_store": "import math\nmath.__package__ = 'os'\n",
    "globals_store": 'globals()["__package__"] = "os"\n',
}


@pytest.mark.parametrize("name", sorted(_PKG_BLOCKED))
@pytest.mark.parametrize("strict", [False, True])
def test_wave4_relative_import_and_package_rebinding_are_blocked(
    tmp_path, name, strict
):
    assert _violations(tmp_path, _PKG_BLOCKED[name], strict=strict), name


def test_wave4_ordinary_dunders_stay_allowed(tmp_path):
    body = (
        '__doc__ = "model"\n__version__ = "1"\n__all__ = ["model"]\n'
        "class C:\n    __slots__ = ('a',)\n    def __init__(self):\n        self.a = 1\n"
        "def f(x):\n    return __name__ == '__main__'\n"
    )
    assert _violations(tmp_path, body) == []


def test_wave4_band_is_centred_on_plotted_curve_for_unsorted_observations():
    import matplotlib.pyplot as plt
    import numpy as np

    from nlsq.cli.visualization import FitVisualizer

    x = np.array([4.0, 0.0, 3.0, 1.0])
    popt = np.array([2.0, 3.0])

    def model(xx, a, b):
        return a * xx + b * np.arange(4.0)  # depends on observation order

    cfg = {"main_plot": {"confidence_band": {"enabled": True}}}
    fig, ax = plt.subplots()
    FitVisualizer()._plot_main(
        ax,
        x,
        model(x, *popt),
        np.ones(4),
        popt,
        np.eye(2),
        model,
        cfg,
        {},
        {"absolute_sigma": True},
    )
    fit = next(ln for ln in ax.lines if ln.get_label() == "Fit")
    np.testing.assert_array_equal(fit.get_xdata(), [0.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(fit.get_ydata(), [3.0, 11.0, 12.0, 8.0])
    poly = ax.collections[-1].get_paths()[0].vertices
    # fill_between polygon: lower edge left-to-right, then upper edge back
    n = 4
    lower, upper = poly[1 : n + 1, 1], poly[n + 2 : 2 * n + 2, 1][::-1]
    np.testing.assert_allclose((lower + upper) / 2, fit.get_ydata())
    plt.close(fig)
