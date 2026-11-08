"""Regression tests for R1 cli-utils review findings F4-F9."""

import logging
import os
import stat

import numpy as np
import pytest

from nlsq.cli.commands.fit import run_fit
from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError, ModelError, setup_logging
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.result_exporter import _atomic_write


def test_f4_atomic_write_respects_umask(tmp_path):
    old = os.umask(0o022)
    try:
        target = tmp_path / "res.json"
        _atomic_write(target, "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        # An existing file keeps its mode, like an in-place write would.
        target.chmod(0o600)
        _atomic_write(target, "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        os.umask(old)


def test_f5_ascii_skip_footer_skips_text_footer(tmp_path):
    f = tmp_path / "footer.txt"
    f.write_text("1 2\n2 4\n3 6\nEND OF DATA\n")
    cfg = {"format": "ascii", "ascii": {"skip_footer": 1}}
    x, y, _ = DataLoader().load(f, cfg)
    np.testing.assert_allclose(x, [1, 2, 3])
    np.testing.assert_allclose(y, [2, 4, 6])


def test_f5_ascii_without_footer_still_fails_on_text(tmp_path):
    f = tmp_path / "footer.txt"
    f.write_text("1 2\n2 4\nEND OF DATA\n")
    with pytest.raises(DataLoadError):
        DataLoader().load(f, {"format": "ascii"})


@pytest.mark.parametrize("degree", [2.9, True, False, "2.5"])
def test_f6_polynomial_degree_must_be_integer(degree):
    with pytest.raises(ModelError):
        ModelRegistry().get_model("p", {"type": "polynomial", "degree": degree})


@pytest.mark.parametrize("degree", [2, 2.0, "3"])
def test_f6_polynomial_integer_degrees_ok(degree):
    ModelRegistry().get_model("p", {"type": "polynomial", "degree": degree})


def test_f7_builtin_polynomial_routes_to_model_builder():
    model = ModelRegistry().get_model(
        "polynomial", {"type": "builtin", "name": "polynomial", "degree": 2}
    )
    # a degree-2 model takes (x, c0, c1, c2): evaluating it must give a number
    assert float(model(2.0, 1.0, 1.0, 1.0)) == pytest.approx(7.0)


def test_f7_builtin_polynomial_without_degree_is_model_error():
    with pytest.raises(ModelError):
        ModelRegistry().get_model(
            "polynomial", {"type": "builtin", "name": "polynomial"}
        )


def test_f8_empty_yaml_sections_do_not_crash(tmp_path):
    data = tmp_path / "d.csv"
    x = np.linspace(0, 1, 20)
    np.savetxt(data, np.c_[x, 2 * x + 1], delimiter=",", header="x,y", comments="")
    out = tmp_path / "out.json"
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        f"metadata:\n"
        f"data:\n  input_file: {data}\n  format: csv\n  columns: {{x: 0, y: 1}}\n"
        f"model:\n  type: builtin\n  name: linear\n"
        f"fitting:\n  p0: [1.0, 1.0]\n"
        f"validation:\n"
        f"visualization:\n"
        f"export:\n  results_file: {out}\n"
    )
    result = run_fit(str(wf))
    assert result is not None
    assert out.exists()


def test_f8_empty_export_section_is_clean_error(tmp_path):
    from nlsq.cli.errors import CLIError
    from nlsq.cli.result_exporter import ResultExporter

    with pytest.raises(CLIError):
        ResultExporter().export({"popt": [1.0], "pcov": [[1.0]]}, {"export": None})


def test_f9_setup_logging_clamps_verbosity():
    logger = setup_logging(verbosity=4, console=True)
    assert logger.verbosity == 3
    assert logger._console_handler.level == logging.DEBUG
