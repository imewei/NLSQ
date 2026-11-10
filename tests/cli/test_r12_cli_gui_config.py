"""Round-12 regressions: quoted CSV, bad tolerance/p0 config, GUI YAML import."""

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import ConfigError
from nlsq.cli.workflow_runner import WorkflowRunner


def test_quoted_numeric_csv_is_parsed(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text('"x","y"\n"0.0","3.5"\n"1.0","4.5"\n"2.0","6.5"\n')
    x, y, _ = DataLoader().load(p, {"format": "csv", "csv": {"has_header": True}})
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y, [3.5, 4.5, 6.5])


def _params(fitting):
    runner = WorkflowRunner()
    x = np.linspace(0, 1, 5)
    return runner._extract_fit_params({"fitting": fitting}, x, x, lambda x, a: a * x)


@pytest.mark.parametrize("fitting", [{"ftol": "abc"}, {"p0": "1, 2"}, {"p0": {"a": 1}}])
def test_bad_fitting_values_raise_config_error(fitting):
    with pytest.raises(ConfigError):
        _params(fitting)


def test_p0_auto_is_case_insensitive():
    assert _params({"p0": "Auto"}) == _params({"p0": "auto"})


def test_gui_yaml_custom_path_and_null_sections():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    st = load_yaml_config("model: {type: custom, path: /abs/m.py, function: f}")
    assert st.model_config["path"] == "/abs/m.py"
    for y in ("fitting:\n", "model:\n", "hybrid_streaming:\n"):
        load_yaml_config(y)


@pytest.mark.parametrize("p0", [2.0, np.array([2.0])])
def test_scalar_and_array_p0_still_accepted(p0):
    out = _params({"p0": p0})
    assert np.asarray(out[0]).shape == (1,)


def test_gui_yaml_code_wins_over_path():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    st = load_yaml_config(
        "model:\n  type: custom\n  custom:\n    file: /abs/m.py\n"
        "    code: 'def f(x, a): return a * x'\n    function: f\n"
    )
    assert "path" not in st.model_config


def test_thread_running_treats_deleted_qthread_as_finished():
    from nlsq.gui_qt.main_window import _thread_running

    class Dead:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    assert _thread_running(Dead()) is False
    assert _thread_running(None) is False


def test_p0_list_accepts_yaml_string_numbers_and_rejects_junk():
    out = _params({"p0": ["1e0"]})
    np.testing.assert_allclose(out[0], [1.0])
    with pytest.raises(ConfigError):
        _params({"p0": ["abc"]})


def test_bounds_yaml_strings_are_coerced_to_float():
    import numpy as np
    import yaml

    from nlsq.cli.workflow_runner import WorkflowRunner

    cfg = yaml.safe_load("fitting: {bounds: {lower: [1e-3, 0], upper: [1e3, 1e1]}}")
    x = np.linspace(0, 1, 5)
    _, bounds, _ = WorkflowRunner()._extract_fit_params(cfg, x, x, None)
    assert bounds[0].dtype == float and bounds[1].dtype == float
    np.testing.assert_allclose(bounds[1], [1e3, 10.0])


def test_bounds_null_is_a_config_error():
    import numpy as np
    import pytest
    import yaml

    from nlsq.cli.errors import ConfigError
    from nlsq.cli.workflow_runner import WorkflowRunner

    cfg = yaml.safe_load("fitting: {bounds: {lower: [null, 0], upper: [1, 2]}}")
    x = np.linspace(0, 1, 5)
    with pytest.raises(ConfigError, match="null/NaN"):
        WorkflowRunner()._extract_fit_params(cfg, x, x, None)
