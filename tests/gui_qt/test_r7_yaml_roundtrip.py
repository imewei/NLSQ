"""R7: GUI YAML round trip keeps the model selection; bare termination keys use defaults."""

import yaml

from nlsq.gui_qt.adapters.config_adapter import export_yaml_config, load_yaml_config
from nlsq.gui_qt.session_state import SessionState


def test_polynomial_round_trip_and_no_stale_name():
    s = SessionState(model_type="polynomial", model_config={"degree": 4})
    text = export_yaml_config(s)
    assert "name" not in yaml.safe_load(text)["model"]
    r = load_yaml_config(text)
    assert r.model_type == "polynomial"
    assert r.polynomial_degree == 4
    assert r.model_config == {"degree": 4}


def test_custom_round_trip():
    s = SessionState(
        model_type="custom",
        model_config={"code": "def f(x, a):\n    return a * x\n", "function": "f"},
    )
    r = load_yaml_config(export_yaml_config(s))
    assert r.model_type == "custom"
    assert r.model_config["function"] == "f"
    assert "return a * x" in r.model_config["code"]


def test_builtin_round_trip_keeps_name():
    s = SessionState(model_type="builtin", model_config={"name": "gaussian"})
    text = export_yaml_config(s)
    assert yaml.safe_load(text)["model"]["name"] == "gaussian"
    r = load_yaml_config(text)
    assert r.model_name == "gaussian"
    assert r.model_config == {"name": "gaussian"}


def test_bare_termination_keys_use_defaults():
    base = SessionState()
    r = load_yaml_config(
        "fitting:\n  termination:\n    gtol:\n    ftol:\n    xtol:\n"
        "    max_iterations:\n    max_function_evals:\n"
    )
    assert (r.gtol, r.ftol, r.xtol) == (base.gtol, base.ftol, base.xtol)
    assert r.max_iterations == base.max_iterations
