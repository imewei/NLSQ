"""Regression tests for the 2026-11-07 GUI review findings (GUI-1..GUI-6)."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from nlsq.gui_qt.adapters.config_adapter import (
    export_yaml_config,
    load_yaml_config,
)
from nlsq.gui_qt.adapters.data_adapter import parse_table_text
from nlsq.gui_qt.session_state import initialize_state

# --- GUI-1 / GUI-5: shared table parser -------------------------------------


def test_parse_table_text_bom_keeps_first_row():
    rows, header = parse_table_text("﻿1.0,10.0\n2.0,20.0\n3.0,30.0\n")
    assert header is None
    assert rows == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_parse_table_text_header_tab_separated():
    rows, header = parse_table_text("time\tsignal\n1\t2\n3\t4\n")
    assert header == ["time", "signal"]
    assert rows == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_table_text_comments_and_blank_lines():
    rows, header = parse_table_text("# c\n\n1 2 # tail\n3 4\n")
    assert header is None
    assert rows == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.serial
def test_load_file_bom_csv_keeps_first_row(qapp, tmp_path: Path):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    f = tmp_path / "bom.csv"
    f.write_text("1.0,10.0\n2.0,20.0\n3.0,30.0\n", encoding="utf-8-sig")
    page = DataLoadingPage(AppState())
    page._load_file(f)
    assert page._raw_data.shape == (3, 2)
    assert page._raw_data[0].tolist() == [1.0, 10.0]


@pytest.mark.serial
def test_clipboard_with_header_row(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(AppState())
    page._clipboard_text.setPlainText("time\tsignal\n1\t2\n3\t4\n")
    page._on_parse_clipboard()
    assert page._raw_data.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert page._column_names == ["time", "signal"]


# --- GUI-2 / GUI-4: YAML export round trip ----------------------------------


def _state(model_type, model_config, **kw):
    s = initialize_state()
    s.model_type = model_type
    s.model_config = model_config
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_export_custom_one_line_code_round_trips():
    code = "def m(x, a): return a*x"
    s = _state("custom", {"code": code, "function": "m"}, gtol=1e-10)
    text = export_yaml_config(s)
    raw = yaml.safe_load(text)  # must parse (was ScannerError)
    assert isinstance(raw["fitting"]["termination"]["gtol"], float)
    back = load_yaml_config(text)
    assert back.model_type == "custom"
    assert back.custom_code == code
    assert back.custom_function_name == "m"
    assert back.custom_file_path is None
    assert back.gtol == 1e-10


def test_export_polynomial_degree_round_trips():
    s = _state("polynomial", {"degree": 4})
    back = load_yaml_config(export_yaml_config(s))
    assert back.model_type == "polynomial"
    assert back.polynomial_degree == 4


def test_export_bounds_without_p0_round_trip():
    s = _state("builtin", {"name": "gaussian"}, bounds=([0.0, 1.0], [10.0, 5.0]))
    assert s.p0 is None
    back = load_yaml_config(export_yaml_config(s))
    assert back.bounds == ([0.0, 1.0], [10.0, 5.0])


def test_export_does_not_mutate_model_config():
    cfg = {"code": "def m(x, a): return a*x", "function": "m"}
    s = _state("custom", cfg)
    before = copy.deepcopy(cfg)
    export_yaml_config(s)
    assert s.model_config == before


@pytest.mark.serial
def test_bundle_config_yaml_round_trips(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.export import ExportPage

    st = AppState()
    st.state.model_type = "custom"
    st.state.model_config = {"code": "def m(x, a): return a*x", "function": "m"}
    st.state.gtol = 1e-10
    text = ExportPage(st)._generate_config_yaml()
    back = load_yaml_config(text)
    assert back.model_type == "custom"
    assert back.custom_code == "def m(x, a): return a*x"
    assert back.gtol == 1e-10


# --- GUI-3: termination coercion --------------------------------------------


def test_termination_values_are_coerced():
    back = load_yaml_config(
        "fitting:\n  termination:\n    gtol: '1e-10'\n    ftol: 1e-9\n"
        "    xtol: '1e-8'\n    max_iterations: '50'\n    max_function_evals: '100'\n"
    )
    assert back.gtol == 1e-10 and isinstance(back.gtol, float)
    assert back.ftol == 1e-9 and isinstance(back.ftol, float)
    assert back.xtol == 1e-8
    assert back.max_iterations == 50 and isinstance(back.max_iterations, int)
    assert back.max_function_evals == 100


# --- GUI-6: callback wrapper survives deepcopy -------------------------------


def test_callback_wrapper_deepcopy_is_identity():
    from nlsq.gui_qt.adapters.fit_adapter import _CallbackWrapper

    w = _CallbackWrapper(None)
    assert copy.deepcopy(w) is w
