"""Regression tests for bugs fixed in the three-brain GUI/CLI review.

Covers gaps flagged by the follow-up /review-pr pass: the GUI's own
custom-model AST validator (_SafeASTValidator) had zero coverage on either
the bypass or happy-path side, config_adapter.merge_configs crashed on any
SessionState carrying numpy array data, and PageState.can_access's
fit_running guard had no test. None of these need a QApplication/qtbot —
they're plain Python logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.gui_qt.adapters.config_adapter import merge_configs
from nlsq.gui_qt.adapters.model_adapter import SecurityError, _SafeASTValidator
from nlsq.gui_qt.pages import PageState
from nlsq.gui_qt.session_state import SessionState


class TestSafeASTValidator:
    """_SafeASTValidator: the GUI's inline-code counterpart to
    cli/model_validation.py's DangerousPatternVisitor."""

    def test_allows_benign_model(self):
        code = """
import numpy as np
def model(x, a, b):
    return a * np.exp(-b * x)
"""
        _SafeASTValidator().validate(code)  # must not raise

    def test_blocks_classic_subclasses_escape(self):
        code = """
def model(x, a):
    c = ().__class__.__bases__[0].__subclasses__()
    return a * x
"""
        with pytest.raises(SecurityError):
            _SafeASTValidator().validate(code)

    def test_blocks_builtins_dict_bypass(self):
        """Regression: bare `__builtins__` Name reference (module dunder,
        not an Attribute) was checked against DANGEROUS_BUILTINS only, which
        didn't include it — letting `__builtins__["__import__"](...)` reach
        __import__ without ever tripping the attribute-access checks."""
        code = """
def model(x, a):
    __builtins__["__import__"]("subprocess").run(["echo", "pwned"])
    return a * x
"""
        with pytest.raises(SecurityError):
            _SafeASTValidator().validate(code)

    def test_blocks_pickle_loads(self):
        """Regression: DANGEROUS_MODULES/DANGEROUS_ATTRS had drifted from
        cli/model_validation.py's list and omitted pickle/dill/cloudpickle/
        inspect/dis/operator/pydoc/telnetlib entirely."""
        code = """
import pickle
def model(x, a, payload=b""):
    pickle.loads(payload)
    return a * x
"""
        with pytest.raises(SecurityError):
            _SafeASTValidator().validate(code)

    def test_blocks_sys_import(self):
        code = "import sys\ndef model(x, a):\n    return a * x\n"
        with pytest.raises(SecurityError):
            _SafeASTValidator().validate(code)


class TestMergeConfigs:
    """merge_configs previously used `!=` to detect overlay-vs-default
    fields, which raises ValueError on numpy arrays (ambiguous truth
    value) — any SessionState carrying xdata/ydata/sigma crashed it."""

    def test_merge_with_array_fields_does_not_crash(self):
        base = SessionState()
        overlay = SessionState(
            xdata=np.array([1.0, 2.0, 3.0]),
            ydata=np.array([4.0, 5.0, 6.0]),
        )
        result = merge_configs(base, overlay)
        assert np.array_equal(result.xdata, overlay.xdata)
        assert np.array_equal(result.ydata, overlay.ydata)

    def test_merge_keeps_base_when_overlay_matches_default(self):
        base = SessionState(gtol=1e-10)
        overlay = SessionState()  # all fields at their dataclass defaults
        result = merge_configs(base, overlay)
        assert result.gtol == 1e-10

    def test_merge_overrides_scalar_field(self):
        base = SessionState(gtol=1e-10)
        overlay = SessionState(gtol=1e-6)
        result = merge_configs(base, overlay)
        assert result.gtol == 1e-6


class TestPageStateFitRunningGuard:
    """PageState.can_access must block navigating to Data Loading/Model
    Selection while a fit is running (the background worker holds a
    snapshot of the current data/model)."""

    def test_blocks_data_and_model_pages_while_fit_running(self):
        state = PageState(
            data_loaded=True,
            model_selected=True,
            fit_complete=False,
            fit_running=True,
        )
        assert state.can_access("data_loading") is False
        assert state.can_access("model_selection") is False
        assert state.can_access("fitting_options") is True

    def test_allows_data_and_model_pages_when_not_running(self):
        state = PageState(
            data_loaded=True,
            model_selected=True,
            fit_complete=False,
            fit_running=False,
        )
        assert state.can_access("data_loading") is True
        assert state.can_access("model_selection") is True

    def test_results_export_gated_on_fit_complete_regardless_of_running(self):
        running_incomplete = PageState(
            data_loaded=True,
            model_selected=True,
            fit_complete=False,
            fit_running=True,
        )
        complete_not_running = PageState(
            data_loaded=True,
            model_selected=True,
            fit_complete=True,
            fit_running=False,
        )
        assert running_incomplete.can_access("results") is False
        assert complete_not_running.can_access("results") is True
