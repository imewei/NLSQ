"""Round-5 security regressions: GUI custom models, exported script, env dump."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.serial

GOOD = "def m(x, a):\n    return a * x\n"
# denylist-clean but outside the strict allowlist
GZIP = "import gzip\n" + GOOD


class TestS1StrictModels:
    def test_strict_env_rejects_non_allowlisted_import(self, monkeypatch):
        from nlsq.gui_qt.adapters.model_adapter import (
            SecurityError,
            parse_custom_model_string,
        )

        monkeypatch.setenv("NLSQ_STRICT_MODELS", "1")
        with pytest.raises(SecurityError):
            parse_custom_model_string(GZIP, "m")

    def test_strict_env_accepts_plain_math_model(self, monkeypatch):
        from nlsq.gui_qt.adapters.model_adapter import parse_custom_model_string

        monkeypatch.setenv("NLSQ_STRICT_MODELS", "1")
        func, params = parse_custom_model_string(GOOD, "m")
        assert params == ["a"] and func(2.0, 3.0) == 6.0

    def test_default_path_unchanged(self, monkeypatch):
        from nlsq.gui_qt.adapters.model_adapter import parse_custom_model_string

        monkeypatch.delenv("NLSQ_STRICT_MODELS", raising=False)
        func, _ = parse_custom_model_string(GZIP, "m")
        assert func(1.0, 2.0) == 2.0


class TestS2ExportInjection:
    @pytest.fixture
    def page(self, app_state, qtbot):
        import numpy as np

        from nlsq.gui_qt.pages.export import ExportPage

        app_state.set_data(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        app_state.set_model("builtin", {"name": "linear"})
        p = ExportPage(app_state)
        qtbot.addWidget(p)
        return p

    def test_hostile_workflow_rejected_not_emitted(self, page, tmp_path):
        marker = tmp_path / "PWNED"
        page._app_state.state.workflow = (
            f'auto"\nopen({str(marker)!r}, "w").write("x")  #'
        )
        with pytest.raises(ValueError, match="workflow"):
            page.generate_python_code()
        assert not marker.exists()

    def test_known_workflow_emitted_with_repr(self, page):
        page._app_state.state.workflow = "auto"
        assert "workflow = 'auto'" in page.generate_python_code()

    def test_hostile_builtin_model_name_rejected(self, page):
        page._app_state.state.model_config = {"name": "linear\nimport os"}
        with pytest.raises(ValueError, match="identifier"):
            page.generate_python_code()

    def test_hostile_custom_function_name_rejected(self, page):
        st = page._app_state.state
        st.model_type = "custom"
        st.model_config = {"code": GOOD, "function": "m; import os"}
        with pytest.raises(ValueError, match="identifier"):
            page.generate_python_code()

    def test_config_adapter_rejects_unknown_workflow(self):
        from nlsq.gui_qt.adapters.config_adapter import _parse_fitting_section
        from nlsq.gui_qt.session_state import SessionState

        with pytest.raises(ValueError, match="workflow"):
            _parse_fitting_section(
                {"fitting": {"workflow": 'auto"\nimport os #'}}, SessionState()
            )


class TestS7EnvDump:
    def test_allowlist_and_redaction(self):
        from nlsq.gui_qt import _env_dump_items

        env = {
            "QT_QPA_PLATFORM": "offscreen",
            "JAX_PLATFORMS": "cpu",
            "NLSQ_API_TOKEN": "s3cr3t",
            "GOOGLE_API_KEY": "leak1",  # pragma: allowlist secret
            "LANGCHAIN_API_KEY": "leak2",  # pragma: allowlist secret
            "GOOGLE_APPLICATION_CREDENTIALS": "leak3",
            "HOME": "/home/x",
        }
        dump = dict(_env_dump_items(env))
        assert dump["QT_QPA_PLATFORM"] == "offscreen"
        assert dump["JAX_PLATFORMS"] == "cpu"
        assert dump["NLSQ_API_TOKEN"] == "<redacted>"
        assert not {"GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "HOME"} & dump.keys()
        assert "leak3" not in str(dump)
