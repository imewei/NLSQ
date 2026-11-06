"""Syntactically valid JSON with the wrong root shape must not break fits."""

import json
from unittest.mock import mock_open, patch

import pytest

from nlsq.config import get_jacobian_mode

pytestmark = pytest.mark.serial


@pytest.mark.parametrize(
    "root", [None, [], ["jacobian_mode"], 1, "jacobian_mode", True]
)
def test_invalid_config_root_warns_and_uses_default(root, monkeypatch):
    monkeypatch.delenv("NLSQ_JACOBIAN_MODE", raising=False)
    with (
        patch("nlsq.config.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(root))),
        pytest.warns(UserWarning, match="config file"),
    ):
        assert get_jacobian_mode() == ("auto", "auto-default")


@pytest.mark.parametrize("mode", ["auto", "fwd", "rev"])
def test_valid_config_object_keeps_its_mode(mode, monkeypatch):
    monkeypatch.delenv("NLSQ_JACOBIAN_MODE", raising=False)
    with (
        patch("nlsq.config.os.path.exists", return_value=True),
        patch(
            "builtins.open", mock_open(read_data=json.dumps({"jacobian_mode": mode}))
        ),
    ):
        assert get_jacobian_mode() == (mode, "config file")
