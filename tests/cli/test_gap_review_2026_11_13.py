"""Attribute-by-string validator bypasses, re-exported model helpers, audit log
fallback, and the global log level for late loggers."""

import logging
from pathlib import Path

import pytest

from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import AuditLogger, validate_model

WRAPS = """
import functools
got = []
class Box:
    def __setattr__(self, k, v):
        got.append(v)
def m(x, a):
    return a * x
functools.update_wrapper(Box(), m, assigned=("__glo" + "bals__",), updated=())
"""

MATCH = """
from collections.abc import Callable
class C(Callable):
    __match_args__ = ("__glo" + "bals__",)
    def __call__(self):
        return None
    @classmethod
    def __subclasshook__(cls, other):
        return True
def m(x, a):
    return a * x
match m:
    case C(g):
        pass
"""

PLAIN_WRAPS = """
import functools
def deco(f):
    @functools.wraps(f)
    def inner(*a, **k):
        return f(*a, **k)
    return inner
@deco
def m(x, a):
    return a * x
"""


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("source", [WRAPS, MATCH])
def test_attribute_by_string_hooks_are_rejected(tmp_path, source, strict):
    f = tmp_path / "m.py"
    f.write_text(source)
    assert not validate_model(f, strict=strict).is_valid


def test_plain_wraps_is_still_accepted(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(PLAIN_WRAPS)
    assert validate_model(f, strict=False).is_valid


def test_reexported_builtin_is_not_given_helpers(tmp_path, monkeypatch):
    from nlsq.core.functions import gaussian

    monkeypatch.chdir(tmp_path)
    before = getattr(gaussian, "estimate_p0", None)
    f = tmp_path / "reexport.py"
    f.write_text(
        "from nlsq.core.functions import gaussian\n"
        "def estimate_p0(x, y):\n    return [9.0, 9.0, 9.0]\n"
    )
    ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": "gaussian"}
    )
    assert getattr(gaussian, "estimate_p0", None) is before


def test_audit_logger_survives_unwritable_location(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    AuditLogger(log_file=blocker / "sub" / "audit.log")  # parent is a file


def test_global_level_applies_to_loggers_created_later():
    from nlsq.utils import logging as nl

    nl.set_global_level(logging.ERROR)
    try:
        late = nl.get_logger("late_created_logger_for_test")
        assert late.logger.level == logging.ERROR
    finally:
        nl._global_level = None
        nl.set_global_level(logging.WARNING)
        nl._global_level = None
        nl._loggers.pop("late_created_logger_for_test", None)
