"""R4 CC-04 / CC-11: DiagnosticsConfig level strings and integral windows."""

import pytest

from nlsq.diagnostics.types import DiagnosticLevel, DiagnosticsConfig


@pytest.mark.parametrize("text", ["full", "FULL", "Full"])
def test_level_string_is_coerced(text):
    assert DiagnosticsConfig(level=text).level is DiagnosticLevel.FULL


def test_level_bogus_string_raises_listing_valid_levels():
    with pytest.raises(ValueError, match="basic"):
        DiagnosticsConfig(level="bogus")


def test_level_wrong_type_raises():
    with pytest.raises(TypeError):
        DiagnosticsConfig(level=3)


@pytest.mark.parametrize("name", ["gradient_window_size", "stagnation_window"])
@pytest.mark.parametrize("bad", [2.9, True, float("nan")])
def test_windows_must_be_integral(name, bad):
    with pytest.raises(ValueError, match=name):
        DiagnosticsConfig(**{name: bad})


def test_integral_float_window_accepted():
    assert DiagnosticsConfig(stagnation_window=10.0).stagnation_window == 10.0
