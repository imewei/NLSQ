"""Attribute names that collide with newly blocked stdlib modules stay usable.

A dataclass field called ``wave`` is a normal physics parameter; only the
module imports (``import wave``) are dangerous. Payloads are only parsed.
"""

import pytest

from nlsq.cli.model_validation import validate_model

TAIL = "\ndef m(x, a):\n    return a * x\n"

ATTR_PREFIX = (
    "from dataclasses import dataclass\n"
    "import jax.numpy as jnp\n"
    "@dataclass\n"
    "class P:\n"
    "    {name}: float = 2.0\n"
    "C = P()\n"
    "K = C.{name}\n"
)

NAMES = ["wave", "pipes", "xml"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("strict", [False, True])
def test_attribute_named_like_blocked_module_validates(tmp_path, name, strict):
    f = tmp_path / "m.py"
    f.write_text(ATTR_PREFIX.format(name=name) + TAIL)
    result = validate_model(f, strict=strict)
    assert result.is_valid, result.violations


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "form",
    [
        "import {name}\n",
        "import {name} as w\n",
        "from os import {name}\n",
        "from typing import {name}\n",
        "import importlib\nw = importlib.import_module('{name}')\n",
    ],
)
def test_module_imports_still_blocked(tmp_path, name, strict, form):
    f = tmp_path / "m.py"
    f.write_text(form.format(name=name) + TAIL)
    assert not validate_model(f, strict=strict).is_valid
