"""Regression tests: default-mode validator bypasses and p0 length for defaulted params.

Validator payloads are only parsed, never executed.
"""

import numpy as np
import pytest

from nlsq.cli.errors import ConfigError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import validate_model
from nlsq.cli.workflow_runner import WorkflowRunner

TAIL = "\ndef m(x, a):\n    return a * x\n"

REJECTED = [
    # numpy/jax load with mmap_mode writes or truncates the file
    'import numpy as np\n_a = np.load("v.npy", mmap_mode="r+")\n_a[:] = -1\n',
    'import numpy as np\n_a = np.load("v.npy", "w+")\n',
    'import numpy\n_a = numpy.load("v.npy", "r+")\n',
    'import numpy as np\nmode = "r+"\n_a = np.load("v.npy", mode)\n',
    'import numpy as np\n_a = np.load("v.npy", mmap_mode="w+")\n',
    'import jax.numpy as jnp\n_a = jnp.load("v.npy", mmap_mode="r+")\n',
    'from numpy import load\n_a = load("v.npy", "w+")\n',
    # xml writers / URL-fetching parsers
    'import xml.etree.ElementTree as ET\nET.ElementTree(ET.Element("x")).write("v")\n',
    'import xml.sax\nxml.sax.parse("http://127.0.0.1:1/x", xml.sax.ContentHandler())\n',
    "from xml.dom import pulldom\n",
    "from xml.etree import ElementTree\n",
    # pstats / profiler dump_stats file writes
    'import pstats\npstats.Stats().dump_stats("v")\n',
    'def f(s):\n    s.dump_stats("v")\n',
    # other stdlib file/network/terminal side-effect modules
    "import mailbox\n",
    "import wave\n",
    "import aifc\n",
    "import sunau\n",
    "import termios\n",
    "import netrc\n",
    "import readline\n",
    "import syslog\n",
    "import cgitb\n",
    "import pipes\n",
    "import winreg\n",
]

ACCEPTED = [
    'import numpy as np\n_a = np.load("v.npy")\n',
    'import numpy as np\n_a = np.load("v.npy", "r")\n',
    'import numpy as np\n_a = np.load("v.npy", mmap_mode="c")\n',
    'import numpy as np\n_a = np.load("v.npy", mmap_mode=None)\n',
    "import math\nimport numpy as np\nimport jax.numpy as jnp\n",
]


@pytest.mark.parametrize("src", REJECTED)
@pytest.mark.parametrize("strict", [False, True])
def test_side_effect_payloads_rejected(tmp_path, src, strict):
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    assert not validate_model(f, strict=strict).is_valid


@pytest.mark.parametrize("src", ACCEPTED)
def test_plain_numeric_models_accepted(tmp_path, src):
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    result = validate_model(f)
    assert result.is_valid, result.violations


def _defaulted():
    def m(x, a, b, c=0.0):
        return a * x + b + c

    return m


def _params(p0, model):
    return WorkflowRunner()._extract_fit_params(
        {"fitting": {"p0": p0}}, np.arange(3.0), np.arange(3.0), model
    )[0]


@pytest.mark.parametrize("p0", [[1.0, 1.0], [1.0, 1.0, 0.5]])
def test_p0_accepts_lengths_between_required_and_total(p0):
    np.testing.assert_array_equal(_params(p0, _defaulted()), p0)


@pytest.mark.parametrize("p0", [[1.0], [1.0, 1.0, 1.0, 1.0]])
def test_p0_rejects_lengths_outside_range(p0):
    with pytest.raises(ConfigError, match=r"fitting\.p0"):
        _params(p0, _defaulted())


def test_p0_unchecked_for_varargs_model():
    def m(x, *params):
        return x

    np.testing.assert_array_equal(_params([1.0] * 5, m), [1.0] * 5)


def test_registry_helpers_accept_defaulted_param_lengths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "mod.py"
    f.write_text(
        "def m(x, a, b, c=0.0, *, scale=1.0):\n"
        "    return a * x + b + c\n"
        "def estimate_p0(x, y):\n"
        "    return [1.0, 1.0]\n"
        "def bounds():\n"
        "    return ([0.0, 0.0], [5.0, 5.0])\n"
    )
    model = ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": "m"}
    )
    assert model.estimate_p0(None, None) == [1.0, 1.0]
    assert model.bounds() == ([0.0, 0.0], [5.0, 5.0])


def test_registry_helpers_reject_wrong_lengths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "mod.py"
    f.write_text(
        "def m(x, a, b, c=0.0):\n"
        "    return a * x + b + c\n"
        "def estimate_p0(x, y):\n"
        "    return [1.0]\n"
        "def bounds():\n"
        "    return ([0.0] * 4, [5.0] * 4)\n"
    )
    model = ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": "m"}
    )
    with pytest.raises(ValueError, match="length 1"):
        model.estimate_p0(None, None)
    with pytest.raises(ValueError, match="length 4"):
        model.bounds()
