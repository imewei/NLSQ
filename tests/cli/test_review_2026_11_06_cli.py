"""Regression tests for round-6 CLI fixes (np.load gadgets, import collisions, None p0/bounds)."""

import numpy as np
import pytest

from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import validate_model

TAIL = "\ndef model(x, a, b):\n    return a * x + b\n"
NP = "import numpy as np\nimport functools\n"


def _valid(tmp_path, monkeypatch, src: str) -> bool:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    return validate_model(f).is_valid


# Benign strings: the validator only parses them, nothing is executed.
LOAD_BLOCKED = [
    NP + "d = np.load('s.npy', None, True)\n",
    NP + "d = np.load(f, **{'allow_pickle': True})\n",
    NP + "d = np.load(f, *(None, True))\n",
    NP + "L = np.load\nd = L(f, None, True)\n",
    "from numpy import load as L\nd = L('s.npy')\n",
    NP + "d = functools.partial(np.load, None, None, True)(f)\n",
    NP + "from scipy.io import mmio\nmmio.mmwrite('o.mtx', m)\n",
]
LOAD_ALLOWED = [
    NP + "d = np.load('x.npy')\n",
    NP + "d = np.load('x.npy', 'r')\n",
    "import json\nd = json.load(fh)\n",
]


@pytest.mark.parametrize("src", LOAD_BLOCKED)
def test_load_pickle_gadgets_rejected(tmp_path, monkeypatch, src):
    assert not _valid(tmp_path, monkeypatch, src)


@pytest.mark.parametrize("src", LOAD_ALLOWED)
def test_plain_load_calls_allowed(tmp_path, monkeypatch, src):
    assert _valid(tmp_path, monkeypatch, src)


@pytest.mark.parametrize(
    "src",
    [
        "from scipy import signal\n",
        "from jax.scipy import signal\n",
        "from jax.numpy import trace\n",
        "from numpy import trace\n",
    ],
)
def test_numeric_symbols_colliding_with_blocked_modules_allowed(
    tmp_path, monkeypatch, src
):
    assert _valid(tmp_path, monkeypatch, src)


@pytest.mark.parametrize(
    "src",
    [
        "from scipy.io.wavfile import sys\n",
        "from jax._src.config import logging\n",
        "from mymodels import signal\n",
        "import signal\n",
    ],
)
def test_colliding_names_from_other_roots_still_blocked(tmp_path, monkeypatch, src):
    assert not _valid(tmp_path, monkeypatch, src)


@pytest.mark.parametrize("helper", ["estimate_p0", "bounds"])
def test_none_from_helper_means_no_default(tmp_path, monkeypatch, helper):
    """R6-CLI-3: None from bounds()/estimate_p0() passes through unchanged."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    args = "xdata, ydata" if helper == "estimate_p0" else ""
    f.write_text(TAIL + f"\ndef {helper}({args}):\n    return None\n")
    model = ModelRegistry().get_model(
        str(f), {"type": "custom", "path": str(f), "function": "model"}
    )
    fn = getattr(model, helper)
    assert (fn(np.zeros(3), np.zeros(3)) if args else fn()) is None
