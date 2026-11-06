"""Regression tests for round-8 validator patches (attribute reachability, match, string)."""

import shutil
from pathlib import Path

import pytest

from nlsq.cli.model_validation import validate_model

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/scripts/10_cli-commands/models"
TAIL = "\ndef model(x, a, b):\n    return a * x + b\n"


def _result(tmp_path, monkeypatch, src: str):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    return validate_model(f)


# Validator verdict only: these are parsed, never executed.
BLOCKED = [
    # A1: public attribute chains reaching blocked modules
    "import typing\nreached = typing.sys\n",
    "import jax\nreached = jax.version.subprocess\n",
    "import dataclasses\nreached = dataclasses.inspect\n",
    "import enum\nreached = enum.bltns\n",
    "import scipy\nreached = scipy.io.wavfile.os\n",
    "import numpy\nreached = numpy.lib._format_impl.pickle\n",
    "import numpy\nnumpy.lib._format_impl.pickle.loads(b'x')\n",
    "import numpy as np\nnp._NoValue\n",
    # A2: match/case keyword attributes are plain strings
    "match len:\n    case object(__self__=m):\n        reached = m\n",
    "match x:\n    case object(_private=m):\n        pass\n",
    "match x:\n    case object(os=m):\n        pass\n",
    "match x:\n    case object(system=m):\n        pass\n",
    "match x:\n    case exec(y):\n        pass\n",
    # A3: string.Formatter field resolution
    "import string\nstring.Formatter().get_field('0.x', [1], {})\n",
    "from string import Formatter\n",
    "f = fmt\nf.vformat('{0.x}', [1], {})\n",
    "f = fmt\nf.format_field(1, '')\n",
    "f = fmt\nf.convert_field(1, 'r')\n",
    # A4: introspection dunders / closures
    "reached = len.__self__\n",
    "f = model\nf.__setattr__('a', 1)\n",
    "f = model\nf.__reduce_ex__(2)\n",
    "f = model\nreached = f.__closure__[0].cell_contents\n",
    "f = model\nreached = f.__wrapped__\n",
    "f = model\nreached = f.__func__\n",
    "f = model\nreached = f.__defaults__\n",
    # A5: scipy.io is no longer exempt
    "import scipy\nscipy.io.wavfile.write('p', 8000, d)\n",
    "from scipy import io\n",
    "import scipy.io as sio\n",
    "from scipy.io import wavfile\n",
    "import scipy as sp\nsp.io.wavfile.write('p', 8000, d)\n",
    # A6: jax profiler writes trace directories
    "import jax\njax.profiler.trace('d')\n",
    "from jax import profiler\n",
    "from jax.profiler import trace\n",
    # private attributes on model-class instances are a measured compat cost
    "class M:\n    def __call__(self, x):\n        return self._k * x\n",
]

ALLOWED = [
    "import jax.numpy as jnp\ny = jnp.trace(a)\n",
    "import numpy as np\ny = np.trace(a)\n",
    "import jax\nfrom jax import lax\ny = lax.add(1, 2)\nz = jax.lax.mul(1, 2)\n",
    "import jax.numpy as jnp\ny = jnp.linalg.norm(a)\n",
    "from scipy.special import gammaln\nimport scipy.special\ny = scipy.special.erf(1)\n",
    "from scipy import signal\ny = signal.convolve(a, b)\n",
    "import scipy.signal\ny = scipy.signal.convolve(a, b)\n",
    "from jax.scipy import signal as s2\ny = s2.convolve(a, b)\n",
    "def f(x):\n    return x.signal\n",
    "def f(fn):\n    return fn.__name__, fn.__doc__\n",
    "import numpy as np\nv = np.__version__\n",
    "y = [1, 2].index(1)\nz = 'a b'.split()\n",
    "match x:\n    case int(real=r):\n        pass\n",
]


@pytest.mark.parametrize("src", BLOCKED)
def test_round8_payloads_rejected(tmp_path, monkeypatch, src):
    assert not _result(tmp_path, monkeypatch, src).is_valid


@pytest.mark.parametrize("src", ALLOWED)
def test_legit_code_still_validates(tmp_path, monkeypatch, src):
    result = _result(tmp_path, monkeypatch, src)
    assert result.is_valid, result.violations


@pytest.mark.parametrize(
    "src",
    [
        *sorted(EXAMPLES.glob("*.py")),
        ROOT / "nlsq/cli/templates/custom_model_template.py",
        *sorted((ROOT / "tests/cli/fixtures").glob("*model*.py")),
    ],
    ids=lambda p: p.name,
)
def test_shipped_models_still_validate(src, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / src.name
    shutil.copyfile(src, f)
    result = validate_model(f)
    assert result.is_valid, result.violations
