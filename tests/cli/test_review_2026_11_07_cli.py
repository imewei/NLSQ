"""Regression tests for round-7 validator bypasses (annotations, pickle readers)."""

import shutil
from pathlib import Path

import pytest

from nlsq.cli.model_validation import validate_model

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/scripts/10_cli-commands/models"
TAIL = "\ndef model(x, a, b):\n    return a * x + b\n"


def _valid(tmp_path, monkeypatch, src: str) -> bool:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    return validate_model(f).is_valid


# Benign strings: the validator only parses them, nothing is ever executed.
BLOCKED = [
    # V1: string annotations are eval'd by singledispatch.register / get_type_hints
    "def h(o: 'f(1)'):\n    pass\n",
    "def h(o) -> 'f(1)':\n    pass\n",
    "v: 'f(1)' = 1\n",
    "def h(o: list['f(1)']):\n    pass\n",
    "def h(o: 'f(1)' | 'B'):\n    pass\n",
    "async def h(o: 'f(1)'):\n    pass\n",
    "def h(*a: 'f(1)', **k: 'f(1)'):\n    pass\n",
    "def h(o: 'not valid python ('):\n    pass\n",
    "def h(o: 'x.__class__'):\n    pass\n",
    "def h(o: 'exec'):\n    pass\n",
    "import typing\ntyping.get_type_hints(model)\n",
    "from typing import ForwardRef\n",
    "import typing\ntyping.ForwardRef('int')\n",
    "import typing\ntyping._eval_type(t, {}, {})\n",
    "import typing\ntyping.evaluate_forward_ref(r)\n",
    "import annotationlib\n",
    "f = model\nf.__annotations__ = {}\n",
    "x = model.__annotate__\n",
    "def f(fn):\n    return get_annotations(fn)\n",
    "def f(fn):\n    return sig(fn, eval_str=True)\n",
    # V2 / V3: pickle-capable numpy readers (positional allow_pickle)
    "from numpy.lib.format import read_array\nread_array(fp, True)\n",
    "import numpy as np\nnp.lib.format.read_array(fp, True)\n",
    "from numpy.lib.npyio import NpzFile\nNpzFile(fid, False, True)\n",
    "import numpy as np\nnp.lib.npyio.NpzFile(fid, False, True)\n",
    "import numpy as np\nnp.lib.format.read_array_header_1_0(fp)\n",
    "from numpy.lib.format import read_array as r\n",
    # sweep: other deserialisers / private modules / writers / network
    "from jax.experimental import serialize_executable\n",
    "import jax.experimental.serialize_executable\n",
    "import jax.experimental.array_serialization\n",
    "import jax\njax.export.deserialize(blob)\n",
    "from jax.experimental.compilation_cache import compilation_cache\n",
    "import jax._src.config\n",
    "from jax._src import pickle_util\n",
    "from numpy.lib import _format_impl\n",
    "from numpy.lib._npyio_impl import NpzFile\n",
    "import numpy as np\nnp._core.arrayprint\n",
    "import numpy as np\nnp.DataSource().open('http://example.invalid')\n",
    "from numpy.testing._private.extbuild import build_and_import_extension\n",
    "import jax\njax.distributed.initialize()\n",
    "import jax\njax.profiler.start_trace('d')\n",
    "import jax\njax.profiler.save_device_memory_profile('f')\n",
    "import scipy.sparse as sp\nsp.save_npz('f.npz', m)\n",
    "from scipy.io import netcdf_file\n",
    "from scipy.io import FortranFile\n",
    "breakpoint()\n",
]

ALLOWED = [
    # forward references made only of names/attributes/subscripts
    "def h(o: 'jax.Array', p: 'list[int] | None' = None) -> 'np.ndarray':\n    pass\n",
    # annotations that are real names/attributes stay fine
    "import numpy as np\nfrom typing import Any\n"
    "def h(o: np.ndarray, p: Any = None, *a: int) -> float:\n    return 1.0\n",
    "from __future__ import annotations\ndef h(o: int) -> int:\n    return o\n",
    "v: int = 1\nw: list[int] = []\n",
    "import numpy as np\nd = np.load('x.npy')\n",
    "import numpy as np\nd = np.load('x.npy', 'r')\n",
    "import json\nd = json.load(fh)\n",
    "from scipy import signal\nfrom jax.scipy import signal as s2\n",
    "from jax.numpy import trace\nfrom numpy import trace as t2\n",
    "import jax\nimport jax.numpy as jnp\nfrom jax import lax\n"
    "from jax.scipy.special import erf\nfrom scipy.special import gammaln\n"
    "import scipy.signal\nimport scipy.optimize\nimport numpy.linalg\n",
    "import jax\ny = jax.jit(lambda x: x)\n",
]


@pytest.mark.parametrize("src", BLOCKED)
def test_round7_payloads_rejected(tmp_path, monkeypatch, src):
    assert not _valid(tmp_path, monkeypatch, src)


@pytest.mark.parametrize("src", ALLOWED)
def test_legit_code_still_validates(tmp_path, monkeypatch, src):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    result = validate_model(f)
    assert result.is_valid, result.violations


@pytest.mark.parametrize(
    "src",
    [
        *sorted(EXAMPLES.glob("*.py")),
        ROOT / "nlsq/cli/templates/custom_model_template.py",
    ],
    ids=lambda p: p.name,
)
def test_shipped_models_still_validate(src, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / src.name
    shutil.copyfile(src, f)
    result = validate_model(f)
    assert result.is_valid, result.violations
