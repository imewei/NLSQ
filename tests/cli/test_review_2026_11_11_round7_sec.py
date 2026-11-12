"""Round-7 security regressions: underscore C-module validator bypass, HDF5 bombs.

Validator verdicts only: payload sources are parsed, never executed.
"""

from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.errors import DataLoadError
from nlsq.cli.model_validation import validate_model

MODEL = "\ndef model(x, a, b):\n    return a * jnp.exp(-b * x)\n"

PAYLOADS = [
    # the reported PoC: libc system() through the C twin of ctypes
    "import _ctypes\nimport jax.numpy as jnp\nh = _ctypes.dlopen(None)\n"
    'p = _ctypes.dlsym(h, "system")\n_ctypes.call_function(p, (b"id",))\n',
    "import _imp\nimport jax.numpy as jnp\n_imp.create_builtin(None)\n",
    "from _ctypes import dlopen\nimport jax.numpy as jnp\n",
    "import _ctypes as c\nimport jax.numpy as jnp\n",
    "import _frozen_importlib\nimport jax.numpy as jnp\n",
    "import _frozen_importlib_external\nimport jax.numpy as jnp\n",
    "import _posixsubprocess\nimport jax.numpy as jnp\n",
    "import _thread\nimport jax.numpy as jnp\n",
    "import _io\nimport jax.numpy as jnp\n",
    "import _socket\nimport jax.numpy as jnp\n",
    "import _winapi\nimport jax.numpy as jnp\n",
    "import posix\nimport jax.numpy as jnp\n",
    "import nt\nimport jax.numpy as jnp\n",
    # other underscore C modules: rejected by the root-underscore rule
    "import _pickle\nimport jax.numpy as jnp\n",
    "import _operator\nimport jax.numpy as jnp\n",
    "from _operator import attrgetter\nimport jax.numpy as jnp\n",
    "import _collections_abc\nimport jax.numpy as jnp\n",
    "import _functools\nimport jax.numpy as jnp\n",
    "import _signal\nimport jax.numpy as jnp\n",
    "import _ctypes.sub\nimport jax.numpy as jnp\n",
    "from _ctypes.sub import x\nimport jax.numpy as jnp\n",
    # relative import of an underscore module: module is None-rooted, name private
    "from . import _ctypes\nimport jax.numpy as jnp\n",
]

LEGIT = [
    "from __future__ import annotations\nimport jax.numpy as jnp\n",
    "import jax\nimport jax.numpy as jnp\nimport numpy as np\n",
    "from jax import jit\nimport jax.numpy as jnp\n",
]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("payload", PAYLOADS)
def test_underscore_module_payloads_rejected(tmp_path, payload, strict):
    f = tmp_path / "m.py"
    f.write_text(payload + MODEL)
    res = validate_model(f, strict=strict)
    assert not res.is_valid, payload


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("header", LEGIT)
def test_legit_models_still_pass(tmp_path, header, strict):
    f = tmp_path / "m.py"
    f.write_text(header + MODEL)
    res = validate_model(f, strict=strict)
    assert res.is_valid, res.violations


def test_shipped_templates_still_pass():
    root = Path(__file__).resolve().parents[2] / "nlsq" / "cli" / "templates"
    files = list(root.rglob("*.py"))
    for p in files:
        res = validate_model(p)
        assert res.is_valid, (p, res.violations)


# ---------------------------------------------------------------- HDF5 loader

h5py = pytest.importorskip("h5py")


def _cfg(xp="x", yp="y"):
    return {"format": "hdf5", "hdf5": {"x_path": xp, "y_path": yp}}


@pytest.mark.serial
def test_hdf5_sparse_chunk_bomb_rejected(tmp_path):
    """(1e8,) gzip dataset with ONE written chunk must not materialise 800MB."""
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "bomb.h5"
    with h5py.File(p, "w") as f:
        for name in ("x", "y"):
            d = f.create_dataset(
                name,
                shape=(100_000_000,),
                dtype="f8",
                chunks=(1024,),
                compression="gzip",
            )
            d[:1024] = 1.0
    with pytest.raises(DataLoadError, match=r"unallocated|sparse|chunk"):
        DataLoader().load(p, _cfg())


@pytest.mark.serial
def test_hdf5_compressed_real_data_still_loads(tmp_path):
    from nlsq.cli.data_loaders import DataLoader

    p = tmp_path / "ok.h5"
    n = 200_001  # not a multiple of the chunk size: edge chunk
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "x", data=np.linspace(0, 1, n), chunks=(4096,), compression="gzip"
        )
        f.create_dataset("y", data=np.zeros(n), chunks=(4096,), compression="gzip")
    x, y, _ = DataLoader().load(p, _cfg())
    assert len(x) == n and len(y) == n
