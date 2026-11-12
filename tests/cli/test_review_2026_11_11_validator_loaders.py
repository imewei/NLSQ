"""Regression tests: help()/numpy readers/jax.config validator gaps, bool column indices.

Validator verdicts only: payload sources are parsed, never executed.
"""

import numpy as np
import pytest

from nlsq.cli.errors import DataLoadError
from nlsq.cli.model_validation import validate_model

TAIL = "\ndef model(x, a, b):\n    return a * x + b\n"

BLOCKED = [
    # site/pydoc builtins: help("mod") imports any module by string
    'help("os")\n',
    'help("antigravity")\n',
    "license()\n",
    "credits()\n",
    "copyright()\n",
    "exit()\n",
    "quit()\n",
    # numpy readers: network (DataSource) + CWD cache writes
    'import numpy as np\nnp.genfromtxt("http://example.invalid/x.txt")\n',
    'import numpy as np\nnp.loadtxt("http://example.invalid/x.txt")\n',
    'import numpy as np\nnp.fromregex("x.txt", r"a", float)\n',
    'import numpy as np\nnp.fromfile("x.bin")\n',
    'from numpy import genfromtxt\ngenfromtxt("x")\n',
    # jax runtime config: process-wide mutation + arbitrary-dir file writes
    'import jax\njax.config.update("jax_dump_ir_to", "d")\n',
    'import jax\njax.config.update("jax_compilation_cache_dir", "d")\n',
    'import jax as j\nj.config.update("jax_dump_ir_to", "d")\n',
    'from jax import config\nconfig.update("jax_dump_ir_to", "d")\n',
    # jax.config reached through a VALUE alias of the module (not an import alias)
    'import jax\nJ = jax\nJ.config.update("jax_enable_x64", False)\n',
    'import jax\n(k := jax).config.update("jax_enable_x64", False)\n',
    'import jax\ndef helper(j=jax):\n    j.config.update("jax_enable_x64", False)\n',
    'import jax\n(j3,) = (jax,)\nj3.config.update("jax_enable_x64", False)\n',
    'import jax\nf = lambda m: m.config\nf(jax).update("jax_enable_x64", False)\n',
    # numpy.ma.mrecords.fromtextfile reads any text file
    'import numpy.ma.mrecords as mr\nmr.fromtextfile("x.txt")\n',
    'import numpy as np\nnp.ma.mrecords.fromtextfile("x.txt")\n',
    'from numpy.ma.mrecords import fromtextfile\nfromtextfile("x.txt")\n',
    'from numpy.ma import mrecords\nmrecords.fromtextfile("x.txt")\n',
    # numpy.ma.mrecords.openfile opens any file (round 4): all reach forms
    'from numpy.ma import mrecords\nmrecords.openfile("/etc/hostname").readline()\n',
    'import numpy.ma as ma\nma.mrecords.openfile("/etc/hostname").readline()\n',
    'import numpy as np\nnp.ma.mrecords.openfile("/etc/hostname").readline()\n',
    'import numpy.ma as ma\nmatch ma:\n    case object(mrecords=r):\n        r.openfile("/etc/hostname")\n',
    'from numpy.ma.mrecords import openfile\nopenfile("/etc/hostname")\n',
    # MatchClass keyword attributes get the same any-base .config rule
    'import jax\nmatch jax:\n    case object(config=c):\n        c.update("jax_enable_x64", False)\n',
    # star import exposes jax.config as a bare name
    'from jax import *\nconfig.update("jax_enable_x64", False)\n',
    "from jax.config import *\n",
    "from jax.config import config\n",
    # module-path rule: import jax.config as c
    'import jax.config as c\nc.update("jax_enable_x64", False)\n',
]


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
@pytest.mark.parametrize("src", BLOCKED)
def test_blocked(tmp_path, src, strict):
    f = tmp_path / "m.py"
    f.write_text(src + TAIL)
    assert not validate_model(f, strict=strict).is_valid


def test_jax_numpy_star_import_still_valid(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "from jax.numpy import *\n\ndef model(x, a):\n    return a * exp(-x)\n"
    )
    assert validate_model(f).is_valid


def test_self_config_attribute_still_valid(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "class Model:\n"
        "    def __init__(self):\n        self.config = {'k': 1.0}\n"
        "    def __call__(self, x, a):\n        return a * x * self.config['k']\n"
        "m = Model()\n"
        "def model(x, a):\n    return m(x, a)\n"
    )
    assert validate_model(f).is_valid


def test_benign_jax_model_still_valid(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "import jax\nimport jax.numpy as jnp\n"
        "@jax.jit\ndef model(x, a, b):\n    return a * jnp.exp(-b * x)\n"
    )
    assert validate_model(f).is_valid


@pytest.mark.parametrize("fmt", ["csv", "ascii"])
def test_bool_column_index_rejected(tmp_path, fmt):
    from nlsq.cli.data_loaders import DataLoader

    # "ascii" is the loader's real format name (extension .txt); "txt" is unknown
    p = tmp_path / ("d.txt" if fmt == "ascii" else f"d.{fmt}")
    p.write_text("1,2,3\n4,5,6\n7,8,9\n")
    cfg = {"format": fmt, "columns": {"x": True, "y": 1}}
    with pytest.raises(DataLoadError, match="boolean"):
        DataLoader().load(str(p), cfg)


def test_validate_data_rejects_multidim_y():
    from nlsq.cli.data_loaders import DataLoader

    with pytest.raises(DataLoadError):
        DataLoader()._validate_data(np.ones(3), np.ones((3, 2)), None, "f", {})
