"""Tests for the opt-in strict allowlist mode of the custom-model validator.

Payload tests only call ``validate_model`` (AST inspection); nothing is executed
except the explicitly benign models loaded in the registry wiring tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlsq.cli.errors import ModelError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.model_validation import (
    ALLOWLIST_MODULES,
    AllowlistVisitor,
    DangerousPatternVisitor,
    validate_model,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_MODELS = sorted((REPO / "examples/scripts/10_cli-commands/models").glob("*.py"))
TEMPLATE = REPO / "nlsq/cli/templates/custom_model_template.py"


def _check(tmp_path: Path, src: str, *, strict: bool = True):
    f = tmp_path / "m.py"
    f.write_text(src)
    return validate_model(f, strict=strict)


# ---------------------------------------------------------------- positives
@pytest.mark.parametrize("model", [*EXAMPLE_MODELS, TEMPLATE], ids=lambda p: p.name)
def test_shipped_models_pass_strict(model: Path):
    result = validate_model(model, strict=True)
    assert result.is_valid, result.violations


def test_shipped_corpus_found():
    assert len(EXAMPLE_MODELS) == 4 and TEMPLATE.exists()


PLAIN_MODELS = {
    "numpy_jnp_special": (
        "import numpy as np\nimport jax\nimport jax.numpy as jnp\n"
        "from scipy import special\nfrom scipy.special import gammaln\n"
        "import scipy.stats\n"
        "@jax.jit\n"
        "def model(x, a, b):\n"
        "    return a * jnp.exp(-b * x) + special.gamma(1.5) + gammaln(2.0)\n"
        "def helper(x):\n    return np.linalg.norm(x) + jax.lax.stop_gradient(x)\n"
    ),
    "typing_dataclasses": (
        "from __future__ import annotations\nimport math\nimport functools\n"
        "import collections.abc\nfrom dataclasses import dataclass\n"
        "from typing import TYPE_CHECKING\n"
        "@dataclass\nclass P:\n    a: float = 1.0\n"
        "def model(x: collections.abc.Sequence, a: float) -> float:\n"
        "    return math.sqrt(a)\n"
    ),
}


@pytest.mark.parametrize("name", PLAIN_MODELS)
def test_plain_models_pass_strict(tmp_path, name):
    result = _check(tmp_path, PLAIN_MODELS[name])
    assert result.is_valid, result.violations


def test_allowlist_matches_spec():
    assert {
        "math", "cmath", "functools", "typing", "collections.abc", "dataclasses",
        "numpy", "jax", "jax.numpy", "jax.scipy", "jax.lax", "scipy.special",
        "scipy.stats", "scipy.constants",
    } <= ALLOWLIST_MODULES  # fmt: skip


def test_strict_visitor_is_denylist_subclass():
    assert issubclass(AllowlistVisitor, DangerousPatternVisitor)


# ---------------------------------------------------------------- negatives
NEGATIVE = {
    # r8-sec Part A
    "A1_typing_sys": "import typing\nreached = typing.sys\n",
    "A1_scipy_io_os": "import scipy.io.wavfile\nreached = scipy.io.wavfile.os\n",
    "A1_jax_subprocess": "import jax\nreached = jax.version.subprocess\n",
    "A1_dataclasses_inspect": "import dataclasses\nreached = dataclasses.inspect\n",
    "A1_numpy_private_pickle": (
        "import numpy\nreached = numpy.lib._format_impl.pickle\n"
    ),
    "A2_match_class": (
        "def f(v):\n    match v:\n        case object(__self__=m):\n"
        "            return m\n"
    ),
    "A3_string_formatter": (
        "import string\nreached = string.Formatter().get_field('0.__self__', [len], {})\n"
    ),
    "A4_dunder_self": "reached = len.__self__\n",
    "A5_wavfile_write": (
        "import scipy.io.wavfile\nscipy.io.wavfile.write('/x', 1, [0])\n"
    ),
    "A6_jax_profiler_trace": "import jax\njax.profiler.trace('/x')\n",
    # prototype holes
    "hole_bare_scipy": "import scipy\nscipy.io.wavfile.write('/x', 1, [0])\n",
    "hole_bare_jax": "import jax\njax.config.update('jax_enable_x64', True)\n",
    "hole_aliased_open": "o = open\no('/x', 'w')\n",
    # misc
    "import_os": "import os\n",
    "import_cprofile": "import cProfile\n",
    "string_annotation": "def f(x: 'len(x)'):\n    return x\n",
    "string_annotation_eval": "def f(x) -> 'int(x)':\n    return x\n",
    "np_load_positional_pickle": (
        "import numpy as np\nnp.load('f.npy', False, True)\n"
    ),
    "np_load_plain": "import numpy as np\nnp.load('f.npy')\n",
    "np_load_alias": "import numpy as np\nL = np.load\nL('f.npy')\n",
    "np_save": "import numpy as np\nnp.save('f.npy', [1])\n",
    "np_fromfile": "import numpy as np\nnp.fromfile('f')\n",
    "np_memmap": "import numpy as np\nnp.memmap('f')\n",
    "np_ctypeslib": "import numpy as np\nnp.ctypeslib.load_library('x', '.')\n",
    "np_f2py": "import numpy as np\nnp.f2py\n",
    "jax_experimental": (
        "import jax\nfrom jax.experimental import io_callback\n"
        "jax.experimental.io_callback\n"
    ),
    "jax_config_update": "import jax\njax.config.update('x', 1)\n",
    "jax_debug": "import jax\njax.debug.print('x')\n",
    "jax_export": "import jax\njax.export.export\n",
    "scipy_wavfile_write": (
        "from scipy.io import wavfile\nwavfile.write('/x', 1, [0])\n"
    ),
    "from_scipy_import_io": "from scipy import io\n",
    "from_scipy_optimize": "from scipy import optimize\n",
    "from_numpy_format_read_array": "from numpy.lib.format import read_array\n",
    "from_jax_import_config": "from jax import config\n",
    "o_equals_open": "o = open\n",
    "bare_eval_name": "f = eval\n",
    "bare_getattr_name": "g = [getattr]\n",
    "bare_globals_name": "x = globals\n",
    "relative_import": "from . import sibling\n",
    "wildcard_import": "from numpy import *\n",
    "module_alias_then_config": ("import jax\nj = jax\nj.config.update('x', 1)\n"),
    "module_as_argument": "import jax\ndef f(m):\n    return m\nf(jax)\n",
    "submodule_alias_attr": "import jax\nm = jax.numpy\nm.load('x')\n",
    "import_pickle": "import pickle\n",
    "import_pandas": "import pandas\n",
    "import_enum": "import enum\n",
    "import_itertools": "import itertools\n",
    "import_scipy_optimize": "import scipy.optimize\n",
}


@pytest.mark.parametrize("name", NEGATIVE)
def test_payload_rejected_in_strict(tmp_path, name):
    result = _check(tmp_path, NEGATIVE[name])
    assert not result.is_valid, f"{name} was accepted"


def test_violation_messages_mention_strict_allowlist(tmp_path):
    result = _check(tmp_path, "import scipy\nscipy.io\n")
    assert any("strict allowlist" in v for v in result.violations)


# ------------------------------------------------------------ default mode
def test_default_mode_unchanged_for_scipy_optimize(tmp_path):
    src = (
        "import scipy.optimize\nimport numpy as np\n"
        "def model(x, a):\n    return a * np.asarray(x)\n"
    )
    assert _check(tmp_path, src, strict=False).is_valid
    assert not _check(tmp_path, src, strict=True).is_valid


def test_default_mode_ignores_strict_only_rules(tmp_path):
    # `import scipy.optimize` is a strict-only block; the default denylist accepts it.
    src = "import scipy.optimize\nimport numpy as np\n"
    assert _check(tmp_path, src, strict=False).is_valid
    assert not _check(tmp_path, src, strict=True).is_valid


def test_jax_config_blocked_in_default_mode(tmp_path):
    # jax.config mutates process-wide state and can write files (jax_dump_ir_to).
    src = "import jax\njax.config.update('jax_enable_x64', True)\n"
    assert not _check(tmp_path, src, strict=False).is_valid


# ------------------------------------------------- registry wiring (d), (e)
SCIPY_OPT_MODEL = "import scipy.optimize\ndef f(x, a):\n    return a * x\n"


def _registry_cfg(path: Path, **extra):
    return {"type": "custom", "path": str(path), "function": "f", **extra}


@pytest.fixture
def model_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("NLSQ_STRICT_MODELS", raising=False)
    f = tmp_path / "m.py"
    f.write_text(SCIPY_OPT_MODEL)
    return f


def test_config_strict_true_enables_allowlist(model_env):
    with pytest.raises(ModelError, match="strict allowlist"):
        ModelRegistry().get_model(str(model_env), _registry_cfg(model_env, strict=True))


def test_config_default_loads_same_model(model_env):
    assert callable(ModelRegistry().get_model(str(model_env), _registry_cfg(model_env)))


def test_trusted_bypasses_strict(model_env, caplog):
    with caplog.at_level("WARNING"):
        model = ModelRegistry().get_model(
            str(model_env), _registry_cfg(model_env, strict=True, trusted=True)
        )
    assert callable(model)
    assert any("strict allowlist" in r.getMessage() for r in caplog.records)


def test_env_var_turns_strict_on(model_env, monkeypatch):
    monkeypatch.setenv("NLSQ_STRICT_MODELS", "1")
    with pytest.raises(ModelError, match="strict allowlist"):
        ModelRegistry().get_model(str(model_env), _registry_cfg(model_env))


def test_env_var_unset_or_zero_stays_default(model_env, monkeypatch):
    monkeypatch.setenv("NLSQ_STRICT_MODELS", "0")
    assert callable(ModelRegistry().get_model(str(model_env), _registry_cfg(model_env)))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_non_bool_strict_rejected(model_env, value):
    with pytest.raises(ModelError, match="'strict' must be true or false"):
        ModelRegistry().get_model(
            str(model_env), _registry_cfg(model_env, strict=value)
        )
