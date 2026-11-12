"""Regression tests: function-local names that shadow site builtins (exit, help, ...).

Validator verdicts only: payload sources are parsed, never executed.
"""

import pytest

from nlsq.cli.model_validation import validate_model

IMPORT = "import jax.numpy as jnp\n"

ACCEPTED = [
    # parameters and locals named like site builtins never reach the builtin
    "def m(x, entry, exit):\n    return entry * jnp.exp(-exit * x)\n",
    "def m(x, a):\n    help = 2.0\n    return a * x + help\n",
    "def m(x, quit, license):\n    copyright = quit * x\n    return copyright + license\n",
    # closure over an enclosing function's local
    "def outer(exit):\n    def m(x):\n        return exit * x\n    return m\n",
    # nonlocal continues to the enclosing function
    "def outer():\n    exit = 1.0\n    def m(x):\n        nonlocal exit\n"
    "        exit = 2.0\n        return exit * x\n    return m\n",
    # free variable of a comprehension resolves to the function local
    "def m(x, credits):\n    return sum([credits * v for v in x])\n",
    # method parameter inside a class
    "class C:\n    def m(self, x, exit):\n        return exit * x\n",
]

REJECTED = [
    # calls and module-level uses reach the builtin
    "exit()\n",
    'help("os")\n',
    "license()\n",
    "h = help\n",
    "exit = exit\n",
    "def m(x):\n    return exit\n",
    "def m(x, exit):\n    return exit()\n",  # calls stay rejected
    # module-level binding does not shadow: order is not tracked there
    "help = 2.0\ndef m(x):\n    return help * x\n",
    "if 0:\n    help = 0\nh = help\n",
    # class scope runs top to bottom with builtins as fallback
    "class C:\n    exit = exit\n",
    "def f(exit):\n    class C:\n        y = exit\n    return C\n",
    # defaults and decorators evaluate in the enclosing scope
    "def m(x, exit=exit):\n    return exit\n",
    "@help\ndef m(x):\n    return x\n",
    # global skips the function binding
    "def m(x):\n    global exit\n    exit = 1\n    return exit\n",
    # a comprehension target is not a function local
    "def f(xs):\n    [0 for exit in xs]\n    return exit\n",
    "h = [help for help in [help]][0]\n",
    # a sibling function's parameter does not shadow another function's use
    "def f(exit):\n    return exit\ndef g(x):\n    return exit\n",
    # lambda parameters are not modelled: conservative
    "f = lambda exit: exit\n",
    "def m(x):\n    y = quit\n    return y\n",
]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("src", ACCEPTED)
def test_shadowing_local_accepted(tmp_path, src, strict):
    f = tmp_path / "m.py"
    f.write_text(IMPORT + src)
    result = validate_model(f, strict=strict)
    assert result.is_valid, result.violations


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("src", REJECTED)
def test_builtin_use_rejected(tmp_path, src, strict):
    f = tmp_path / "m.py"
    f.write_text(IMPORT + src)
    assert not validate_model(f, strict=strict).is_valid


def test_strict_alias_of_shadowing_local_not_tainted(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(IMPORT + "def m(x, exit):\n    y = exit\n    return y * x\n")
    result = validate_model(f, strict=True)
    assert result.is_valid, result.violations
