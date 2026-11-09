import jax.numpy as jnp
import pytest

from nlsq.global_optimization.method_selector import MethodSelector
from nlsq.precision.parameter_normalizer import ParameterNormalizer


def test_normalizer_accepts_scalar_p0_and_rejects_inverted_bounds():
    n = ParameterNormalizer(2.0, bounds=(jnp.array([0.0]), jnp.array([4.0])))
    assert n.n_params == 1
    with pytest.raises(ValueError, match="lower bounds"):
        ParameterNormalizer(
            [1.0], bounds=(jnp.array([5.0]), jnp.array([1.0])), strategy="bounds"
        )


def test_method_selector_rejects_typos():
    sel = MethodSelector()
    lb, ub = jnp.array([0.0]), jnp.array([1.0])
    with pytest.raises(ValueError, match="Unknown global"):
        sel.select("cmeas", lb, ub)
    assert sel.select("multi-start", lb, ub) == "multi-start"
