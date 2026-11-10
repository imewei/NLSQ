"""curve_fit_large legacy keywords -> strategy (_legacy_to_strategy) and strategy=."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import _legacy_to_strategy, curve_fit_large
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.specs import (
    Auto,
    Chunked,
    Exact,
    Hybrid,
    InMemory,
    Multistart,
)

TRUE = np.array([2.0, 0.5])
N = 2000


def model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    x = np.linspace(0.0, 5.0, N)
    y = TRUE[0] * np.exp(-TRUE[1] * x) + 0.001 * np.random.default_rng(0).normal(size=N)
    return x, y


def _auto(ms=None, **chunked):
    ms = ms or Multistart(enabled=False)
    return Auto(
        in_memory=InMemory(multistart=ms),
        chunked=Chunked(multistart=ms, **chunked),
    )


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({}, Auto()),
        ({"chunk_size": 500, "multistart": True}, _auto(Multistart(), chunk_size=500)),
        ({"global_search": True}, _auto(Multistart(n_starts=20))),
        (
            {"method": "hybrid_streaming", "gauss_newton_tol": 1e-6},
            Hybrid(config=HybridStreamingConfig(gauss_newton_tol=1e-6), verbose=1),
        ),
        ({"method": "exact_streaming", "gtol": 1e-10}, Exact(gtol=1e-10)),
    ],
    ids=["plain", "chunk_size_multistart", "global_search", "hybrid", "exact"],
)
def test_legacy_mapping(legacy, expected):
    assert _legacy_to_strategy(**legacy) == expected


def test_legacy_mapping_carries_stability_on_in_memory_only():
    s = _legacy_to_strategy(
        stability="check",
        rescale_data=False,
        max_jacobian_elements_for_svd=5,
        size_threshold=7,
        auto_size_detection=False,
        loss="huber",
    )
    assert s == Auto(
        size_threshold=7,
        auto_size_detection=False,
        in_memory=InMemory(
            stability="check",
            rescale_data=False,
            max_jacobian_elements_for_svd=5,
            options={"loss": "huber"},
        ),
        chunked=Chunked(options={"loss": "huber"}),
    )


@pytest.mark.parametrize(
    ("legacy", "exc", "match"),
    [
        (
            {"method": "exact_streaming", "stability": "auto"},
            ValueError,
            r"^stability='auto' is not supported by method='exact_streaming'; "
            r"leave it at its default$",
        ),
        (
            {"method": "exact_streaming", "loss": "huber"},
            TypeError,
            r"^Unrecognized keyword argument\(s\) for method='exact_streaming': "
            r"\['loss'\]\. Accepted: \['ftol', 'gtol', 'max_nfev', 'n_points', "
            r"'xtol'\]\.$",
        ),
        (
            {"method": "hybrid_streaming", "callback": print},
            TypeError,
            r"^Unrecognized keyword argument\(s\) for method='hybrid_streaming': "
            r"\['callback'\]\. These are not HybridStreamingConfig fields\.$",
        ),
    ],
    ids=["exact_stability", "exact_loss", "hybrid_callback"],
)
def test_legacy_errors_keep_messages(legacy, exc, match):
    with pytest.raises(exc, match=match):
        _legacy_to_strategy(**legacy)


def test_removed_sampling_parameter_still_type_error():
    x, y = _data()
    with pytest.raises(TypeError, match=r"'enable_sampling'.*removed in v0\.2\.0"):
        curve_fit_large(model, x, y, enable_sampling=True)


@pytest.mark.parametrize(
    ("strategy", "legacy", "name"),
    [
        (Chunked(), {"size_threshold": 5}, "size_threshold"),
        (InMemory(), {"method": "trf"}, "method"),
        (Exact(), {"ftol": 1e-10}, "ftol"),
        (Chunked(), {"global_search": True}, "global_search"),
    ],
)
def test_explicit_strategy_rejects_non_default_legacy_option(strategy, legacy, name):
    x, y = _data()
    with pytest.raises(TypeError, match=name):
        curve_fit_large(model, x, y, p0=[1.5, 0.4], strategy=strategy, **legacy)


def test_explicit_strategy_accepts_explicit_defaults():
    x, y = _data()
    popt, pcov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.5, 0.4],
        strategy=Chunked(chunk_size=500),
        method=None,
        stability=False,
        size_threshold=np.int64(1_000_000),
        n_starts=10,
        sampler="lhs",
        scale_factor=1.0,
        memory_limit_gb=4.0,
        show_progress=False,
    )
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)
    assert np.shape(pcov) == (2, 2)


def test_explicit_strategy_array_valued_legacy_option_is_type_error():
    x, y = _data()
    with pytest.raises(TypeError, match="scale_factor"):
        curve_fit_large(
            model, x, y, strategy=InMemory(), scale_factor=np.array([1.0, 2.0])
        )


def test_explicit_strategy_rejects_engine_keyword():
    x, y = _data()
    with pytest.raises(TypeError, match="together with strategy="):
        curve_fit_large(model, x, y, p0=[1.5, 0.4], strategy=InMemory(), ftol=1e-10)


@pytest.mark.parametrize(
    "value",
    [np.array(1.0), np.array([1.0]), np.array([[1.0]])],
    ids=["0d", "size1_1d", "size1_2d"],
)
def test_explicit_strategy_rejects_size_one_array_legacy_option(value):
    x, y = _data()
    with pytest.raises(TypeError, match="scale_factor"):
        curve_fit_large(model, x, y, strategy=InMemory(), scale_factor=value)


def test_explicit_strategy_accepts_numpy_scalar_equal_to_default():
    x, y = _data()
    popt, _ = curve_fit_large(
        model, x, y, p0=[1.5, 0.4], strategy=InMemory(), scale_factor=np.float64(1.0)
    )
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)


def test_method_multi_start_keeps_sampling_values_when_flag_false(monkeypatch):
    seen = {}

    def spy(f, x, y, **kwargs):
        seen.update(kwargs)
        return np.ones(2), np.eye(2)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    x, y = _data()
    curve_fit_large(
        model,
        x,
        y,
        p0=[1.5, 0.4],
        method="multi-start",
        multistart=False,
        n_starts=3,
        sampler="sobol",
    )
    assert seen["method"] == "multi-start"
    assert seen["multistart"] is False
    assert seen["n_starts"] == 3
    assert seen["sampler"] == "sobol"


@pytest.mark.parametrize("method", ["trf", "lm", "auto", "cmaes", "multi-start"])
def test_every_curve_fit_method_translates(method):
    s = _legacy_to_strategy(method=method)
    assert isinstance(s, Auto)
    assert s.in_memory.method == method
    assert s.chunked.method == method


def test_exact_accepts_multistart_suboptions_when_flag_false():
    s = _legacy_to_strategy(
        method="exact_streaming",
        n_starts=3,
        sampler="sobol",
        center_on_p0=False,
        scale_factor=2.0,
    )
    assert s == Exact()


def test_hybrid_double_invalid_reports_keyword_type_error():
    # Spec behavior change 2: the keyword check now runs before source validation.
    with pytest.raises(TypeError, match="hybrid_streaming"):
        curve_fit_large(model, [], [], method="hybrid_streaming", bogus=1)


def test_strategy_call_matches_legacy_call():
    x, y = _data()
    p0 = [1.5, 0.4]
    popt_s, pcov_s = curve_fit_large(
        model,
        x,
        y,
        p0=p0,
        strategy=Auto(auto_size_detection=False, chunked=Chunked(chunk_size=500)),
    )
    popt_l, pcov_l = curve_fit_large(
        model, x, y, p0=p0, chunk_size=500, auto_size_detection=False
    )
    np.testing.assert_allclose(popt_s, popt_l, rtol=1e-12)
    np.testing.assert_allclose(pcov_s, pcov_l, rtol=1e-12)


# One distinct non-default value per legacy path option
_NON_DEFAULT_LEGACY = {
    "method": "trf",
    "stability": "check",
    "rescale_data": False,
    "max_jacobian_elements_for_svd": 5,
    "auto_size_detection": False,
    "size_threshold": 5,
    "chunk_size": 100,
    "multistart": True,
    "n_starts": 3,
    "global_search": True,
    "sampler": "sobol",
    "center_on_p0": False,
    "scale_factor": 2.0,
}


def test_non_default_table_covers_every_legacy_option():
    assert set(_NON_DEFAULT_LEGACY) == set(nlsq._LEGACY_PATH_OPTIONS)
    for name, value in _NON_DEFAULT_LEGACY.items():
        assert value != nlsq._LEGACY_DEFAULTS[name]


@pytest.mark.parametrize("name", sorted(_NON_DEFAULT_LEGACY))
def test_every_legacy_option_is_rejected_with_explicit_strategy(name):
    x, y = _data()
    with pytest.raises(TypeError, match="together with strategy="):
        curve_fit_large(
            model, x, y, strategy=Chunked(), **{name: _NON_DEFAULT_LEGACY[name]}
        )


def test_explicit_strategy_accepts_signature_default_chunk_size():
    x, y = _data()
    popt, _ = curve_fit_large(
        model,
        x,
        y,
        p0=[1.5, 0.4],
        strategy=Chunked(chunk_size=200),
        chunk_size=nlsq._LEGACY_DEFAULTS["chunk_size"],
    )
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)


@pytest.mark.parametrize(
    "strategy", [Hybrid(verbose=0), Exact()], ids=["hybrid", "exact"]
)
def test_hybrid_and_exact_strategies_accept_default_equal_legacy_options(strategy):
    x, y = _data()
    popt, _ = curve_fit_large(
        model,
        x,
        y,
        p0=[1.5, 0.4],
        strategy=strategy,
        method=None,
        stability=False,
        chunk_size=None,
        n_starts=10,
    )
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)


def test_legacy_mapping_places_every_option_in_its_spec_field():
    ms = Multistart(n_starts=3, sampler="sobol", center_on_p0=False, scale_factor=2.0)
    got = _legacy_to_strategy(
        method="trf",
        stability="check",
        rescale_data=False,
        max_jacobian_elements_for_svd=5,
        auto_size_detection=False,
        size_threshold=7,
        chunk_size=100,
        multistart=True,
        n_starts=3,
        sampler="sobol",
        center_on_p0=False,
        scale_factor=2.0,
        ftol=1e-9,
    )
    assert got.size_threshold == 7
    assert got.auto_size_detection is False
    assert got.in_memory.multistart == ms
    assert got.chunked.multistart == ms
    assert got == Auto(
        size_threshold=7,
        auto_size_detection=False,
        in_memory=InMemory(
            method="trf",
            stability="check",
            rescale_data=False,
            max_jacobian_elements_for_svd=5,
            multistart=ms,
            options={"ftol": 1e-9},
        ),
        chunked=Chunked(
            chunk_size=100, method="trf", multistart=ms, options={"ftol": 1e-9}
        ),
    )


def test_global_search_forces_twenty_starts_over_user_n_starts():
    got = _legacy_to_strategy(global_search=True, n_starts=3)
    for spec in (got.in_memory, got.chunked):
        assert spec.multistart.enabled is True
        assert spec.multistart.n_starts == 20


def test_global_search_with_exact_streaming_is_rejected():
    with pytest.raises(ValueError, match=r"multistart=True .*exact_streaming"):
        _legacy_to_strategy(method="exact_streaming", global_search=True)


def test_hybrid_mapping_carries_verbose():
    got = _legacy_to_strategy(method="hybrid_streaming", verbose=0)
    assert got == Hybrid(config=HybridStreamingConfig(), verbose=0)


def test_exact_mapping_carries_chunk_size_and_solver_settings():
    got = _legacy_to_strategy(
        method="exact_streaming",
        chunk_size=100,
        ftol=1e-9,
        xtol=1e-10,
        gtol=1e-11,
        max_nfev=5,
        n_points=10,
    )
    assert got == Exact(
        chunk_size=100, ftol=1e-9, xtol=1e-10, gtol=1e-11, max_nfev=5, n_points=10
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("stability", "auto"),
        ("rescale_data", False),
        ("max_jacobian_elements_for_svd", 5),
        ("auto_size_detection", False),
        ("size_threshold", 7),
    ],
)
def test_exact_rejects_each_unsupported_option(name, value):
    with pytest.raises(ValueError, match=rf"{name}={value!r} is not supported"):
        _legacy_to_strategy(method="exact_streaming", **{name: value})


def test_hybrid_rejects_gtol_as_on_main():
    # d2de6bc8 raised the same TypeError: gtol is not a HybridStreamingConfig field
    with pytest.raises(TypeError, match=r"'hybrid_streaming': \['gtol'\]"):
        _legacy_to_strategy(method="hybrid_streaming", gtol=1e-6)


def test_shim_and_curve_fit_large_emit_no_deprecation_warning():
    # pyproject filterwarnings hide DeprecationWarning, so record with "always"
    import warnings

    x = np.linspace(0.0, 5.0, 200)
    y = 2.0 * np.exp(-0.5 * x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _legacy_to_strategy(method="trf", stability="check", loss="huber")
        _legacy_to_strategy(method="exact_streaming", ftol=1e-9)
        curve_fit_large(model, x, y, p0=[1.5, 0.4])
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []
