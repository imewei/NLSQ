"""Tests for the large-fit strategy dataclasses."""

import dataclasses

import pytest

from nlsq.streaming.specs import Auto, Chunked, Exact, Hybrid, InMemory, Multistart


def test_auto_resolves_below_threshold_to_in_memory():
    auto = Auto()
    assert auto.resolve(auto.size_threshold - 1) is auto.in_memory


def test_auto_resolves_at_threshold_to_chunked():
    auto = Auto()
    assert auto.resolve(auto.size_threshold) is auto.chunked


def test_auto_without_size_detection_is_always_chunked():
    auto = Auto(auto_size_detection=False)
    assert auto.resolve(10) is auto.chunked


def test_exact_has_no_loss_field():
    with pytest.raises(TypeError):
        Exact(loss="huber")  # type: ignore[call-arg]


def test_strategies_are_frozen():
    for obj in (Multistart(), InMemory(), Chunked(), Hybrid(), Exact(), Auto()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.foo = 1  # type: ignore[attr-defined]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Auto().size_threshold = 5  # type: ignore[misc]


def test_multistart_defaults():
    ms = Multistart()
    assert (ms.n_starts, ms.sampler, ms.center_on_p0, ms.scale_factor, ms.enabled) == (
        10,
        "lhs",
        True,
        1.0,
        True,
    )


def test_inmemory_and_chunked_default_multistart_disabled():
    assert InMemory().multistart.enabled is False
    assert Chunked().multistart.enabled is False


def test_chunked_has_no_stability_fields():
    with pytest.raises(TypeError):
        Chunked(stability="auto")  # type: ignore[call-arg]


def test_options_defaults_not_shared():
    assert InMemory().options == {}
    assert InMemory().options is not InMemory().options


@pytest.mark.parametrize(
    "name",
    ["Auto", "Chunked", "Exact", "Hybrid", "InMemory", "Multistart", "LargeStrategy"],
)
def test_streaming_exports(name):
    from nlsq import streaming
    from nlsq.streaming import specs

    assert getattr(streaming, name) is getattr(specs, name)
    assert name in streaming.__all__


@pytest.mark.parametrize("cls", [InMemory, Chunked])
@pytest.mark.parametrize(
    "key",
    ["multistart", "sigma", "method", "bounds", "n_starts", "p0", "global_search"],
)
def test_options_colliding_with_strategy_keys_raise(cls, key):
    with pytest.raises(TypeError, match=key):
        cls(options={key: 1})


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (InMemory, "stability"),
        (InMemory, "rescale_data"),
        (InMemory, "max_jacobian_elements_for_svd"),
        (Chunked, "chunk_size"),
        (Chunked, "show_progress"),
    ],
)
def test_options_colliding_with_class_specific_keys_raise(cls, key):
    with pytest.raises(TypeError, match=key):
        cls(options={key: 1})


def test_non_colliding_option_accepted_and_forwardable():
    strategy = InMemory(options={"ftol": 1e-9})
    assert dict(strategy.options) == {"ftol": 1e-9}
    assert {**strategy.options, "xtol": 1e-6} == {"ftol": 1e-9, "xtol": 1e-6}
    assert dataclasses.replace(strategy, method="trf").options == {"ftol": 1e-9}
    assert Chunked(options={"ftol": 1e-9}).options["ftol"] == 1e-9


@pytest.mark.parametrize("cls", [InMemory, Chunked])
def test_options_are_private_copy(cls):
    original = {"ftol": 1e-9}
    strategy = cls(options=original)
    original["ftol"] = 1.0
    original["xtol"] = 1.0
    assert dict(strategy.options) == {"ftol": 1e-9}


def test_legacy_auto_has_independent_options():
    from nlsq import _legacy_to_strategy

    auto = _legacy_to_strategy(ftol=1e-9)
    assert dict(auto.in_memory.options) == {"ftol": 1e-9}
    assert dict(auto.chunked.options) == {"ftol": 1e-9}
    assert auto.in_memory.options is not auto.chunked.options


@pytest.mark.parametrize("cls", [InMemory, Chunked])
def test_options_copied_and_spec_copyable(cls):
    import copy
    import dataclasses
    import pickle

    src = {"loss": "soft_l1"}
    spec = cls(options=src)
    src["loss"] = "linear"
    assert spec.options["loss"] == "soft_l1"
    assert copy.deepcopy(spec) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert dataclasses.replace(spec).options == spec.options
