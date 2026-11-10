"""Run a large-dataset fit for an explicit strategy.

``fit_with_strategy`` holds the path bodies that ``curve_fit_large`` used to
contain: source validation, ``Auto`` resolution, and the in-memory, chunked,
hybrid and exact paths. This module imports the engines inside
the functions, not at its own top level. Importing it still runs
``nlsq/streaming/__init__.py``, which loads them; ``import nlsq`` alone does not.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from nlsq.core._option_checks import coerce_absolute_sigma
from nlsq.streaming.specs import (
    Auto,
    Chunked,
    Exact,
    Hybrid,
    InMemory,
    LargeStrategy,
    Multistart,
    reject_owned_kwargs,
)
from nlsq.utils.logging import get_logger

__all__ = ["fit_with_strategy"]

_logger = get_logger(__name__)

# Chunked options the exact solver honours; any other key keeps the chunked engine
_EXACT_OPTION_KEYS = frozenset({"ftol", "xtol", "gtol", "max_nfev"})
# Probe length of the pointwise-model check on the default route
_PROBE_POINTS = 4096
_CHUNKED_HINT = (
    "Chunked GLS is an approximation, not full-batch equivalent for nonlinear "
    "models; pass strategy=Chunked() to accept it."
)


def _is_default(value: Any, default: Any) -> bool:
    """``value`` is the field default, also by type: ``np.float32(1e-8) == 1e-8``."""
    return value == default and type(value) is type(default)


def _multistart_kwargs(ms: Multistart) -> dict[str, Any]:
    """The five multi-start keywords every in-memory and chunked fit receives."""
    return {
        "multistart": ms.enabled,
        "n_starts": ms.n_starts,
        "sampler": ms.sampler,
        "center_on_p0": ms.center_on_p0,
        "scale_factor": ms.scale_factor,
    }


def _is_unbounded(lb: Any, ub: Any) -> bool:
    return bool(np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))


def fit_with_strategy(
    strategy: LargeStrategy,
    f: Any,
    xdata: Any,
    ydata: Any,
    p0: Any,
    *,
    sigma: Any,
    absolute_sigma: bool,
    bounds: Any,
    check_finite: bool,
    memory_limit_gb: float | None,
    show_progress: bool,
    compute_covariance: bool = True,
    **kwargs: Any,
) -> Any:
    """Fit ``f`` to the data with ``strategy``.

    Parameters
    ----------
    strategy : Auto, InMemory, Chunked, Hybrid or Exact
        The path to run. ``Auto`` is resolved from the validated point count.
    f, xdata, ydata, p0, sigma, absolute_sigma, bounds, check_finite
        As in ``curve_fit_large``.
    memory_limit_gb, show_progress
        Used by the chunked and exact paths; ignored by the others.
    compute_covariance
        False returns ``pcov=None``. Chunked: only the final ``pcov`` is dropped
        (the GLS combiner still needs the chunk covariances). Hybrid: ``ValueError``.
    **kwargs
        Extra keyword arguments for the engine. In-memory and chunked: merged
        over the strategy's ``options``; a key the strategy owns (see
        ``InMemory.options``, ``Chunked.options``) raises ``TypeError``. Hybrid:
        none accepted (``TypeError``); ``verbose`` and the config live on
        ``Hybrid``. Exact: merged over the strategy's solver settings, except
        that its own fields (``chunk_size``, ``ftol``, ``xtol``, ``gtol``,
        ``max_nfev``, ``n_points``) raise ``TypeError``.

    Returns
    -------
    CurveFitResult or tuple
        In-memory: ``curve_fit``'s native return, unchanged. Otherwise
        ``(popt, pcov)``.
    """
    common: dict[str, Any] = {
        "sigma": sigma,
        "absolute_sigma": coerce_absolute_sigma(absolute_sigma),
        "bounds": bounds,
        "check_finite": check_finite,
    }
    if isinstance(strategy, Exact):
        # Lazy sources must not be read whole: no np.asarray on this path
        return _fit_exact(
            strategy,
            f,
            xdata,
            ydata,
            p0,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress,
            compute_covariance=compute_covariance,
            kwargs=kwargs,
            **common,
        )

    from nlsq.streaming.exact_lm import is_chunk_factory

    if is_chunk_factory(xdata) or ydata is None:
        raise ValueError(
            "a chunk factory (callable xdata, ydata=None) requires "
            "method='exact_streaming'; pass xdata and ydata arrays otherwise"
        )

    # Input validation
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check for edge cases (scipy convention: 2-D xdata is (k, M), points on last axis)
    n_x = xdata.shape[-1] if xdata.ndim > 1 else len(xdata)
    if n_x == 0:
        raise ValueError("`xdata` cannot be empty.")
    if len(ydata) == 0:
        raise ValueError("`ydata` cannot be empty.")
    if n_x != len(ydata):
        raise ValueError(
            f"`xdata` and `ydata` must have the same length: {n_x} vs {len(ydata)}.",
        )
    if n_x < 2:
        raise ValueError(f"Need at least 2 data points for fitting, got {n_x}.")

    n_points = n_x

    if isinstance(strategy, Hybrid):
        if not compute_covariance:
            raise ValueError(
                "compute_covariance=False is not supported by hybrid_streaming"
            )
        return _fit_hybrid(strategy, f, xdata, ydata, p0, kwargs=kwargs, **common)
    if isinstance(strategy, Auto):
        auto = strategy
        strategy = auto.resolve(n_points)
        if isinstance(strategy, Exact):
            # an owned key raises TypeError here, as on the chunked path, not after a fallback
            reject_owned_kwargs(auto.chunked, kwargs)
            opts = {**auto.chunked.options, **kwargs}  # what chunked would have used
            hit = _exact_ineligible(
                auto, opts, f, xdata, n_points, sigma, p0, bounds, check_finite
            )
            if hit is None:
                return _fit_exact(
                    _effective_exact(strategy, auto, opts),
                    f,
                    xdata,
                    ydata,
                    p0,
                    memory_limit_gb=memory_limit_gb,
                    show_progress=show_progress,
                    compute_covariance=compute_covariance,
                    kwargs={},
                    route="default",
                    **common,
                )
            kind, reason = hit
            if kind == "data":
                raise ValueError(
                    f"curve_fit_large: exact streaming cannot fit this input "
                    f"({reason}). {_CHUNKED_HINT}"
                )
            msg = (
                f"curve_fit_large: exact streaming not used ({reason}); using chunked "
                "GLS, which is not full-batch equivalent for nonlinear models. Pass "
                "strategy=Chunked() to choose it explicitly."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            _logger.info(msg)
            strategy = auto.chunked
    if isinstance(strategy, InMemory):
        return _fit_in_memory(
            strategy,
            f,
            xdata,
            ydata,
            p0,
            compute_covariance=compute_covariance,
            kwargs=kwargs,
            **common,
        )
    if isinstance(strategy, Chunked):
        return _fit_chunked(
            strategy,
            f,
            xdata,
            ydata,
            p0,
            n_points=n_points,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress,
            compute_covariance=compute_covariance,
            kwargs=kwargs,
            **common,
        )
    raise TypeError(f"unknown large-fit strategy {strategy!r}")


def _fit_exact(
    strategy: Exact,
    f: Any,
    xdata: Any,
    ydata: Any,
    p0: Any,
    *,
    sigma: Any,
    absolute_sigma: bool,
    bounds: Any,
    check_finite: bool,
    memory_limit_gb: float | None,
    show_progress: bool,
    compute_covariance: bool,
    kwargs: dict[str, Any],
    route: Literal["explicit", "default"] = "explicit",
) -> Any:
    """Exact streaming fit. ``route="default"`` (reached through ``Auto``) appends the
    ``strategy=Chunked()`` hint to a pointwise-contract ``ValueError`` and to
    ``ExactSolverError``; the explicit route raises them unchanged."""
    from nlsq.streaming import exact_lm
    from nlsq.streaming.strategy import (
        build_large_strategy,
        check_exact_sources,
        normalize_p0_bounds,
    )

    reject_owned_kwargs(strategy, kwargs)
    try:
        check_exact_sources(xdata, ydata, sigma, strategy.n_points)
    except ValueError as err:
        shape_msg = " must be 1-D for method='exact_streaming'"
        if route == "default" and shape_msg in str(err):
            # the user never chose method='exact_streaming'
            reason = str(err).replace(shape_msg, " must be 1-D")
            raise ValueError(
                f"curve_fit_large: exact streaming cannot fit this input ({reason}). "
                f"{_CHUNKED_HINT}"
            ) from err
        raise
    if p0 is not None:
        p0 = np.asarray(p0, dtype=float)  # a complex list p0 must raise TypeError
    p0_arr, exact_bounds = normalize_p0_bounds(f, p0, bounds)
    # (p,) shapes and lb < ub elementwise; prepare_bounds only broadcasts scalars
    lb, ub = exact_lm.check_bounds(exact_bounds, len(p0_arr))
    # Only settings that differ from the defaults (which equal exact_lm_fit's)
    # reach the engine, as with the legacy keywords; check_exact_sources
    # guarantees n_points is None for a non-factory source. A value of another
    # type is forwarded verbatim: np.float32(1e-8) == 1e-8 although they differ.
    solver_kwargs = {
        fld.name: getattr(strategy, fld.name)
        for fld in dataclasses.fields(strategy)
        if fld.name != "chunk_size"
        and not _is_default(getattr(strategy, fld.name), fld.default)
    }
    solver_kwargs.update(kwargs)
    large_strategy = build_large_strategy("exact", None)
    try:
        native = large_strategy.fit(
            f,
            xdata,
            ydata,
            p0_arr,
            bounds=(lb, ub),
            chunk_size=strategy.chunk_size,
            check_finite=check_finite,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            memory_limit_gb=memory_limit_gb,
            show_progress=show_progress,
            compute_covariance=compute_covariance,
            **solver_kwargs,
        )
        return large_strategy.unpack(native)
    except ValueError as err:
        if route == "default" and str(err).startswith("model is not pointwise"):
            raise ValueError(f"{err}. {_CHUNKED_HINT}") from err
        raise
    except exact_lm.ExactSolverError as err:
        if route == "default":
            raise exact_lm.ExactSolverError(
                f"{err} (default exact path; strategy=Chunked() restores the chunked "
                "GLS fit)",
                err.result,
            ) from err
        raise


def _effective_exact(exact: Exact, auto: Auto, opts: Mapping[str, Any]) -> Exact:
    """``exact`` (``auto.exact``) with the large-path settings the caller gave elsewhere.

    Per field, a non-default value on ``exact`` wins; otherwise ``chunk_size``
    comes from ``auto.chunked`` and ``ftol``, ``xtol``, ``gtol``, ``max_nfev`` from
    ``opts`` (``auto.chunked.options`` merged with the call kwargs).
    """
    inherited = {"chunk_size": auto.chunked.chunk_size} | {
        k: opts[k] for k in _EXACT_OPTION_KEYS if k in opts
    }
    defaults = {fld.name: fld.default for fld in dataclasses.fields(Exact)}
    changes = {
        name: value
        for name, value in inherited.items()
        if _is_default(getattr(exact, name), defaults[name])
    }
    return dataclasses.replace(exact, **changes)


def _exact_ineligible(
    auto: Auto,
    opts: Mapping[str, Any],
    f: Any,
    xdata: np.ndarray,
    n_points: int,
    sigma: Any,
    p0: Any,
    bounds: Any,
    check_finite: bool,
) -> tuple[Literal["engine", "data"], str] | None:
    """Why the ``Auto`` default cannot use exact, or None.

    ``"engine"``: ``auto.chunked`` asks for a chunked-engine feature (fall back with a
    warning). ``"data"``: exact cannot fit this input (raise). Scalar and signed
    ``sigma`` are not reasons; a 2-D ``sigma`` raises in the exact path itself.
    """
    from nlsq.streaming.exact_lm import check_model_contract
    from nlsq.streaming.strategy import normalize_p0_bounds

    chunked = auto.chunked
    if chunked.method is not None:
        return "engine", "Auto.chunked sets method"
    if chunked.multistart.enabled:
        return "engine", "Auto.chunked enables multistart"
    extra = sorted(set(opts) - _EXACT_OPTION_KEYS)
    if extra:
        return (
            "engine",
            f"Auto.chunked option(s) {extra} are not supported by exact",
        )
    if xdata.ndim != 1:
        return "data", "multi-feature xdata"
    if (
        sigma is not None
        and not check_finite
        and not np.isfinite(np.asarray(sigma)).all()
    ):
        return "data", "sigma has non-finite values (check_finite=False)"
    if p0 is not None:
        p0 = np.asarray(p0, dtype=float)
    p0_arr, _ = normalize_p0_bounds(f, p0, bounds)
    try:
        check_model_contract(f, p0_arr, xdata[: min(n_points, _PROBE_POINTS)])
    except ValueError as err:
        if not str(err).startswith("model is not pointwise"):
            raise
        return "data", str(err)
    return None


def _fit_hybrid(
    strategy: Hybrid,
    f: Any,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Any,
    *,
    sigma: Any,
    absolute_sigma: bool,
    bounds: Any,
    check_finite: bool,
    kwargs: dict[str, Any],
) -> Any:
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.strategy import (
        build_large_strategy,
        normalize_p0_bounds,
        scipy_layout_to_points_first,
    )

    if kwargs:
        # curve_fit_large's shim builds the config; Hybrid takes no engine kwargs
        raise TypeError(
            f"Unrecognized keyword argument(s) for a Hybrid strategy: {sorted(kwargs)}. "
            "Set verbose and HybridStreamingConfig fields on Hybrid(config, verbose).",
        )
    config = strategy.config if strategy.config is not None else HybridStreamingConfig()

    if check_finite and not (
        np.isfinite(np.asarray(xdata)).all() and np.isfinite(np.asarray(ydata)).all()
    ):
        raise ValueError("x and y must be finite (found NaN or inf)")

    # Prepare p0 and bounds
    p0, (lb, ub) = normalize_p0_bounds(f, p0, bounds)
    bounds_tuple = None if _is_unbounded(lb, ub) else (lb, ub)

    # Build the hybrid engine (AdaptiveHybridStreamingOptimizer)
    large_strategy = build_large_strategy("hybrid", config)

    # curve_fit_large's contract is scipy's (k, M); the engine is points-first.
    f_engine, x_engine = scipy_layout_to_points_first(f, xdata)

    # Run optimization
    native = large_strategy.fit(
        f_engine,
        x_engine,
        ydata,
        p0,
        bounds=bounds_tuple,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        callback=None,  # Hybrid has no callback; an unknown keyword raised above
        verbose=strategy.verbose,
    )

    return large_strategy.unpack(native)


def _fit_in_memory(
    strategy: InMemory,
    f: Any,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Any,
    *,
    sigma: Any,
    absolute_sigma: bool,
    bounds: Any,
    check_finite: bool,
    compute_covariance: bool,
    kwargs: dict[str, Any],
) -> Any:
    from nlsq.streaming.strategy import build_large_strategy

    reject_owned_kwargs(strategy, kwargs)
    fit_kwargs = {**strategy.options, **kwargs}
    if not compute_covariance:
        fit_kwargs["compute_covariance"] = False
    if sigma is not None:
        fit_kwargs["sigma"] = sigma
    _lb, _ub = bounds
    _lb = np.atleast_1d(np.asarray(_lb, dtype=float))
    _ub = np.atleast_1d(np.asarray(_ub, dtype=float))
    if not _is_unbounded(_lb, _ub):
        fit_kwargs["bounds"] = bounds
    if strategy.method is not None:
        fit_kwargs["method"] = strategy.method
    fit_kwargs["absolute_sigma"] = absolute_sigma
    fit_kwargs["check_finite"] = check_finite
    fit_kwargs["stability"] = strategy.stability
    fit_kwargs["rescale_data"] = strategy.rescale_data
    fit_kwargs["max_jacobian_elements_for_svd"] = strategy.max_jacobian_elements_for_svd
    # Always forwarded: method="multi-start" uses these even when enabled is False
    fit_kwargs.update(_multistart_kwargs(strategy.multistart))

    large_strategy = build_large_strategy("in_memory", None)
    return large_strategy.unpack(large_strategy.fit(f, xdata, ydata, p0, **fit_kwargs))


def _fit_chunked(
    strategy: Chunked,
    f: Any,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Any,
    *,
    sigma: Any,
    absolute_sigma: bool,
    bounds: Any,
    check_finite: bool,
    n_points: int,
    memory_limit_gb: float | None,
    show_progress: bool,
    compute_covariance: bool,
    kwargs: dict[str, Any],
) -> Any:
    from nlsq.config import (
        LargeDatasetConfig,
        MemoryConfig,
        get_memory_config,
        large_dataset_context,
        memory_context,
    )
    from nlsq.streaming.large_dataset import LDMemoryConfig
    from nlsq.streaming.strategy import build_large_strategy

    reject_owned_kwargs(strategy, kwargs)
    chunk_size = strategy.chunk_size
    fit_kwargs = {**strategy.options, **kwargs}

    # Configure memory settings if provided
    if memory_limit_gb is None:
        # Auto-detect available memory
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = min(8.0, available_gb * 0.7)  # Use 70% of available
        except ImportError:
            memory_limit_gb = 8.0  # Conservative default

    # Create memory configuration
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # Default ceiling: a tenth of the data, but never below the 1000-point floor
    # (or the whole input if smaller), so tiny inputs fit as one chunk, not many.
    max_chunk_size = (
        min(1_000_000, max(n_points // 10, min(1000, n_points)))
        if chunk_size is None
        else chunk_size
    )
    memory_config = MemoryConfig(
        progress_reporting=show_progress,
        # The floor must not exceed the ceiling (small n_points or chunk_size)
        min_chunk_size=min(max(1000, n_points // 10000), max_chunk_size),
        max_chunk_size=max_chunk_size,
    )

    # Create large dataset configuration (v0.2.0: no more sampling params)
    large_dataset_config = LargeDatasetConfig(
        enable_automatic_solver_selection=True,
    )

    # Use context managers to temporarily set configuration
    # Layer only the chunk bounds and progress flag over the user's global
    # settings; the fitter below gets its limit explicitly.
    context_config = dataclasses.replace(
        get_memory_config(),
        progress_reporting=memory_config.progress_reporting,
        min_chunk_size=memory_config.min_chunk_size,
        max_chunk_size=memory_config.max_chunk_size,
    )
    with memory_context(context_config), large_dataset_context(large_dataset_config):
        # Build the chunked engine (LargeDatasetFitter) inside the contexts
        large_strategy = build_large_strategy(
            "chunked",
            LDMemoryConfig(
                memory_limit_gb=memory_limit_gb,
                min_chunk_size=memory_config.min_chunk_size,
                max_chunk_size=memory_config.max_chunk_size,
            ),
            show_progress=show_progress,
            memory_limit_gb=memory_limit_gb,
        )

        # Handle sigma parameter by including it in kwargs if provided
        if sigma is not None:
            fit_kwargs["sigma"] = sigma
        fit_kwargs["absolute_sigma"] = absolute_sigma
        fit_kwargs["check_finite"] = check_finite

        # Add multi-start parameters to kwargs (inert in the chunk fits; pinned)
        fit_kwargs.update(_multistart_kwargs(strategy.multistart))

        # Convert p0 to appropriate type for LargeDatasetFitter
        # LargeDatasetFitter expects np.ndarray | list | None (no tuple or jnp.ndarray)
        converted_p0: np.ndarray | list | None
        if p0 is None:
            converted_p0 = None
        elif isinstance(p0, list):
            converted_p0 = p0
        else:
            # Convert tuple, jnp.ndarray, or np.ndarray to np.ndarray
            converted_p0 = np.asarray(p0)

        # Provide default method if None
        final_method = strategy.method if strategy.method is not None else "trf"

        # Perform the fit (fit_with_progress when show_progress)
        native = large_strategy.fit(
            f,
            xdata,
            ydata,
            converted_p0,
            bounds=bounds,
            method=final_method,
            **fit_kwargs,
        )

        popt, pcov = large_strategy.unpack(native)
        if not compute_covariance:
            # The GLS combiner needed the chunk covariances; only the result drops
            _logger.debug("covariance dropped (compute_covariance=False)")
            return popt, None
        return popt, pcov
