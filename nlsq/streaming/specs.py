"""Strategy specifications for large-dataset fitting.

Frozen dataclasses describing how a large fit is run. This module itself imports
no engines; it only declares configuration. (Importing it still runs
``nlsq/streaming/__init__.py``, which loads the engines; ``import nlsq`` alone
loads none of them.)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

# nlsq.types.MethodLiteral omits "lm"; "hybrid_streaming" and "exact_streaming"
# are selected by strategy type and never reach these specs.
FitMethod = Literal["trf", "lm", "auto", "cmaes", "multi-start"]

# Keys that nlsq.streaming.large_fit assigns after merging ``options`` with the call
# kwargs (and the multistart/method/bounds keywords it passes explicitly); a colliding
# key in ``options`` would be silently overwritten, so it is rejected up front.
_COMMON_RESERVED = frozenset(
    {
        "sigma",
        "bounds",
        "absolute_sigma",
        "check_finite",
        "multistart",
        "n_starts",
        "sampler",
        "center_on_p0",
        "scale_factor",
        "method",
    }
)
_IN_MEMORY_RESERVED = _COMMON_RESERVED | {
    "stability",
    "rescale_data",
    "max_jacobian_elements_for_svd",
}
_CHUNKED_RESERVED = _COMMON_RESERVED | {"chunk_size"}


def _freeze_options(obj: Any, reserved: frozenset[str]) -> None:
    """Reject keys of ``obj.options`` that the strategy owns; store a private copy."""
    clash = sorted(set(obj.options) & reserved)
    if clash:
        raise TypeError(
            f"{type(obj).__name__}.options must not contain {clash}: set them on "
            f"{type(obj).__name__} or pass them to curve_fit_large directly."
        )
    object.__setattr__(obj, "options", dict(obj.options))


@dataclass(frozen=True)
class Multistart:
    """Multi-start settings.

    Parameters
    ----------
    n_starts : int
        Number of starting points.
    sampler : {"lhs", "sobol", "halton"}
        Sampling scheme.
    center_on_p0 : bool
        Center the samples on the initial guess.
    scale_factor : float
        Scale of the sampling region.
    enabled : bool
        Whether multi-start is active.
    """

    n_starts: int = 10
    sampler: Literal["lhs", "sobol", "halton"] = "lhs"
    center_on_p0: bool = True
    scale_factor: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class InMemory:
    """Fit with the whole dataset resident in memory.

    Parameters
    ----------
    method : FitMethod or None
        Optimizer method; ``None`` uses the default.
    stability : {"auto", "check", False}
        Numerical stability mode.
    rescale_data : bool
        Reserved: data rescaling is currently disabled in ``curve_fit``; with
        ``stability="auto"`` it only warns when rescaling would have applied.
    max_jacobian_elements_for_svd : int
        Largest Jacobian size for which SVD checks are run.
    multistart : Multistart
        Multi-start settings (disabled by default).
    options : Mapping
        Extra keyword options forwarded to the fit. Keys that this class or
        the fit sets itself (its fields, ``sigma``, ``bounds``,
        ``absolute_sigma``, ``check_finite`` and the multi-start keywords)
        raise ``TypeError``. Stored as a private copy (so the spec stays picklable and
        deep-copyable).

    Notes
    -----
    Instances are immutable but, because ``options`` is a mapping, not
    hashable.
    """

    method: FitMethod | None = None
    stability: Literal["auto", "check", False] = False
    rescale_data: bool = True
    max_jacobian_elements_for_svd: int = 10_000_000
    multistart: Multistart = field(default_factory=lambda: Multistart(enabled=False))
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_options(self, _IN_MEMORY_RESERVED)


@dataclass(frozen=True)
class Chunked:
    """Fit by processing the data in chunks.

    Parameters
    ----------
    chunk_size : int or None
        Points per chunk; ``None`` chooses automatically.
    method : FitMethod or None
        Optimizer method; ``None`` uses the default.
    multistart : Multistart
        Multi-start settings (disabled by default).
    options : Mapping
        Extra keyword options forwarded to the fit. Keys that this class or
        the fit sets itself (its fields, ``sigma``, ``bounds``,
        ``absolute_sigma``, ``check_finite`` and the multi-start keywords)
        raise ``TypeError``. Stored as a private copy (so the spec stays picklable and
        deep-copyable).

    Notes
    -----
    Instances are immutable but, because ``options`` is a mapping, not
    hashable.
    """

    chunk_size: int | None = None
    method: FitMethod | None = None
    multistart: Multistart = field(default_factory=lambda: Multistart(enabled=False))
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_options(self, _CHUNKED_RESERVED)


@dataclass(frozen=True)
class Hybrid:
    """Adaptive hybrid streaming fit.

    Parameters
    ----------
    config : HybridStreamingConfig or None
        Hybrid streaming configuration; ``None`` uses the defaults.
    verbose : int
        Verbosity level.
    """

    config: HybridStreamingConfig | None = None
    verbose: int = 1


@dataclass(frozen=True)
class Exact:
    """Exact chunk-accumulated Levenberg-Marquardt fit.

    Parameters
    ----------
    chunk_size : int or None
        Points per chunk; ``None`` chooses automatically.
    ftol, xtol, gtol : float
        Termination tolerances.
    max_nfev : int or None
        Maximum number of function evaluations.
    n_points : int or None
        Total number of points, when the data source cannot report it.
    """

    chunk_size: int | None = None
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int | None = None
    n_points: int | None = None


@dataclass(frozen=True)
class Auto:
    """Choose between in-memory and chunked fitting by dataset size.

    Parameters
    ----------
    size_threshold : int
        Point count at and above which the chunked strategy is used.
    auto_size_detection : bool
        If False, always use the chunked strategy.
    in_memory : InMemory
        Strategy used below the threshold.
    chunked : Chunked
        Strategy used at or above the threshold.
    """

    size_threshold: int = 1_000_000
    auto_size_detection: bool = True
    in_memory: InMemory = field(default_factory=InMemory)
    chunked: Chunked = field(default_factory=Chunked)

    def resolve(self, n_points: int) -> InMemory | Chunked:
        """Return the concrete strategy for ``n_points`` data points."""
        if self.auto_size_detection and n_points < self.size_threshold:
            return self.in_memory
        return self.chunked


LargeStrategy = Auto | InMemory | Chunked | Hybrid | Exact
