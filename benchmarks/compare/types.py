"""Shared result types for the comparison harness."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Status = Literal["ok", "unsupported", "unavailable", "did_not_reach_optimum"]


@dataclass(frozen=True)
class FitOutcome:
    """Result of one fit: parameters, convergence flag, function evaluations."""

    params: np.ndarray
    converged: bool
    nfev: int | None


@dataclass(frozen=True)
class Timing:
    """Cold call time plus median and IQR over warm repeats, in seconds."""

    cold_s: float
    warm_median_s: float
    warm_iqr_s: float


Outcomes = FitOutcome | list[FitOutcome]
