"""Benchmark problems. Models take the array module `xp` (numpy or jax.numpy)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    p_true: tuple[float, ...]
    p0: tuple[float, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None
    x_range: tuple[float, float]
    model: Callable[..., Any]  # model(xp, x, *params)

    def _x(self, n: int) -> np.ndarray:
        return np.linspace(*self.x_range, n)

    def data(
        self, n: int, seed: int = 0, noise: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Noisy observations `(x, y)` at the true parameters."""
        x = self._x(n)
        rng = np.random.default_rng(seed)
        return x, self.model(np, x, *self.p_true) + noise * rng.standard_normal(n)

    def batch_data(
        self, n: int, batch: int, seed: int = 0, noise: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared `x` (n,) and `Y` (batch, n) with independent noise rows."""
        x = self._x(n)
        rng = np.random.default_rng(seed)
        clean = self.model(np, x, *self.p_true)
        return x, clean + noise * rng.standard_normal((batch, n))


def _exp_decay(xp, x, a, b, c):
    return a * xp.exp(-b * x) + c


def _gaussian(xp, x, a, mu, s):
    return a * xp.exp(-((x - mu) ** 2) / (2 * s**2))


def _two_exp(xp, x, a1, k1, a2, k2):
    return a1 * xp.exp(-k1 * x) + a2 * xp.exp(-k2 * x)


_EXP = {"p_true": (2.5, 1.3, 0.5), "p0": (1.0, 1.0, 0.0), "x_range": (0.0, 4.0)}

PROBLEMS: dict[str, Problem] = {
    "exp_decay": Problem("exp_decay", bounds=None, model=_exp_decay, **_EXP),
    "gaussian": Problem(
        "gaussian",
        p_true=(3.0, 1.0, 0.5),
        p0=(1.0, 0.0, 1.0),
        bounds=None,
        x_range=(-3.0, 5.0),
        model=_gaussian,
    ),
    "two_exp": Problem(
        "two_exp",
        p_true=(2.0, 1.0, 1.5, 1.5),
        p0=(1.0, 0.5, 1.0, 2.0),
        bounds=None,
        x_range=(0.0, 5.0),
        model=_two_exp,
    ),
    # Upper bound on `a` (2.0) is below the true 2.5, so it is active at the optimum.
    "bounded_exp": Problem(
        "bounded_exp",
        bounds=((0.0, 0.0, 0.0), (2.0, 3.0, 1.0)),
        model=_exp_decay,
        **_EXP,
    ),
}
