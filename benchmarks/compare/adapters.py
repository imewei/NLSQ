"""Library adapters: one call convention per library, shared tolerance and budget.

Every library gets `TOL` for each convergence criterion it exposes and `MAX_ITER`
as its iteration budget; criteria a library cannot take are noted in
`Capabilities.note`. Library imports are lazy so a broken install is reported by
`availability()` instead of breaking the harness.
"""

import functools
import importlib.metadata as md
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

import nlsq

from .problems import PROBLEMS, Problem
from .types import FitOutcome

TOL = 1e-8
MAX_ITER = 1000


@dataclass(frozen=True)
class Capabilities:
    bounds: bool
    sigma_weights: bool
    covariance: bool
    vmappable: bool
    maintained: bool
    note: str


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str
    version: str | None


class Adapter(Protocol):
    name: str
    capabilities: Capabilities
    modes: tuple[str, ...]

    def availability(self) -> Availability: ...
    def supports(self, problem: Problem) -> tuple[bool, str]: ...
    def prepare(
        self, problem: Problem, x: np.ndarray, y: np.ndarray, mode: str
    ) -> Callable[[], FitOutcome]: ...
    def prepare_batch(
        self, problem: Problem, X: np.ndarray, Y: np.ndarray, mode: str
    ) -> Callable[[], list[FitOutcome]]: ...


def _np_model(problem: Problem) -> Callable:
    return lambda x, *p: problem.model(np, x, *p)


def _jnp_model(problem: Problem) -> Callable:
    return lambda x, *p: problem.model(jnp, x, *p)


class _Base:
    name: str
    capabilities: Capabilities
    modes: tuple[str, ...] = ("end_to_end",)
    dist: str  # distribution name for importlib.metadata

    def _fit(self, problem: Problem, x, y) -> FitOutcome:
        raise NotImplementedError

    def supports(self, problem: Problem) -> tuple[bool, str]:
        if problem.bounds is not None and not self.capabilities.bounds:
            return False, "no bounds support"
        return True, ""

    def prepare(self, problem, x, y, mode):
        return lambda: self._fit(problem, x, y)

    def prepare_batch(self, problem, X, Y, mode):
        runs = [self.prepare(problem, X, y, mode) for y in Y]
        return lambda: [run() for run in runs]

    @functools.cache
    def availability(self) -> Availability:
        try:
            version = md.version(self.dist)
            x, y = PROBLEMS["exp_decay"].data(50)
            self.prepare(PROBLEMS["exp_decay"], x, y, "end_to_end")()
        except Exception as e:
            return Availability(False, f"{type(e).__name__}: {e}", None)
        return Availability(True, "", version)


def _bounds(problem: Problem):
    return problem.bounds if problem.bounds is not None else (-np.inf, np.inf)


class _NLSQ(_Base):
    name, dist = "nlsq", "nlsq"
    capabilities = Capabilities(True, True, True, False, True, "")

    def _fit(self, problem, x, y):
        r = nlsq.curve_fit(
            _jnp_model(problem),
            x,
            y,
            p0=problem.p0,
            bounds=_bounds(problem),
            ftol=TOL,
            xtol=TOL,
            gtol=TOL,
            max_nfev=MAX_ITER,
            full_output=True,
        )
        popt, _, info, _, ier = r
        return FitOutcome(np.asarray(popt), ier in (1, 2, 3, 4), int(info["nfev"]))


class _SciPy(_Base):
    name, dist = "scipy", "scipy"
    capabilities = Capabilities(True, True, True, False, True, "")

    def _fit(self, problem, x, y):
        from scipy.optimize import curve_fit

        popt, _, info, _, ier = curve_fit(
            _np_model(problem),
            x,
            y,
            p0=problem.p0,
            bounds=_bounds(problem),
            method="trf",
            ftol=TOL,
            xtol=TOL,
            gtol=TOL,
            max_nfev=MAX_ITER,
            full_output=True,
        )
        return FitOutcome(popt, ier in (1, 2, 3, 4), int(info["nfev"]))


class _LMFit(_Base):
    name, dist = "lmfit", "lmfit"
    capabilities = Capabilities(
        True, True, True, False, True, "wraps SciPy leastsq / least_squares"
    )

    def _fit(self, problem, x, y):
        from lmfit import Model

        # lmfit reads parameter names from the signature, so expose the model's own
        names = list(inspect.signature(problem.model).parameters)[2:]

        def f(x, **kw):
            return problem.model(np, x, *(kw[n] for n in names))

        P = inspect.Parameter
        f.__signature__ = inspect.Signature(
            [P("x", P.POSITIONAL_OR_KEYWORD)]
            + [P(n, P.POSITIONAL_OR_KEYWORD) for n in names]
        )
        lo, hi = _bounds(problem)
        lo, hi = np.broadcast_to(lo, len(names)), np.broadcast_to(hi, len(names))
        model = Model(f, independent_vars=["x"])
        params = model.make_params(
            **{
                n: {"value": v, "min": float(l), "max": float(h)}
                for n, v, l, h in zip(names, problem.p0, lo, hi, strict=True)
            }
        )
        bounded = problem.bounds is not None
        r = model.fit(
            y,
            params,
            x=x,
            method="least_squares" if bounded else "leastsq",
            max_nfev=MAX_ITER,
            fit_kws={"ftol": TOL, "xtol": TOL, "gtol": TOL},
        )
        return FitOutcome(
            np.array([r.params[n].value for n in names]), bool(r.success), int(r.nfev)
        )


class _JaxBase(_Base):
    """Shared jit/vmap plumbing: subclasses supply `_core(problem)`.

    `_core` returns `f(p0, x, y) -> (params, converged)` on device arrays.
    `jitted_core` keeps inputs resident on device; both modes read the small
    result back after `jax.block_until_ready`.
    """

    modes = ("end_to_end", "jitted_core")

    def _core(self, problem: Problem) -> Callable:
        raise NotImplementedError

    def _runner(self, problem, x, y, mode, batched):
        core = self._core(problem)
        fn = jax.jit(jax.vmap(core, (None, None, 0)) if batched else core)
        p0 = jnp.asarray(problem.p0)
        if mode == "jitted_core":
            x, y = jax.device_put(x), jax.device_put(y)

        def run():
            params, ok = fn(p0, jnp.asarray(x), jnp.asarray(y))
            params, ok = jax.block_until_ready((params, ok))
            return np.asarray(params), np.asarray(ok)

        return run

    def prepare(self, problem, x, y, mode):
        run = self._runner(problem, x, y, mode, batched=False)

        def fit():
            params, ok = run()
            return FitOutcome(params, bool(ok), None)

        return fit

    def prepare_batch(self, problem, X, Y, mode):
        run = self._runner(problem, X, Y, mode, batched=True)

        def fit():
            params, ok = run()
            return [
                FitOutcome(p, bool(c), None) for p, c in zip(params, ok, strict=True)
            ]

        return fit


class _JAXopt(_JaxBase):
    name, dist = "jaxopt", "jaxopt"
    capabilities = Capabilities(
        False,
        False,
        False,
        True,
        False,
        'README: "JAXopt is no longer maintained nor developed"; '
        "tol/xtol/gtol all honoured",
    )

    def _core(self, problem):
        import jaxopt

        model = _jnp_model(problem)
        solver = jaxopt.LevenbergMarquardt(
            residual_fun=lambda p, x, y: y - model(x, *p),
            tol=TOL,
            xtol=TOL,
            gtol=TOL,
            maxiter=MAX_ITER,
        )

        def core(p0, x, y):
            r = solver.run(p0, x, y)
            # any of tol/xtol/gtol stops the loop early; hitting maxiter means not converged
            return r.params, r.state.iter_num < MAX_ITER

        return core


class _Optimistix(_JaxBase):
    name, dist = "optimistix", "optimistix"
    capabilities = Capabilities(
        False, False, False, True, True, "rtol/atol only; no separate gtol"
    )

    def _core(self, problem):
        import optimistix as optx

        model = _jnp_model(problem)
        solver = optx.LevenbergMarquardt(rtol=TOL, atol=TOL)

        def core(p0, x, y):
            sol = optx.least_squares(
                lambda p, args: args[1] - model(args[0], *p),
                solver,
                p0,
                args=(x, y),
                max_steps=MAX_ITER,
                throw=False,
            )
            return sol.value, sol.result == optx.RESULTS.successful

        return core


ADAPTERS: dict[str, Adapter] = {
    a.name: a for a in (_NLSQ(), _SciPy(), _LMFit(), _JAXopt(), _Optimistix())
}
