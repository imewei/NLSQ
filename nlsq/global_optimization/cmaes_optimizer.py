"""CMA-ES global optimizer with NLSQ refinement.

This module provides the CMAESOptimizer class that runs CMA-ES global search
using evosax followed by NLSQ Trust Region Reflective refinement for proper
parameter covariance estimation.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization.bounds_transform import (
    compute_default_popsize,
    transform_from_bounds,
    transform_to_bounds,
)
from nlsq.global_optimization.cmaes_config import (
    CMAESConfig,
    is_evosax_available,
)
from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from nlsq.global_optimization.checkpoint import HPCCheckpointManager

__all__ = ["CMAESOptimizer", "CMAESPreempted"]

logger = logging.getLogger(__name__)

# BIPOP exploratory-restart sampling half-range, in unbounded (sigmoid)
# space. +-2.0 (sigmoid(2)~=0.88) permanently excludes the outer ~12% of
# each bound from ever being an explore-restart center; +-4.6
# (sigmoid(4.6)~=0.99) recovers ~11 of those 12 points, leaving only the
# outer ~1% unreachable.
_BIPOP_EXPLORE_RANGE = 4.6


class CMAESPreempted(SystemExit):
    """Raised when a preemption signal (SIGTERM/SIGUSR1) is caught after a
    checkpoint has been safely written. Exit code 75 lets a wrapping HPC
    resubmission script distinguish a clean checkpointed stop from a crash."""

    def __init__(self, generation: int) -> None:
        super().__init__(75)
        self.generation = generation


def _create_fitness_function(  # noqa: C901
    model_func: Callable,
    xdata: jax.Array,
    ydata: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
    sigma: jax.Array | None = None,
    population_batch_size: int | None = None,
    data_chunk_size: int | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Create a fitness function for CMA-ES optimization.

    evosax's CMA-ES minimizes fitness (best_solution/best_fitness are tracked
    via argmin, and default fitness-shaping ranks ascending), so we return the
    raw SSR (sum of squared residuals) directly -- do not negate it.

    Parameters
    ----------
    model_func : Callable
        Model function f(x, *params) -> y.
    xdata : jax.Array
        Independent variable data.
    ydata : jax.Array
        Dependent variable data.
    lower_bounds : jax.Array
        Lower bounds for parameters.
    upper_bounds : jax.Array
        Upper bounds for parameters.
    sigma : jax.Array | None, optional
        Standard deviation of ydata for weighted residuals.
    population_batch_size : int | None, optional
        Batch size for population evaluation to avoid OOM.
    data_chunk_size : int | None, optional
        Chunk size for data streaming to avoid OOM on large datasets.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        Fitness function that takes unbounded parameters and returns fitness.
    """
    n_data = xdata.shape[0]

    # Determine if we need data streaming
    use_data_streaming = data_chunk_size is not None and n_data > data_chunk_size

    if use_data_streaming:
        # Mypy doesn't infer not-None from the boolean flag
        assert data_chunk_size is not None

        # Calculate number of full chunks and remainder
        n_full_chunks = n_data // data_chunk_size
        remainder = n_data % data_chunk_size

        # Pad data to exact multiple of chunk_size for efficient slicing
        if remainder > 0:
            pad_size = data_chunk_size - remainder
            xdata_padded = jnp.pad(xdata, (0, pad_size), constant_values=0.0)
            ydata_padded = jnp.pad(ydata, (0, pad_size), constant_values=0.0)
            if sigma is not None:
                # Pad sigma with 1.0 to avoid division issues (residual will be 0)
                sigma_padded = jnp.pad(sigma, (0, pad_size), constant_values=1.0)
            else:
                sigma_padded = None
            n_chunks = n_full_chunks + 1
        else:
            xdata_padded = xdata
            ydata_padded = ydata
            sigma_padded = sigma
            n_chunks = n_full_chunks

        # Reshape data into chunks for efficient access
        xdata_chunked = xdata_padded.reshape(n_chunks, data_chunk_size)
        ydata_chunked = ydata_padded.reshape(n_chunks, data_chunk_size)
        if sigma_padded is not None:
            sigma_chunked = sigma_padded.reshape(n_chunks, data_chunk_size)
        else:
            sigma_chunked = None

        # Create validity mask for the last chunk (handles padding)
        if remainder > 0:
            last_chunk_mask = jnp.arange(data_chunk_size) < remainder
        else:
            last_chunk_mask = jnp.ones(data_chunk_size, dtype=bool)

        @jax.jit
        def compute_chunk_ssr(
            params_bounded: jax.Array,
            x_chunk: jax.Array,
            y_chunk: jax.Array,
            sigma_chunk: jax.Array | None,
            valid_mask: jax.Array,
        ) -> jax.Array:
            """Compute SSR for one data chunk."""
            predictions = model_func(x_chunk, *params_bounded)
            residuals = y_chunk - predictions

            if sigma_chunk is not None:
                residuals = residuals / sigma_chunk

            # Apply validity mask to handle padding in last chunk
            residuals_sq = jnp.where(valid_mask, residuals**2, 0.0)
            return jnp.sum(residuals_sq)

        def fitness_single_streaming(params_unbounded: jax.Array) -> jax.Array:
            """Compute fitness by streaming over data chunks."""
            params_bounded = transform_to_bounds(
                params_unbounded,
                lower_bounds,
                upper_bounds,
            )

            # Accumulate SSR over chunks
            ssr_total = jnp.array(0.0)

            for chunk_idx in range(n_chunks):
                x_chunk = xdata_chunked[chunk_idx]
                y_chunk = ydata_chunked[chunk_idx]
                sigma_chunk = (
                    sigma_chunked[chunk_idx] if sigma_chunked is not None else None
                )

                # Use appropriate mask for last chunk
                if chunk_idx == n_chunks - 1 and remainder > 0:
                    valid_mask = last_chunk_mask
                else:
                    valid_mask = jnp.ones(data_chunk_size, dtype=bool)

                ssr_total = ssr_total + compute_chunk_ssr(
                    params_bounded,
                    x_chunk,
                    y_chunk,
                    sigma_chunk,
                    valid_mask,
                )

            return jnp.where(jnp.isfinite(ssr_total), ssr_total, jnp.inf)

        fitness_single = fitness_single_streaming

        logger.debug(
            f"Data streaming enabled: {n_data} points -> {n_chunks} chunks of {data_chunk_size}",
        )
    else:
        # Original non-streaming fitness function
        @jax.jit
        def fitness_single(params_unbounded: jax.Array) -> jax.Array:
            """Compute fitness for a single parameter set."""
            # Transform to bounded space
            params_bounded = transform_to_bounds(
                params_unbounded,
                lower_bounds,
                upper_bounds,
            )

            # Compute predictions
            predictions = model_func(xdata, *params_bounded)

            # Compute residuals
            residuals = ydata - predictions

            # Weight by sigma if provided
            if sigma is not None:
                residuals = residuals / sigma

            # Sum of squared residuals
            ssr = jnp.sum(residuals**2)

            # Handle NaN/Inf (assign worst fitness)
            fitness = jnp.where(jnp.isfinite(ssr), ssr, jnp.inf)

            return fitness

    @jax.jit
    def fitness_population_jit(population: jax.Array) -> jax.Array:
        """Compute fitness for entire population (vectorized)."""
        return jax.vmap(fitness_single)(population)

    if population_batch_size is None:
        return fitness_population_jit

    def fitness_population_batched(population: jax.Array) -> jax.Array:
        """Compute fitness for population in batches (sequential loop)."""
        n = population.shape[0]
        # If population fits in one batch, run directly
        if n <= population_batch_size:
            return fitness_population_jit(population)

        results = []
        for i in range(0, n, population_batch_size):
            batch = population[i : i + population_batch_size]
            results.append(fitness_population_jit(batch))

        return jnp.concatenate(results)

    return fitness_population_batched


def _require_finite_bounds(lower_bounds: jax.Array, upper_bounds: jax.Array) -> None:
    """Raise ``ValueError`` unless every bound -- and every range -- is finite.

    ``transform_to_bounds`` computes ``lb + (ub - lb) * sigmoid(x)``: any
    non-finite bound (or a finite-but-overflowing range, e.g.
    ``lb=-1e308``/``ub=1e308``) turns every candidate's bounded parameters
    into NaN/inf, so CMA-ES would silently search a NaN fitness landscape
    forever. ``MethodSelector``'s scale-ratio heuristic can even prefer
    ``"cmaes"`` for a partially-unbounded problem (an infinite range
    dominates the ratio), so this must be enforced here -- the one place
    every CMA-ES entry point (direct API use, ``method="cmaes"``, and
    MethodSelector-selected ``"auto"``) funnels through -- not re-derived
    per caller.
    """
    finite = (
        jnp.isfinite(lower_bounds)
        & jnp.isfinite(upper_bounds)
        & jnp.isfinite(upper_bounds - lower_bounds)
    )
    if not bool(jnp.all(finite)):
        raise ValueError(
            "CMA-ES requires every parameter to have a finite lower AND "
            "upper bound (no +/-inf, and no overflowing range). Got "
            f"lower={np.asarray(lower_bounds)}, upper={np.asarray(upper_bounds)}.",
        )


def _finite_fitness_spread(fitness: jax.Array) -> float:
    """BIPOP stagnation spread: max - min over finite fitness values only.

    A single diverging candidate (fitness = inf, assigned by the fitness
    function on NaN/Inf residuals) would otherwise make max(fitness) inf
    and the spread inf forever, permanently masking real stagnation in the
    rest of the population. Falls back to 0.0 ("stagnant") if every
    candidate in the generation failed.
    """
    finite_fitness = fitness[jnp.isfinite(fitness)]
    if finite_fitness.size == 0:
        return 0.0
    return float(jnp.max(finite_fitness) - jnp.min(finite_fitness))


def _warn_if_never_finite(best_fitness: float, generations: int) -> None:
    """Log a warning if CMA-ES never found a finite fitness in the whole run.

    If every candidate evaluated across the whole run (all generations, all
    BIPOP restarts) produced a non-finite fitness -- typically the model
    itself is NaN/Inf everywhere in the (now bounds-validated finite)
    search region, e.g. a sqrt/log of a value that goes negative across the
    whole box -- ``best_params`` is an arbitrary, unconverged point, not a
    real optimum. Surface that instead of returning a normal-looking result
    built on it silently.
    """
    if not np.isfinite(best_fitness):
        logger.warning(
            f"CMA-ES never found a finite fitness across {generations} "
            "generation(s) -- best_params is unconverged/arbitrary, not "
            "a genuine optimum. Check that the model function produces "
            "finite output across the full parameter bounds.",
        )


class CMAESOptimizer:
    """CMA-ES global optimizer with NLSQ refinement using evosax.

    Uses evosax's CMA-ES implementation for gradient-free global optimization,
    followed by NLSQ Trust Region Reflective refinement for proper parameter
    covariance estimation.

    Parameters
    ----------
    config : CMAESConfig | None, optional
        Configuration for CMA-ES optimization. If None, uses default config.

    Attributes
    ----------
    config : CMAESConfig
        Configuration for CMA-ES optimization.

    Examples
    --------
    >>> from nlsq.global_optimization import CMAESOptimizer, CMAESConfig
    >>> import jax.numpy as jnp
    >>>
    >>> def model(x, a, b):
    ...     return a * jnp.exp(-b * x)
    >>>
    >>> x = jnp.linspace(0, 5, 100)
    >>> y = 2.5 * jnp.exp(-0.5 * x)
    >>> bounds = ([0.1, 0.01], [10.0, 2.0])
    >>>
    >>> optimizer = CMAESOptimizer()
    >>> result = optimizer.fit(model, x, y, bounds=bounds)
    >>> print(f"Optimal params: {result['popt']}")
    """

    def __init__(self, config: CMAESConfig | None = None) -> None:
        """Initialize CMAESOptimizer.

        Parameters
        ----------
        config : CMAESConfig | None, optional
            Configuration for CMA-ES optimization. If None, uses default config
            (BIPOP enabled, 100 generations, 9 max restarts).
        """
        self.config = config if config is not None else CMAESConfig()

        # Verify evosax is available
        if not is_evosax_available():
            raise ImportError(
                "evosax is required for CMA-ES optimization. "
                "Install with: pip install 'nlsq[global]'",
            )

    @classmethod
    def from_preset(cls, preset_name: str) -> CMAESOptimizer:
        """Create optimizer from a named preset.

        Parameters
        ----------
        preset_name : str
            Name of the preset. One of 'cmaes-fast', 'cmaes', 'cmaes-global'.

        Returns
        -------
        CMAESOptimizer
            Optimizer configured with the specified preset.

        Examples
        --------
        >>> optimizer = CMAESOptimizer.from_preset('cmaes-fast')
        >>> optimizer.config.max_generations
        50
        """
        config = CMAESConfig.from_preset(preset_name)
        return cls(config=config)

    def fit(
        self,
        f: Callable,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: ArrayLike | None = None,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        sigma: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run CMA-ES global optimization followed by NLSQ refinement.

        Parameters
        ----------
        f : Callable
            Model function ``f(x, *params) -> y``.
        xdata : ArrayLike
            Independent variable data.
        ydata : ArrayLike
            Dependent variable data.
        p0 : ArrayLike | None, optional
            Initial parameter guess. If None, uses center of bounds.
        bounds : tuple[ArrayLike, ArrayLike] | None
            Lower and upper bounds for parameters. Required for CMA-ES.
        sigma : ArrayLike | None, optional
            Standard deviation of ydata for weighted residuals.
        **kwargs : Any
            Additional keyword arguments (passed to NLSQ refinement).

        Returns
        -------
        dict[str, Any]
            Result dictionary containing:
            - popt: Optimal parameters
            - pcov: Parameter covariance matrix (from NLSQ refinement)
            - Additional fields from NLSQ result

        Raises
        ------
        ValueError
            If bounds are not provided (required for CMA-ES).
        """
        # Validate bounds
        if bounds is None:
            raise ValueError(
                "CMA-ES requires explicit bounds. "
                "Provide bounds as (lower_bounds, upper_bounds).",
            )

        # Convert inputs to JAX arrays
        xdata_jax = jnp.asarray(xdata)
        ydata_jax = jnp.asarray(ydata)
        # bounds may be SciPy-style scalars (e.g. (0, 10)) meant to broadcast
        # to every parameter. Resize using p0's length when available -- same
        # convention prepare_bounds() uses elsewhere in this codebase.
        lower_bounds = jnp.asarray(bounds[0], dtype=float)
        upper_bounds = jnp.asarray(bounds[1], dtype=float)
        n_params_hint = None
        if p0 is not None:
            n_params_hint = len(jnp.atleast_1d(jnp.asarray(p0)))
        elif lower_bounds.ndim == 0 or upper_bounds.ndim == 0:
            # p0 is None and bounds are scalar: this method's docstring
            # promises p0 defaults to the bounds center, which needs
            # n_params from somewhere. Infer it from f's signature (same
            # pattern used in large_dataset.py's LargeDatasetFitter).
            try:
                from inspect import signature

                n_params_hint = len(signature(f).parameters) - 1
            except (TypeError, ValueError):
                n_params_hint = None
            if n_params_hint is None or n_params_hint < 1:
                raise ValueError(
                    "CMA-ES needs p0 or array-shaped bounds to determine "
                    "the number of parameters (got scalar bounds, no p0, "
                    "and f's signature could not be inspected)",
                )
        if n_params_hint is not None:
            if lower_bounds.ndim == 0:
                lower_bounds = jnp.full(n_params_hint, lower_bounds)
            if upper_bounds.ndim == 0:
                upper_bounds = jnp.full(n_params_hint, upper_bounds)

        _require_finite_bounds(lower_bounds, upper_bounds)
        sigma_jax = jnp.asarray(sigma) if sigma is not None else None

        n_params = len(lower_bounds)
        n_data = len(ydata_jax)

        # Log initialization
        logger.info(
            f"CMA-ES optimizer initialized: n_params={n_params}, n_data={n_data}, "
            f"restart_strategy={self.config.restart_strategy}",
        )
        logger.debug(
            f"CMA-ES bounds: lower={np.asarray(lower_bounds)}, "
            f"upper={np.asarray(upper_bounds)}",
        )

        # Determine population size
        popsize = self.config.popsize
        if popsize is None:
            popsize = compute_default_popsize(n_params)

            # Double population for cmaes-global preset
            # (detected by max_generations == 200 and bipop). Only applies
            # to the auto-computed default -- an explicit user popsize must
            # never be silently doubled (matches the reference logic in
            # nlsq/core/minpack.py's own auto-memory popsize estimate).
            if (
                self.config.max_generations == 200
                and self.config.restart_strategy == "bipop"
            ):
                popsize = popsize * 2
                logger.debug("CMA-ES: Using 2x population for cmaes-global preset")

        # Log memory optimization settings
        if self.config.population_batch_size is not None:
            logger.info(
                f"CMA-ES memory optimization: population_batch_size="
                f"{self.config.population_batch_size}",
            )
        if self.config.data_chunk_size is not None:
            logger.info(
                f"CMA-ES memory optimization: data_chunk_size="
                f"{self.config.data_chunk_size} (data streaming enabled)",
            )

        # Determine initial solution
        if p0 is not None:
            p0_jax = jnp.asarray(p0)
            # Transform to unbounded space
            initial_solution = transform_from_bounds(p0_jax, lower_bounds, upper_bounds)
            logger.debug(f"CMA-ES starting from p0={np.asarray(p0_jax)}")
        else:
            # Start at center of bounds (x=0 in unbounded space = midpoint)
            initial_solution = jnp.zeros(n_params)
            midpoint = (lower_bounds + upper_bounds) / 2
            logger.debug(f"CMA-ES starting from bounds midpoint={np.asarray(midpoint)}")

        # Create fitness function
        fitness_fn = _create_fitness_function(
            f,
            xdata_jax,
            ydata_jax,
            lower_bounds,
            upper_bounds,
            sigma_jax,
            population_batch_size=self.config.population_batch_size,
            data_chunk_size=self.config.data_chunk_size,
        )

        checkpoint_manager, checkpoint_path, checkpoint_fingerprint = (
            self._setup_checkpointing(
                xdata_jax, ydata_jax, sigma_jax, lower_bounds, upper_bounds, popsize
            )
        )

        # Track wall time
        import time

        start_time = time.perf_counter()

        # Initialize diagnostics
        diagnostics = CMAESDiagnostics()

        # Run CMA-ES optimization (diagnostics updated in place)
        with self._preemption_handling(checkpoint_manager) as preemption_event:
            best_params_unbounded, best_fitness, generations = self._run_cmaes(
                fitness_fn,
                initial_solution,
                popsize,
                n_params,
                diagnostics,
                checkpoint_manager=checkpoint_manager,
                checkpoint_path=checkpoint_path,
                checkpoint_fingerprint=checkpoint_fingerprint,
                preemption_event=preemption_event,
            )

        # Update diagnostics
        diagnostics.total_generations = generations
        diagnostics.best_fitness = float(best_fitness)
        diagnostics.wall_time = time.perf_counter() - start_time

        # Transform best solution back to bounded space
        best_params = transform_to_bounds(
            best_params_unbounded,
            lower_bounds,
            upper_bounds,
        )

        logger.info(
            f"CMA-ES optimization completed: {generations} generations, "
            f"best_fitness={float(best_fitness):.6e}, "
            f"wall_time={diagnostics.wall_time:.2f}s",
        )
        _warn_if_never_finite(diagnostics.best_fitness, generations)

        # NLSQ refinement phase for proper pcov estimation
        if self.config.refine_with_nlsq:
            result = self._nlsq_refinement(
                f,
                xdata,
                ydata,
                best_params,
                bounds,
                sigma,
                **kwargs,
            )
            diagnostics.nlsq_refinement = True
        else:
            # Return CMA-ES result without refinement
            result = {
                "popt": np.asarray(best_params),
                "pcov": self._estimate_pcov_from_cmaes(n_params),
            }
            diagnostics.nlsq_refinement = False

        # Add diagnostics to result
        result["cmaes_diagnostics"] = diagnostics.to_dict()

        return result

    def _setup_checkpointing(
        self,
        xdata_jax: jax.Array,
        ydata_jax: jax.Array,
        sigma_jax: jax.Array | None,
        lower_bounds: jax.Array,
        upper_bounds: jax.Array,
        popsize: int,
    ) -> tuple[HPCCheckpointManager | None, Path | None, dict[str, Any] | None]:
        """Build the checkpoint manager/path/fingerprint for `fit()`, or
        `(None, None, None)` when `checkpoint_dir` is unset.

        `self.config.model_id`/`self.config.run_id` are guaranteed non-`None`
        here -- `CMAESConfig` validation raises at construction time if
        `checkpoint_dir` is set without them.
        """
        if self.config.checkpoint_dir is None:
            return None, None, None

        from nlsq.global_optimization.checkpoint import (
            HPCCheckpointManager,
            compute_fingerprint,
        )

        assert self.config.model_id is not None
        assert self.config.run_id is not None

        manager = HPCCheckpointManager()
        path = Path(self.config.checkpoint_dir) / f"{self.config.run_id}.h5"
        fingerprint = compute_fingerprint(
            model_id=self.config.model_id,
            xdata=np.asarray(xdata_jax),
            ydata=np.asarray(ydata_jax),
            sigma=np.asarray(sigma_jax) if sigma_jax is not None else None,
            bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
            config_fields={
                "popsize": popsize,
                "sigma": self.config.sigma,
                "tol_fun": self.config.tol_fun,
                "tol_x": self.config.tol_x,
                "seed": self.config.seed,
            },
        )
        return manager, path, fingerprint

    @contextlib.contextmanager
    def _preemption_handling(
        self,
        checkpoint_manager: HPCCheckpointManager | None,
    ) -> Iterator[threading.Event | None]:
        """Register SIGTERM/SIGUSR1 handlers for `fit()` when checkpointing
        is enabled and we're on the main thread, yielding the preemption
        event (or `None` if unavailable); restores the original handlers
        on exit, however `fit()`'s `_run_cmaes(...)` call inside the `with`
        block returns or raises (including via `CMAESPreempted`, a
        `SystemExit` subclass).
        """
        preemption_event: threading.Event | None = None
        previous_handlers: dict[int, Any] = {}
        if checkpoint_manager is None:
            yield preemption_event
            return

        if threading.current_thread() is not threading.main_thread():
            # signal.signal() raises ValueError off the main thread. Periodic
            # interval saves (Task 4) remain the crash-safety net when
            # CMAESOptimizer.fit() is called from a worker thread; this is a
            # documented reduction in coverage, not a silent no-op -- log it.
            logger.warning(
                "Checkpointing enabled but fit() was not called on the main "
                "thread: SIGTERM/SIGUSR1 preemption handling is unavailable "
                "here (Python's signal.signal() requires the main thread). "
                "Periodic interval saves still apply.",
            )
            yield preemption_event
            return

        if not hasattr(signal, "SIGUSR1"):
            # SIGUSR1 doesn't exist on Windows (signal.SIGTERM does, but
            # Windows' signal handling is not POSIX-equivalent either way).
            # Same degradation as the off-main-thread case above: log and
            # fall back to periodic interval saves rather than crashing on
            # an AttributeError for an attribute that's simply absent here.
            logger.warning(
                "Checkpointing enabled but SIGUSR1 is not available on this "
                "platform (Windows): SIGTERM/SIGUSR1 preemption handling is "
                "unavailable here. Periodic interval saves still apply.",
            )
            preemption_event = None
            yield preemption_event
            return

        preemption_event = threading.Event()

        def _handle_preemption(signum: int, frame: Any) -> None:
            preemption_event.set()

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[sig] = signal.signal(sig, _handle_preemption)
        try:
            yield preemption_event
        finally:
            for restore_sig, handler in previous_handlers.items():
                signal.signal(restore_sig, handler)

    def _run_cmaes(
        self,
        fitness_fn: Callable,
        initial_solution: jax.Array,
        popsize: int,
        n_params: int,
        diagnostics: CMAESDiagnostics,
        *,
        checkpoint_manager: HPCCheckpointManager | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: dict[str, Any] | None = None,
        preemption_event: threading.Event | None = None,
    ) -> tuple[jax.Array, jax.Array, int]:
        """Run CMA-ES optimization loop with optional BIPOP restarts.

        Parameters
        ----------
        fitness_fn : Callable
            Fitness function for population evaluation.
        initial_solution : jax.Array
            Initial solution in unbounded space.
        popsize : int
            Population size (base population for BIPOP).
        n_params : int
            Number of parameters.
        diagnostics : CMAESDiagnostics
            Diagnostics object to update with run information.

        Returns
        -------
        tuple[jax.Array, jax.Array, int]
            Best solution, best fitness, and total number of generations.
        """
        if self.config.restart_strategy == "bipop":
            return self._run_cmaes_with_bipop(
                fitness_fn,
                initial_solution,
                popsize,
                n_params,
                diagnostics,
            )
        return self._run_cmaes_single(
            fitness_fn,
            initial_solution,
            popsize,
            n_params,
            diagnostics,
            checkpoint_manager=checkpoint_manager,
            checkpoint_path=checkpoint_path,
            checkpoint_fingerprint=checkpoint_fingerprint,
            preemption_event=preemption_event,
        )

    def _run_cmaes_single(
        self,
        fitness_fn: Callable,
        initial_solution: jax.Array,
        popsize: int,
        n_params: int,
        diagnostics: CMAESDiagnostics,
        *,
        checkpoint_manager: HPCCheckpointManager | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: dict[str, Any] | None = None,
        preemption_event: threading.Event | None = None,
    ) -> tuple[jax.Array, jax.Array, int]:
        """Run single CMA-ES optimization without restarts.

        Parameters
        ----------
        fitness_fn : Callable
            Fitness function for population evaluation.
        initial_solution : jax.Array
            Initial solution in unbounded space.
        popsize : int
            Population size.
        n_params : int
            Number of parameters.
        diagnostics : CMAESDiagnostics
            Diagnostics object to update with run information.

        Returns
        -------
        tuple[jax.Array, jax.Array, int]
            Best solution, best fitness, and number of generations.
        """
        from evosax.algorithms import (  # type: ignore[import-not-found,import-untyped]
            CMA_ES,
        )

        from nlsq.global_optimization.checkpoint import (
            CMAESCheckpointState,
            deserialize_evosax_state,
            deserialize_key,
            serialize_evosax_state,
            serialize_key,
        )

        checkpointing = checkpoint_manager is not None

        logger.info(
            f"Starting CMA-ES: popsize={popsize}, max_gen={self.config.max_generations}",
        )

        # Initialize CMA-ES
        es = CMA_ES(population_size=popsize, solution=initial_solution)
        params = es.default_params

        # Set initial sigma
        params = params.replace(std_init=self.config.sigma)

        # Initialize random key
        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(np.random.randint(0, 2**31))

        # Initialize state
        key, subkey = jax.random.split(key)
        state = es.init(subkey, initial_solution, params)

        # Track best solution (evosax/NLSQ fitness is raw SSR: lower is better)
        best_solution = initial_solution
        best_fitness = jnp.array(jnp.inf)
        convergence_reason = "max_generations"
        start_gen = 0

        if checkpointing:
            assert checkpoint_path is not None
            assert checkpoint_manager is not None
            assert checkpoint_fingerprint is not None
            checkpoint_bak_path = checkpoint_manager.bak_path(checkpoint_path)
            # Check for .bak too, not just the primary: if a crash deleted or
            # never finished writing the primary but a prior successful
            # save's rotated .bak survives, resume must still attempt it --
            # gating on the primary alone would silently skip
            # HPCCheckpointManager.load()'s own mandatory .bak fallback (FR8)
            # entirely and start fresh instead (caught in the third review
            # pass).
            if checkpoint_path.exists() or checkpoint_bak_path.exists():
                loaded = checkpoint_manager.load(
                    checkpoint_path, checkpoint_fingerprint
                )
                # serialize_evosax_state duck-types on `loaded`'s field
                # names (a CMAESCheckpointState) exactly like it does on a
                # real evosax State -- single source of truth for the
                # field list instead of hand-writing it again here.
                state = deserialize_evosax_state(
                    serialize_evosax_state(loaded),
                    state,
                )
                key = deserialize_key(loaded.key_data)
                best_solution = loaded.best_solution
                best_fitness = jnp.asarray(loaded.best_fitness)
                diagnostics.fitness_history = list(loaded.fitness_history)
                start_gen = loaded.generation_counter
                logger.info(
                    f"Resumed CMA-ES from checkpoint at generation {start_gen} "
                    f"({checkpoint_path})",
                )

        # Progress milestones for logging (25%, 50%, 75%)
        # Build the dict from a list so later entries don't silently overwrite
        # earlier ones when max_generations is small (e.g. <=3 causes collisions).
        milestones: dict[int, str] = {}
        for pct, label in ((0.25, "25%"), (0.50, "50%"), (0.75, "75%")):
            gen_idx = int(self.config.max_generations * pct)
            if gen_idx not in milestones:
                milestones[gen_idx] = label

        def _save_checkpoint(gen_idx: int, current_key: jax.Array) -> None:
            assert checkpoint_manager is not None
            assert checkpoint_path is not None
            assert checkpoint_fingerprint is not None
            checkpoint_state = CMAESCheckpointState(
                generation_counter=gen_idx + 1,
                mean=state.mean,
                std=state.std,
                p_std=state.p_std,
                p_c=state.p_c,
                C=state.C,
                B=state.B,
                D=state.D,
                best_solution=best_solution,
                best_fitness=float(best_fitness),
                key_data=serialize_key(current_key),
                fitness_history=list(diagnostics.fitness_history),
                popsize=popsize,
            )
            checkpoint_manager.save(
                checkpoint_path, checkpoint_state, checkpoint_fingerprint
            )

        # Main optimization loop
        gen = start_gen - 1
        for gen in range(start_gen, self.config.max_generations):
            key, key_ask, key_tell = jax.random.split(key, 3)

            # Ask for new population
            population, state = es.ask(key_ask, state, params)

            # Evaluate fitness
            fitness = fitness_fn(population)

            # Update CMA-ES state
            state, _metrics = es.tell(key_tell, population, fitness, state, params)

            # Track best (fitness is raw SSR, so lower is better)
            if state.best_fitness < best_fitness:
                best_fitness = state.best_fitness
                best_solution = state.best_solution

            # Record fitness history
            diagnostics.fitness_history.append(float(best_fitness))

            if preemption_event is not None and preemption_event.is_set():
                jax.block_until_ready((state.mean, state.C, state.best_solution))
                _save_checkpoint(gen, key)
                raise CMAESPreempted(gen + 1)

            if checkpointing and (gen + 1) % self.config.checkpoint_interval == 0:
                _save_checkpoint(gen, key)

            # Simple convergence check based on std
            if float(state.std) < self.config.tol_x:
                logger.info(
                    f"CMA-ES converged at generation {gen + 1}: "
                    f"std={float(state.std):.2e} < tol_x={self.config.tol_x:.2e}",
                )
                convergence_reason = "xtol"
                break

            # Log progress at milestones (INFO level)
            if gen + 1 in milestones:
                logger.info(
                    f"CMA-ES progress {milestones[gen + 1]}: "
                    f"gen={gen + 1}/{self.config.max_generations}, "
                    f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.2e}",
                )

            # Log detailed progress at debug level
            if logger.isEnabledFor(logging.DEBUG) and (gen + 1) % 10 == 0:
                logger.debug(
                    f"Generation {gen + 1}/{self.config.max_generations}: "
                    f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.6e}",
                )

        # `gen >= start_gen` is false only when a resumed checkpoint's
        # generation_counter already reached max_generations, so the `for`
        # loop above ran zero iterations (fully-converged-and-resumed edge
        # case) -- skip re-saving a checkpoint that made no new progress.
        if checkpointing and gen >= start_gen:
            _save_checkpoint(gen, key)

        # Update diagnostics
        diagnostics.final_sigma = float(state.std)
        diagnostics.convergence_reason = convergence_reason
        diagnostics.total_restarts = 0

        return best_solution, best_fitness, gen + 1

    def _run_cmaes_with_bipop(
        self,
        fitness_fn: Callable,
        initial_solution: jax.Array,
        base_popsize: int,
        n_params: int,
        diagnostics: CMAESDiagnostics,
    ) -> tuple[jax.Array, jax.Array, int]:
        """Run CMA-ES with BIPOP restart strategy.

        Alternates between large and small population runs, tracking the
        global best across all restarts.

        Parameters
        ----------
        fitness_fn : Callable
            Fitness function for population evaluation.
        initial_solution : jax.Array
            Initial solution in unbounded space.
        base_popsize : int
            Base population size for BIPOP (will be doubled for large runs).
        n_params : int
            Number of parameters.
        diagnostics : CMAESDiagnostics
            Diagnostics object to update with run information.

        Returns
        -------
        tuple[jax.Array, jax.Array, int]
            Best solution, best fitness, and total number of generations.
        """
        from evosax.algorithms import (  # type: ignore[import-not-found,import-untyped]
            CMA_ES,
        )

        from nlsq.global_optimization.bipop import BIPOPRestarter

        logger.info(
            f"Starting CMA-ES with BIPOP: base_popsize={base_popsize}, "
            f"max_restarts={self.config.max_restarts}, max_gen={self.config.max_generations}",
        )

        # Initialize BIPOP restarter
        restarter = BIPOPRestarter(
            base_popsize=base_popsize,
            n_params=n_params,
            max_restarts=self.config.max_restarts,
            min_fitness_spread=self.config.tol_fun,
            seed=self.config.seed,
        )

        # Initialize random key
        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(np.random.randint(0, 2**31))

        total_generations = 0
        convergence_reason = "max_restarts"
        final_sigma = self.config.sigma
        original_solution = initial_solution

        while not restarter.exhausted:
            # Get population size for this run
            popsize = restarter.get_next_popsize()
            run_type = "large" if popsize >= base_popsize * 2 else "small"

            logger.info(
                f"BIPOP restart #{restarter.restart_count + 1}: "
                f"popsize={popsize} ({run_type}), "
                f"max_gen={self.config.max_generations}",
            )

            # Initialize CMA-ES for this run
            es = CMA_ES(population_size=popsize, solution=initial_solution)
            params = es.default_params
            params = params.replace(std_init=self.config.sigma)

            key, subkey = jax.random.split(key)
            state = es.init(subkey, initial_solution, params)

            # Track best for this run (fitness is raw SSR: lower is better)
            run_best_solution = initial_solution
            run_best_fitness = jnp.array(jnp.inf)

            # Run optimization loop
            stagnation_counter = 0
            gen = -1
            for gen in range(self.config.max_generations):
                key, key_ask, key_tell = jax.random.split(key, 3)

                population, state = es.ask(key_ask, state, params)
                fitness = fitness_fn(population)
                state, _metrics = es.tell(key_tell, population, fitness, state, params)

                # Track best for this run (fitness is raw SSR: lower is better)
                if state.best_fitness < run_best_fitness:
                    run_best_fitness = state.best_fitness
                    run_best_solution = state.best_solution

                # Record fitness history
                diagnostics.fitness_history.append(float(run_best_fitness))

                # Check for stagnation.
                fitness_spread = _finite_fitness_spread(fitness)
                if restarter.check_stagnation(fitness_spread):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                # Trigger restart after sustained stagnation (5 consecutive)
                if stagnation_counter >= 5:
                    logger.info(
                        f"BIPOP run #{restarter.restart_count + 1}: "
                        f"stagnation at gen {gen + 1}, fitness_spread={fitness_spread:.2e}",
                    )
                    break

                # Also check std-based convergence
                if float(state.std) < self.config.tol_x:
                    logger.info(
                        f"BIPOP run #{restarter.restart_count + 1}: "
                        f"converged at gen {gen + 1}, std={float(state.std):.2e}",
                    )
                    break

                # Log progress
                if logger.isEnabledFor(logging.DEBUG) and (gen + 1) % 10 == 0:
                    logger.debug(
                        f"BIPOP Run {restarter.restart_count + 1}: "
                        f"gen {gen + 1}/{self.config.max_generations}, "
                        f"best_fitness={float(run_best_fitness):.6e}, "
                        f"std={float(state.std):.6e}",
                    )

            total_generations += gen + 1
            final_sigma = float(state.std)

            logger.info(
                f"BIPOP run #{restarter.restart_count + 1} completed: "
                f"{gen + 1} generations, best_fitness={float(run_best_fitness):.6e}",
            )

            # Record restart info
            diagnostics.restart_history.append(
                {
                    "popsize": popsize,
                    "generations": gen + 1,
                    "best_fitness": float(run_best_fitness),
                    "final_sigma": final_sigma,
                },
            )

            # Update global best
            restarter.update_best(run_best_solution, float(run_best_fitness))

            # Check if this run converged well (no need for more restarts)
            if float(state.std) < self.config.tol_x and stagnation_counter < 5:
                logger.info("BIPOP: Good convergence achieved, stopping restarts early")
                convergence_reason = "xtol"
                break

            # Register restart for next iteration
            restarter.register_restart()

            # Choose the next run's starting point: best-so-far (exploit),
            # the original starting point, or a fresh random point drawn
            # uniformly from the unbounded-space search region (explore) --
            # true BIPOP (Hansen 2009) randomizes large-population restarts
            # so a bad first basin doesn't anchor every subsequent run.
            #
            # `key` is split unconditionally into three keys (key, choice_key,
            # explore_key) on every iteration regardless of which branch
            # fires below -- consuming a *variable* number of splits per
            # branch would make the RNG stream for every later restart
            # depend on which branch upstream floating-point comparisons
            # (restarter.best_solution) happened to take, so two runs that
            # should be bitwise-identical (e.g. same seed, different
            # population_batch_size) could silently diverge after the first
            # restart whose branch choice differs between them. Only the
            # *split* needs to be unconditional for this guarantee -- the
            # actual (n_params,) array draw from explore_key is a pure,
            # side-effect-free function of that key, so it's only computed
            # in the branch that uses it.
            key, choice_key, explore_key = jax.random.split(key, 3)
            choice = jax.random.uniform(choice_key)
            if choice < 1.0 / 3.0 and restarter.best_solution is not None:
                initial_solution = restarter.best_solution
            elif choice < 2.0 / 3.0:
                initial_solution = jax.random.uniform(
                    explore_key,
                    shape=(n_params,),
                    minval=-_BIPOP_EXPLORE_RANGE,
                    maxval=_BIPOP_EXPLORE_RANGE,
                )
            else:
                initial_solution = original_solution

        # Get global best
        best_solution, best_fitness = restarter.get_best()
        if best_solution is None:
            best_solution = initial_solution
            best_fitness = float("inf")

        logger.info(
            f"BIPOP completed: {restarter.restart_count} restarts, "
            f"{total_generations} total generations",
        )

        # Update diagnostics
        diagnostics.total_restarts = restarter.restart_count
        diagnostics.final_sigma = final_sigma
        diagnostics.convergence_reason = convergence_reason

        return best_solution, jnp.array(best_fitness), total_generations

    def _nlsq_refinement(
        self,
        f: Callable,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: jax.Array,
        bounds: tuple[ArrayLike, ArrayLike],
        sigma: ArrayLike | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run NLSQ Trust Region Reflective refinement.

        This phase provides proper parameter covariance estimation via Jacobian.

        Parameters
        ----------
        f : Callable
            Model function.
        xdata : ArrayLike
            Independent variable data.
        ydata : ArrayLike
            Dependent variable data.
        p0 : jax.Array
            Initial parameters from CMA-ES.
        bounds : tuple[ArrayLike, ArrayLike]
            Parameter bounds.
        sigma : ArrayLike | None
            Standard deviation for weighted residuals.
        **kwargs : Any
            Additional arguments for curve_fit.

        Returns
        -------
        dict[str, Any]
            Result dictionary with popt, pcov, and additional fields.
        """
        from nlsq.core.minpack import curve_fit

        # Convert p0 to numpy for NLSQ
        p0_numpy = np.asarray(p0)

        logger.info(
            f"Starting NLSQ Trust Region Reflective refinement "
            f"(n_params={len(p0_numpy)})",
        )
        logger.debug(f"NLSQ refinement starting from: {p0_numpy}")

        # Convert to numpy arrays for NLSQ compatibility
        xdata_np = np.asarray(xdata)
        ydata_np = np.asarray(ydata)
        sigma_np = np.asarray(sigma) if sigma is not None else None

        try:
            # Run NLSQ curve_fit for refinement with memory-aware workflow
            # Use workflow='auto' to auto-select memory strategy (standard/chunked/streaming)
            # This prevents OOM on large datasets that were handled with data_chunk_size
            # during the CMA-ES evolutionary phase
            refinement_kwargs = {**kwargs}
            refinement_kwargs.pop(
                "workflow",
                None,
            )  # Remove if present to avoid conflict

            n_points = len(ydata_np)
            logger.debug(
                f"NLSQ refinement using workflow='auto' for {n_points:,} points",
            )

            result = curve_fit(
                f,
                xdata_np,
                ydata_np,
                p0=p0_numpy,
                sigma=sigma_np,
                bounds=bounds,
                workflow="auto",  # Memory-aware: auto-selects standard/chunked/streaming
                **refinement_kwargs,
            )

            # CurveFitResult has .x for parameters, .pcov for covariance
            popt = np.asarray(result.x)  # type: ignore[union-attr]
            pcov = np.asarray(result.pcov)  # type: ignore[union-attr]

            # Compute parameter change from CMA-ES to NLSQ
            param_change = np.linalg.norm(popt - p0_numpy)
            logger.info(
                f"NLSQ refinement completed: "
                f"parameter adjustment norm={param_change:.6e}",
            )
            logger.debug(f"NLSQ refined popt={popt}")

            return {
                "popt": popt,
                "pcov": pcov,
                "nlsq_result": result,  # Include full result for diagnostics
            }

        except NotImplementedError:
            # A caller explicitly asked for something curve_fit() cannot
            # honor (e.g. sigma on a chunked/streaming refinement fit) --
            # silently falling back to the unrefined CMA-ES result here
            # would defeat that same "fail loudly instead of silently
            # dropping it" contract one level up. Let it propagate.
            raise
        except Exception as e:
            logger.warning(f"NLSQ refinement failed: {e}. Using CMA-ES result.")
            # Return CMA-ES result if refinement fails
            return {
                "popt": p0_numpy,
                "pcov": self._estimate_pcov_from_cmaes(len(p0_numpy)),
            }

    def _estimate_pcov_from_cmaes(self, n_params: int) -> NDArray[np.floating]:
        """Estimate parameter covariance when NLSQ refinement is disabled.

        This is a rough estimate; for proper pcov, use refine_with_nlsq=True.

        Parameters
        ----------
        n_params : int
            Number of parameters.

        Returns
        -------
        NDArray[np.floating]
            Estimated covariance matrix (diagonal approximation).
        """
        # Return diagonal matrix with inf variance to indicate unknown covariance
        # Proper pcov requires Jacobian from NLSQ
        # Use np.full to avoid RuntimeWarning from 0.0 * np.inf = nan in np.eye() * np.inf
        pcov = np.full((n_params, n_params), 0.0)
        np.fill_diagonal(pcov, np.inf)
        return pcov
