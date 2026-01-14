"""CMA-ES global optimizer with NLSQ refinement.

This module provides the CMAESOptimizer class that runs CMA-ES global search
using evosax followed by NLSQ Trust Region Reflective refinement for proper
parameter covariance estimation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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

__all__ = ["CMAESOptimizer"]

logger = logging.getLogger(__name__)


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

    CMA-ES maximizes fitness, so we return negative SSR (sum of squared residuals).

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
                params_unbounded, lower_bounds, upper_bounds
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
                    params_bounded, x_chunk, y_chunk, sigma_chunk, valid_mask
                )

            return jnp.where(jnp.isfinite(ssr_total), -ssr_total, -jnp.inf)

        fitness_single = fitness_single_streaming

        logger.debug(
            f"Data streaming enabled: {n_data} points -> {n_chunks} chunks of {data_chunk_size}"
        )
    else:
        # Original non-streaming fitness function
        @jax.jit
        def fitness_single(params_unbounded: jax.Array) -> jax.Array:
            """Compute fitness for a single parameter set."""
            # Transform to bounded space
            params_bounded = transform_to_bounds(
                params_unbounded, lower_bounds, upper_bounds
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
            fitness = jnp.where(jnp.isfinite(ssr), -ssr, -jnp.inf)

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
                "Install with: pip install 'nlsq[global]'"
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
                "Provide bounds as (lower_bounds, upper_bounds)."
            )

        # Convert inputs to JAX arrays
        xdata_jax = jnp.asarray(xdata)
        ydata_jax = jnp.asarray(ydata)
        lower_bounds = jnp.asarray(bounds[0])
        upper_bounds = jnp.asarray(bounds[1])
        sigma_jax = jnp.asarray(sigma) if sigma is not None else None

        n_params = len(lower_bounds)
        n_data = len(ydata_jax)

        # Log initialization
        logger.info(
            f"CMA-ES optimizer initialized: n_params={n_params}, n_data={n_data}, "
            f"restart_strategy={self.config.restart_strategy}"
        )
        logger.debug(
            f"CMA-ES bounds: lower={np.asarray(lower_bounds)}, "
            f"upper={np.asarray(upper_bounds)}"
        )

        # Determine population size
        popsize = self.config.popsize
        if popsize is None:
            popsize = compute_default_popsize(n_params)

        # Double population for cmaes-global preset
        # (detected by max_generations == 200 and bipop)
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
                f"{self.config.population_batch_size}"
            )
        if self.config.data_chunk_size is not None:
            logger.info(
                f"CMA-ES memory optimization: data_chunk_size="
                f"{self.config.data_chunk_size} (data streaming enabled)"
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

        # Track wall time
        import time

        start_time = time.perf_counter()

        # Initialize diagnostics
        diagnostics = CMAESDiagnostics()

        # Run CMA-ES optimization (diagnostics updated in place)
        best_params_unbounded, best_fitness, generations = self._run_cmaes(
            fitness_fn, initial_solution, popsize, n_params, diagnostics
        )

        # Update diagnostics
        diagnostics.total_generations = generations
        diagnostics.best_fitness = float(best_fitness)
        diagnostics.wall_time = time.perf_counter() - start_time

        # Transform best solution back to bounded space
        best_params = transform_to_bounds(
            best_params_unbounded, lower_bounds, upper_bounds
        )

        logger.info(
            f"CMA-ES optimization completed: {generations} generations, "
            f"best_fitness={float(best_fitness):.6e}, "
            f"wall_time={diagnostics.wall_time:.2f}s"
        )

        # NLSQ refinement phase for proper pcov estimation
        if self.config.refine_with_nlsq:
            result = self._nlsq_refinement(
                f, xdata, ydata, best_params, bounds, sigma, **kwargs
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

    def _run_cmaes(
        self,
        fitness_fn: Callable,
        initial_solution: jax.Array,
        popsize: int,
        n_params: int,
        diagnostics: CMAESDiagnostics,
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
                fitness_fn, initial_solution, popsize, n_params, diagnostics
            )
        else:
            return self._run_cmaes_single(
                fitness_fn, initial_solution, popsize, n_params, diagnostics
            )

    def _run_cmaes_single(
        self,
        fitness_fn: Callable,
        initial_solution: jax.Array,
        popsize: int,
        n_params: int,
        diagnostics: CMAESDiagnostics,
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

        logger.info(
            f"Starting CMA-ES: popsize={popsize}, max_gen={self.config.max_generations}"
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

        # Track best solution
        best_solution = initial_solution
        best_fitness = jnp.array(-jnp.inf)
        convergence_reason = "max_generations"

        # Progress milestones for logging (25%, 50%, 75%)
        milestones = {
            int(self.config.max_generations * 0.25): "25%",
            int(self.config.max_generations * 0.50): "50%",
            int(self.config.max_generations * 0.75): "75%",
        }

        # Main optimization loop
        for gen in range(self.config.max_generations):
            key, key_ask, key_tell = jax.random.split(key, 3)

            # Ask for new population
            population, state = es.ask(key_ask, state, params)

            # Evaluate fitness
            fitness = fitness_fn(population)

            # Update CMA-ES state
            state, _metrics = es.tell(key_tell, population, fitness, state, params)

            # Track best (CMA-ES maximizes, so higher is better)
            if state.best_fitness > best_fitness:
                best_fitness = state.best_fitness
                best_solution = state.best_solution

            # Record fitness history
            diagnostics.fitness_history.append(float(best_fitness))

            # Simple convergence check based on std
            if float(state.std) < self.config.tol_x:
                logger.info(
                    f"CMA-ES converged at generation {gen + 1}: "
                    f"std={float(state.std):.2e} < tol_x={self.config.tol_x:.2e}"
                )
                convergence_reason = "xtol"
                break

            # Log progress at milestones (INFO level)
            if gen + 1 in milestones:
                logger.info(
                    f"CMA-ES progress {milestones[gen + 1]}: "
                    f"gen={gen + 1}/{self.config.max_generations}, "
                    f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.2e}"
                )

            # Log detailed progress at debug level
            if logger.isEnabledFor(logging.DEBUG) and (gen + 1) % 10 == 0:
                logger.debug(
                    f"Generation {gen + 1}/{self.config.max_generations}: "
                    f"best_fitness={float(best_fitness):.6e}, std={float(state.std):.6e}"
                )

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
            f"max_restarts={self.config.max_restarts}, max_gen={self.config.max_generations}"
        )

        # Initialize BIPOP restarter
        restarter = BIPOPRestarter(
            base_popsize=base_popsize,
            n_params=n_params,
            max_restarts=self.config.max_restarts,
            min_fitness_spread=self.config.tol_fun,
        )

        # Initialize random key
        if self.config.seed is not None:
            key = jax.random.key(self.config.seed)
        else:
            key = jax.random.key(np.random.randint(0, 2**31))

        total_generations = 0
        convergence_reason = "max_restarts"
        final_sigma = self.config.sigma

        while not restarter.exhausted:
            # Get population size for this run
            popsize = restarter.get_next_popsize()
            run_type = "large" if popsize >= base_popsize * 2 else "small"

            logger.info(
                f"BIPOP restart #{restarter.restart_count + 1}: "
                f"popsize={popsize} ({run_type}), "
                f"max_gen={self.config.max_generations}"
            )

            # Initialize CMA-ES for this run
            es = CMA_ES(population_size=popsize, solution=initial_solution)
            params = es.default_params
            params = params.replace(std_init=self.config.sigma)

            key, subkey = jax.random.split(key)
            state = es.init(subkey, initial_solution, params)

            # Track best for this run
            run_best_solution = initial_solution
            run_best_fitness = jnp.array(-jnp.inf)

            # Run optimization loop
            stagnation_counter = 0
            for gen in range(self.config.max_generations):
                key, key_ask, key_tell = jax.random.split(key, 3)

                population, state = es.ask(key_ask, state, params)
                fitness = fitness_fn(population)
                state, _metrics = es.tell(key_tell, population, fitness, state, params)

                # Track best for this run
                if state.best_fitness > run_best_fitness:
                    run_best_fitness = state.best_fitness
                    run_best_solution = state.best_solution

                # Record fitness history
                diagnostics.fitness_history.append(float(run_best_fitness))

                # Check for stagnation
                fitness_spread = float(jnp.max(fitness) - jnp.min(fitness))
                if restarter.check_stagnation(fitness_spread):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                # Trigger restart after sustained stagnation (5 consecutive)
                if stagnation_counter >= 5:
                    logger.info(
                        f"BIPOP run #{restarter.restart_count + 1}: "
                        f"stagnation at gen {gen + 1}, fitness_spread={fitness_spread:.2e}"
                    )
                    break

                # Also check std-based convergence
                if float(state.std) < self.config.tol_x:
                    logger.info(
                        f"BIPOP run #{restarter.restart_count + 1}: "
                        f"converged at gen {gen + 1}, std={float(state.std):.2e}"
                    )
                    break

                # Log progress
                if logger.isEnabledFor(logging.DEBUG) and (gen + 1) % 10 == 0:
                    logger.debug(
                        f"BIPOP Run {restarter.restart_count + 1}: "
                        f"gen {gen + 1}/{self.config.max_generations}, "
                        f"best_fitness={float(run_best_fitness):.6e}, "
                        f"std={float(state.std):.6e}"
                    )

            total_generations += gen + 1
            final_sigma = float(state.std)

            logger.info(
                f"BIPOP run #{restarter.restart_count + 1} completed: "
                f"{gen + 1} generations, best_fitness={float(run_best_fitness):.6e}"
            )

            # Record restart info
            diagnostics.restart_history.append(
                {
                    "popsize": popsize,
                    "generations": gen + 1,
                    "best_fitness": float(run_best_fitness),
                    "final_sigma": final_sigma,
                }
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

            # Use current best as starting point for next run
            # (adds exploitation around best solution)
            if restarter.best_solution is not None:
                # 50% chance to restart from best, 50% from origin
                key, subkey = jax.random.split(key)
                if jax.random.uniform(subkey) > 0.5:
                    initial_solution = restarter.best_solution

        # Get global best
        best_solution, best_fitness = restarter.get_best()
        if best_solution is None:
            best_solution = initial_solution
            best_fitness = -jnp.inf

        logger.info(
            f"BIPOP completed: {restarter.restart_count} restarts, "
            f"{total_generations} total generations"
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
            f"(n_params={len(p0_numpy)})"
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
                "workflow", None
            )  # Remove if present to avoid conflict

            n_points = len(ydata_np)
            logger.debug(
                f"NLSQ refinement using workflow='auto' for {n_points:,} points"
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
                f"parameter adjustment norm={param_change:.6e}"
            )
            logger.debug(f"NLSQ refined popt={popt}")

            return {
                "popt": popt,
                "pcov": pcov,
                "nlsq_result": result,  # Include full result for diagnostics
            }

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
        # Return identity matrix as rough estimate
        # Proper pcov requires Jacobian from NLSQ
        return np.eye(n_params) * np.inf
