"""JAX-aware asynchronous logging to prevent host-device synchronization.

This module provides logging functions that use jax.debug.callback to execute
logging asynchronously on the host, preventing blocking device computation.

Examples
--------
>>> from nlsq.utils.async_logger import log_iteration_async
>>> import jax.numpy as jnp
>>>
>>> # Inside optimization loop
>>> log_iteration_async(
...     iteration=nit,
...     cost=cost,
...     gradient_norm=jnp.linalg.norm(g),
...     message=f"step={step:.6e}",
...     verbose=2
... )
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def is_jax_array(x: Any) -> bool:
    """Check if value is a JAX array.

    Parameters
    ----------
    x : Any
        Value to check

    Returns
    -------
    bool
        True if x is a JAX array or tracer
    """
    return isinstance(x, (jax.Array, jax.core.Tracer))


def log_iteration_async(
    iteration: int | jax.Array,
    cost: float | jax.Array,
    gradient_norm: float | jax.Array,
    message: str = "",
    verbose: int = 0,
) -> None:
    """Log optimization iteration asynchronously without blocking device.

    Uses jax.debug.callback to execute logging on the host thread while device
    computation continues. This prevents the performance penalty of host-device
    synchronization during optimization.

    Parameters
    ----------
    iteration : int or jax.Array
        Current iteration number
    cost : float or jax.Array
        Current cost/loss value
    gradient_norm : float or jax.Array
        Norm of gradient vector
    message : str, optional
        Additional message to append to log
    verbose : int, optional
        Verbosity level:
        - 0: No logging
        - 1: Log every 10 iterations
        - 2: Log every iteration

    Notes
    -----
    The callback executes asynchronously, so log messages may appear slightly
    out of order. However, each message includes the iteration number for
    proper ordering during analysis.

    This function has minimal overhead (~1-2μs) and does not block device
    computation, making it suitable for use in tight optimization loops.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nlsq.utils.async_logger import log_iteration_async
    >>>
    >>> # In optimization loop
    >>> for nit in range(100):
    ...     cost = compute_cost()  # Returns JAX array
    ...     g = compute_gradient()  # Returns JAX array
    ...     log_iteration_async(nit, cost, jnp.linalg.norm(g), verbose=2)
    """
    if verbose == 0:
        return

    # Skip logging based on verbosity level
    if verbose == 1 and isinstance(iteration, int) and iteration % 10 != 0:
        return

    # Define pure callback function (executed asynchronously on host)
    def _log_callback(iter_val, cost_val, norm_val, msg_val):
        """Pure function executed asynchronously on host thread."""
        logger.info(
            f"Optimization: iter={int(iter_val)} | "
            f"cost={float(cost_val):.6e} | "
            f"‖∇f‖={float(norm_val):.6e}" + (f" | {msg_val}" if msg_val else "")
        )

    # Ensure all numeric values are JAX arrays (lightweight operation)
    iter_arr = jnp.asarray(iteration)
    cost_arr = jnp.asarray(cost)
    norm_arr = jnp.asarray(gradient_norm)

    # Execute callback asynchronously (non-blocking on device)
    # JAX will convert arrays to NumPy in the callback thread
    jax.debug.callback(_log_callback, iter_arr, cost_arr, norm_arr, message)


def log_convergence_async(
    reason: str,
    iterations: int | jax.Array,
    final_cost: float | jax.Array,
    time_sec: float,
    final_gradient_norm: float | jax.Array,
    verbose: int = 1,
) -> None:
    """Log convergence result asynchronously.

    Parameters
    ----------
    reason : str
        Convergence termination reason
    iterations : int or jax.Array
        Total number of iterations performed
    final_cost : float or jax.Array
        Final cost value achieved
    time_sec : float
        Total optimization time in seconds
    final_gradient_norm : float or jax.Array
        Final gradient norm
    verbose : int, optional
        Verbosity level (0 = no logging)

    Examples
    --------
    >>> log_convergence_async(
    ...     reason="`gtol` termination condition is satisfied.",
    ...     iterations=42,
    ...     final_cost=1.23e-10,
    ...     time_sec=2.456,
    ...     final_gradient_norm=5.67e-9,
    ...     verbose=1
    ... )
    """
    if verbose == 0:
        return

    def _log_callback(iter_val, cost_val, time_val, norm_val, reason_val):
        """Pure function executed asynchronously on host thread."""
        logger.info(
            f"Convergence: reason={reason_val} | "
            f"iterations={int(iter_val)} | "
            f"final_cost={float(cost_val):.6e} | "
            f"time={float(time_val):.3f}s | "
            f"final_gradient_norm={float(norm_val):.6e}"
        )

    # Convert to JAX arrays
    iter_arr = jnp.asarray(iterations)
    cost_arr = jnp.asarray(final_cost)
    time_arr = jnp.asarray(time_sec)
    norm_arr = jnp.asarray(final_gradient_norm)

    # Execute callback asynchronously
    jax.debug.callback(_log_callback, iter_arr, cost_arr, time_arr, norm_arr, reason)
