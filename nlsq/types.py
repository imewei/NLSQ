"""Type aliases and protocols for NLSQ.

This module provides type hints for the NLSQ public API to improve IDE support,
documentation, and static type checking with mypy.

Note: These types are primarily for documentation and tooling. Python's duck typing
means functions will work with any compatible objects at runtime.

Note: JAX imports are deferred using TYPE_CHECKING to avoid import-time errors
during documentation builds or in environments where JAX is not fully configured.
At runtime, JAX array types are represented as Any to allow the module to be
imported without JAX being available.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import numpy as np

# =============================================================================
# Array-like types
# =============================================================================
# At runtime, we use Any for JAX arrays to avoid import-time JAX dependency.
# Static type checkers will see the full type via TYPE_CHECKING block.

if TYPE_CHECKING:
    import jax.numpy as jnp

    # Full type definitions for static analysis
    ArrayLike: TypeAlias = np.ndarray | jnp.ndarray | list | tuple
    JAXArray: TypeAlias = jnp.ndarray
else:
    # Runtime definitions - avoid JAX import
    # Use Any as a stand-in for jax.numpy.ndarray
    ArrayLike: TypeAlias = np.ndarray | Any | list | tuple
    JAXArray: TypeAlias = Any

# NumPy array of floating point numbers.
FloatArray: TypeAlias = np.ndarray

# =============================================================================
# Function types
# =============================================================================

# Model function f(x, *params) -> y_pred.
#
# The model function takes independent variable(s) x and fit parameters,
# returning predicted dependent variable values.
#
# Examples:
#     - Linear: f(x, a, b) = a*x + b
#     - Exponential: f(x, a, b) = a * exp(-b * x)
#     - Multi-parameter: f(x, p1, p2, ..., pN) = ...
ModelFunction: TypeAlias = Callable[..., ArrayLike]

# Jacobian function jac(x, *params) -> J.
#
# The Jacobian function computes the matrix of partial derivatives:
# J[i, j] = ∂f[i]/∂params[j]
#
# Parameters:
#     x: Independent variable(s)
#     *params: Fit parameters
#
# Returns:
#     J: Jacobian matrix of shape (m, n) where m = len(f(x)) and n = len(params)
JacobianFunction: TypeAlias = Callable[..., ArrayLike]

# Callback function for monitoring optimization progress.
#
# Parameters:
#     params: Current parameter estimates
#     residuals: Current residual values
#
# Returns:
#     True to stop optimization, False/None to continue
CallbackFunction: TypeAlias = Callable[[FloatArray, FloatArray], bool | None]

# Loss function rho(z) for robust fitting.
#
# Robust loss functions reduce the influence of outliers by applying
# a non-linear transformation to residuals.
#
# Parameters:
#     z: Squared residuals (z = residuals**2)
#
# Returns:
#     rho: Transformed residuals for robust fitting
LossFunction: TypeAlias = Callable[[FloatArray], FloatArray]

# =============================================================================
# Bounds types
# =============================================================================

# Parameter bounds as (lower, upper) tuple.
#
# Examples:
#     - Unbounded: (-np.inf, np.inf)
#     - Lower only: (0, np.inf) for positive parameters
#     - Both: ([-1, 0], [1, 10]) for constrained parameters
BoundsTuple: TypeAlias = tuple[ArrayLike, ArrayLike]

# =============================================================================
# Configuration types
# =============================================================================

# Optimization method name, as accepted by curve_fit()'s `method` parameter
# (nlsq/core/minpack.py) -- this is the top-level workflow/global-method
# selector, not the classical scipy trust-region sub-method. fit() forwards
# its own `method` argument straight into curve_fit()'s `method`, so the two
# must share this exact Literal (previously `str`, which let an invalid
# value like method="bogus" type-check silently).
#
# Options:
#     - "auto": Automatic selection (default)
#     - "cmaes": CMA-ES global optimization
#     - "multi-start": Multi-start local optimization
#     - "trf": Trust Region Reflective (local, supports bounds)
#     - "hybrid_streaming": Adaptive hybrid streaming optimizer
MethodLiteral: TypeAlias = Literal[
    "auto", "cmaes", "multi-start", "trf", "hybrid_streaming"
]


# =============================================================================
# Streaming diagnostic types (Task 6.2)
# =============================================================================


class CheckpointInfo(TypedDict, total=False):
    """Checkpoint information in streaming diagnostics."""

    path: str | None  # Path to checkpoint file
    saved_at: str  # Timestamp when checkpoint was saved
    batch_idx: int  # Batch index at checkpoint
    iteration: int  # Iteration number at checkpoint
    file_size: int  # Size of checkpoint file in bytes (optional)


class CommonError(TypedDict):
    """Common error entry in diagnostics."""

    type: str  # Error type name
    count: int  # Number of occurrences


class AggregateStats(TypedDict, total=False):
    """Aggregate statistics across batches."""

    mean_loss: float  # Mean batch loss
    std_loss: float  # Standard deviation of batch losses
    mean_grad_norm: float  # Mean gradient norm
    min_loss: float  # Minimum batch loss
    max_loss: float  # Maximum batch loss


class StreamingDiagnostics(TypedDict, total=False):
    """Comprehensive diagnostics for streaming optimization.

    This structure matches the format from chunked processing for consistency.
    """

    failed_batches: list[int]  # List of failed batch indices
    retry_counts: dict[int, int]  # Retry attempts per batch index
    error_types: dict[str, int]  # Count of each error type
    batch_success_rate: float  # Overall success rate (0.0 to 1.0)
    checkpoint_info: CheckpointInfo  # Checkpoint details
    recent_batch_stats: list[dict[str, Any]]  # Circular buffer of recent batch stats
    aggregate_stats: AggregateStats  # Aggregate metrics across all batches
    common_errors: list[CommonError]  # Top 3 most common errors

    # Streaming-specific fields
    total_batches_processed: int  # Total number of batches attempted
    total_retries: int  # Total number of retry attempts
    convergence_achieved: bool  # Whether convergence criteria was met
    final_epoch: int  # Epoch at which optimization ended

    # Timing information
    total_time: float  # Total optimization time in seconds
    mean_batch_time: float  # Average time per batch
    checkpoint_save_time: float  # Total time spent saving checkpoints


# Re-export commonly used types from dependencies
__all__ = [
    "AggregateStats",
    # Array types
    "ArrayLike",
    # Bounds and results
    "BoundsTuple",
    "CallbackFunction",
    "CheckpointInfo",
    "CommonError",
    "FloatArray",
    "JAXArray",
    "JacobianFunction",
    "LossFunction",
    # Method literal
    "MethodLiteral",
    # Function types
    "ModelFunction",
    # Streaming diagnostics
    "StreamingDiagnostics",
]
