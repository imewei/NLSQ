"""Type aliases and protocols for NLSQ.

This module provides type hints for the NLSQ public API to improve IDE support,
documentation, and static type checking with mypy.

Note: These types are primarily for documentation and tooling. Python's duck typing
means functions will work with any compatible objects at runtime.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

import jax.numpy as jnp
import numpy as np

# Array-like types
# Array-like type that can be converted to numpy/JAX arrays.
type ArrayLike = np.ndarray | jnp.ndarray | list | tuple

# NumPy array of floating point numbers.
type FloatArray = np.ndarray

# JAX array for GPU/TPU-accelerated computations.
type JAXArray = jnp.ndarray

# Function types
# Model function f(x, *params) -> y_pred.
#
# The model function takes independent variable(s) x and fit parameters,
# returning predicted dependent variable values.
#
# Examples:
#     - Linear: f(x, a, b) = a*x + b
#     - Exponential: f(x, a, b) = a * exp(-b * x)
#     - Multi-parameter: f(x, p1, p2, ..., pN) = ...
type ModelFunction = Callable[..., ArrayLike]

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
type JacobianFunction = Callable[..., ArrayLike]

# Callback function for monitoring optimization progress.
#
# Parameters:
#     params: Current parameter estimates
#     residuals: Current residual values
#
# Returns:
#     True to stop optimization, False/None to continue
type CallbackFunction = Callable[[FloatArray, FloatArray], bool | None]

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
type LossFunction = Callable[[FloatArray], FloatArray]

# Bounds types
# Parameter bounds as (lower, upper) tuple.
#
# Examples:
#     - Unbounded: (-np.inf, np.inf)
#     - Lower only: (0, np.inf) for positive parameters
#     - Both: ([-1, 0], [1, 10]) for constrained parameters
type BoundsTuple = tuple[ArrayLike, ArrayLike]

# Result types (using dict for backward compatibility)
# Optimization result dictionary with parameters and diagnostics.
#
# Common fields:
#     - x: Optimized parameters
#     - success: Whether optimization converged
#     - message: Optimization status message
#     - fun: Final residual values (optional)
#     - jac: Final Jacobian (optional)
#     - cost: Final cost function value
#     - optimality: Final gradient norm
#     - nfev: Number of function evaluations
#     - njev: Number of Jacobian evaluations (optional)
type OptimizeResultDict = dict[str, Any]

# Configuration types
# Optimization method name.
#
# Options:
#     - "trf": Trust Region Reflective (default, supports bounds)
#     - "dogbox": Dogleg algorithm for box-constrained problems
#     - "lm": Levenberg-Marquardt (unconstrained only, faster)
type MethodLiteral = str  # "trf" | "dogbox" | "lm"

# Linear solver for trust region subproblems.
#
# Options:
#     - "exact": Direct solver using SVD (default, more accurate)
#     - "lsmr": Iterative solver (faster for large problems)
type SolverLiteral = str  # "exact" | "lsmr"


# Streaming diagnostic types (Task 6.2)
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


# Protocols for structural typing
class HasShape(Protocol):
    """Protocol for objects with a shape attribute."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        ...


class SupportsFloat(Protocol):
    """Protocol for objects that can be converted to float."""

    def __float__(self) -> float:
        """Convert to float."""
        ...


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
    # Protocols
    "HasShape",
    "JAXArray",
    "JacobianFunction",
    "LossFunction",
    # Method/solver literals
    "MethodLiteral",
    # Function types
    "ModelFunction",
    "OptimizeResultDict",
    "SolverLiteral",
    # Streaming diagnostics
    "StreamingDiagnostics",
    "SupportsFloat",
]
