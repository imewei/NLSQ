"""Constants for NLSQ optimization algorithms.

These values are derived from:
- SciPy's trust-region algorithms
- Numerical optimization best practices
- JAX performance characteristics
"""

# =============================================================================
# Trust Region Reflective (TRF) Algorithm Constants
# =============================================================================

# Function Evaluations
DEFAULT_MAX_NFEV_MULTIPLIER = 100  # Max function evaluations per parameter
"""int: Multiplier for maximum function evaluations (max_nfev = n_params * MULTIPLIER)"""

# Trust Region Step Quality
STEP_ACCEPTANCE_THRESHOLD = 0.5  # Trust region step acceptance ratio (ρ > 0.5)
"""float: Minimum ratio for accepting a trust region step"""

STEP_QUALITY_EXCELLENT = 0.75  # Excellent step quality threshold (ρ > 0.75)
"""float: Threshold for excellent step quality (increase trust radius)"""

STEP_QUALITY_GOOD = 0.25  # Good step quality threshold (0.25 < ρ < 0.75)
"""float: Threshold for acceptable step quality (keep trust radius)"""

# Trust Region Radius
INITIAL_TRUST_RADIUS = 1.0  # Initial trust region radius
"""float: Initial value for trust region radius"""

MAX_TRUST_RADIUS = 1000.0  # Maximum trust region radius
"""float: Upper bound for trust region radius"""

MIN_TRUST_RADIUS = 1e-10  # Minimum trust region radius
"""float: Lower bound for trust region radius (termination criterion)"""

# Levenberg-Marquardt Damping
INITIAL_LEVENBERG_MARQUARDT_LAMBDA = 0.0  # Initial LM damping parameter (α)
"""float: Initial damping parameter for Levenberg-Marquardt algorithm"""

# =============================================================================
# Convergence Tolerances
# =============================================================================

DEFAULT_FTOL = 1e-8  # Function tolerance
"""float: Default tolerance for function value convergence"""

DEFAULT_XTOL = 1e-8  # Parameter tolerance
"""float: Default tolerance for parameter convergence"""

DEFAULT_GTOL = 1e-8  # Gradient tolerance
"""float: Default tolerance for gradient convergence"""

# =============================================================================
# Algorithm Selection Thresholds
# =============================================================================

SMALL_DATASET_THRESHOLD = 1000  # Switch to different algorithms for small datasets
"""int: Number of points below which to use specialized small dataset methods"""

LARGE_DATASET_THRESHOLD = 1_000_000  # Use chunking/streaming for very large datasets
"""int: Number of points above which to use chunking or streaming"""

XLARGE_DATASET_THRESHOLD = 10_000_000  # Extremely large - requires streaming
"""int: Number of points requiring streaming optimization"""

# =============================================================================
# Numerical Stability
# =============================================================================

MIN_POSITIVE_VALUE = 1e-15  # Minimum positive value for numerical stability
"""float: Small positive value to prevent division by zero"""

MAX_CONDITION_NUMBER = 1e12  # Maximum matrix condition number
"""float: Threshold for detecting ill-conditioned matrices"""

JACOBIAN_SPARSITY_THRESHOLD = 0.1  # Threshold for sparse Jacobian (10% non-zero)
"""float: Fraction of non-zero elements below which to use sparse methods"""

# =============================================================================
# Memory Management
# =============================================================================

DEFAULT_MEMORY_LIMIT_GB = 4.0  # Default memory limit in gigabytes
"""float: Default memory limit for optimization (GB)"""

MEMORY_SAFETY_FACTOR = 0.9  # Use 90% of available memory
"""float: Safety factor for memory allocation (leave 10% buffer)"""

# =============================================================================
# Trust Region Step Scaling
# =============================================================================

TRUST_RADIUS_INCREASE_FACTOR = 2.0  # Factor to increase trust radius on success
"""float: Multiplier for trust radius after excellent step"""

TRUST_RADIUS_DECREASE_FACTOR = 0.5  # Factor to decrease trust radius on failure
"""float: Multiplier for trust radius after poor step"""

# =============================================================================
# Termination Status Codes
# =============================================================================

TERMINATION_GTOL = 1  # Gradient tolerance satisfied
"""int: Termination due to gradient tolerance"""

TERMINATION_FTOL = 2  # Function tolerance satisfied
"""int: Termination due to function value change tolerance"""

TERMINATION_XTOL = 3  # Parameter tolerance satisfied
"""int: Termination due to parameter change tolerance"""

TERMINATION_MAX_NFEV = 0  # Maximum function evaluations reached
"""int: Termination due to reaching max function evaluations"""

# =============================================================================
# Loss Function Constants
# =============================================================================

DEFAULT_F_SCALE = 1.0  # Default scale for loss function
"""float: Default scale parameter for robust loss functions"""

HUBER_LOSS_THRESHOLD = 1.0  # Threshold for Huber loss
"""float: Threshold parameter for Huber robust loss"""

# =============================================================================
# Finite Difference Parameters
# =============================================================================

FINITE_DIFF_REL_STEP = 1e-8  # Relative step for finite differences
"""float: Relative step size for finite difference Jacobian approximation"""

FINITE_DIFF_ABS_STEP_MIN = 1e-12  # Minimum absolute step
"""float: Minimum absolute step size for finite differences"""

# =============================================================================
# Validation Constants
# =============================================================================

MIN_DATA_POINTS = 1  # Minimum number of data points
"""int: Minimum number of data points required for fitting"""

MIN_PARAMETERS = 1  # Minimum number of parameters
"""int: Minimum number of parameters to fit"""

MAX_REASONABLE_PARAMETERS = 1000  # Warning threshold for parameter count
"""int: Number of parameters above which to warn (may be inefficient)"""

# =============================================================================
# Exported Constants (for backwards compatibility)
# =============================================================================

__all__ = [
    # TRF Algorithm
    "DEFAULT_MAX_NFEV_MULTIPLIER",
    "STEP_ACCEPTANCE_THRESHOLD",
    "STEP_QUALITY_EXCELLENT",
    "STEP_QUALITY_GOOD",
    "INITIAL_TRUST_RADIUS",
    "MAX_TRUST_RADIUS",
    "MIN_TRUST_RADIUS",
    "INITIAL_LEVENBERG_MARQUARDT_LAMBDA",
    "TRUST_RADIUS_INCREASE_FACTOR",
    "TRUST_RADIUS_DECREASE_FACTOR",
    # Tolerances
    "DEFAULT_FTOL",
    "DEFAULT_XTOL",
    "DEFAULT_GTOL",
    # Thresholds
    "SMALL_DATASET_THRESHOLD",
    "LARGE_DATASET_THRESHOLD",
    "XLARGE_DATASET_THRESHOLD",
    "JACOBIAN_SPARSITY_THRESHOLD",
    # Numerical
    "MIN_POSITIVE_VALUE",
    "MAX_CONDITION_NUMBER",
    # Memory
    "DEFAULT_MEMORY_LIMIT_GB",
    "MEMORY_SAFETY_FACTOR",
    # Termination
    "TERMINATION_GTOL",
    "TERMINATION_FTOL",
    "TERMINATION_XTOL",
    "TERMINATION_MAX_NFEV",
    # Loss
    "DEFAULT_F_SCALE",
    "HUBER_LOSS_THRESHOLD",
    # Finite Differences
    "FINITE_DIFF_REL_STEP",
    "FINITE_DIFF_ABS_STEP_MIN",
    # Validation
    "MIN_DATA_POINTS",
    "MIN_PARAMETERS",
    "MAX_REASONABLE_PARAMETERS",
]
