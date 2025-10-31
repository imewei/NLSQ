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

# Trust Region Step Quality
STEP_ACCEPTANCE_THRESHOLD = 0.5  # Trust region step acceptance ratio (rho > 0.5)

STEP_QUALITY_EXCELLENT = 0.75  # Excellent step quality threshold (rho > 0.75)

STEP_QUALITY_GOOD = 0.25  # Good step quality threshold (0.25 < rho < 0.75)

# Trust Region Radius
INITIAL_TRUST_RADIUS = 1.0  # Initial trust region radius

MAX_TRUST_RADIUS = 1000.0  # Maximum trust region radius

MIN_TRUST_RADIUS = 1e-10  # Minimum trust region radius

# Levenberg-Marquardt Damping
INITIAL_LEVENBERG_MARQUARDT_LAMBDA = 0.0  # Initial LM damping parameter (alpha)

# =============================================================================
# Convergence Tolerances
# =============================================================================

DEFAULT_FTOL = 1e-8  # Function tolerance

DEFAULT_XTOL = 1e-8  # Parameter tolerance

DEFAULT_GTOL = 1e-8  # Gradient tolerance

# =============================================================================
# Algorithm Selection Thresholds
# =============================================================================

SMALL_DATASET_THRESHOLD = 1000  # Switch to different algorithms for small datasets

LARGE_DATASET_THRESHOLD = 1_000_000  # Use chunking/streaming for very large datasets

XLARGE_DATASET_THRESHOLD = 10_000_000  # Extremely large - requires streaming

# =============================================================================
# Numerical Stability
# =============================================================================

MIN_POSITIVE_VALUE = 1e-15  # Minimum positive value for numerical stability

MAX_CONDITION_NUMBER = 1e12  # Maximum matrix condition number

JACOBIAN_SPARSITY_THRESHOLD = 0.1  # Threshold for sparse Jacobian (10% non-zero)

# =============================================================================
# Memory Management
# =============================================================================

DEFAULT_MEMORY_LIMIT_GB = 4.0  # Default memory limit in gigabytes

MEMORY_SAFETY_FACTOR = 0.9  # Use 90% of available memory

# =============================================================================
# Trust Region Step Scaling
# =============================================================================

TRUST_RADIUS_INCREASE_FACTOR = 2.0  # Factor to increase trust radius on success

TRUST_RADIUS_DECREASE_FACTOR = 0.5  # Factor to decrease trust radius on failure

# =============================================================================
# Termination Status Codes
# =============================================================================

TERMINATION_GTOL = 1  # Gradient tolerance satisfied

TERMINATION_FTOL = 2  # Function tolerance satisfied

TERMINATION_XTOL = 3  # Parameter tolerance satisfied

TERMINATION_MAX_NFEV = 0  # Maximum function evaluations reached

# =============================================================================
# Loss Function Constants
# =============================================================================

DEFAULT_F_SCALE = 1.0  # Default scale for loss function

HUBER_LOSS_THRESHOLD = 1.0  # Threshold for Huber loss

# =============================================================================
# Finite Difference Parameters
# =============================================================================

FINITE_DIFF_REL_STEP = 1e-8  # Relative step for finite differences

FINITE_DIFF_ABS_STEP_MIN = 1e-12  # Minimum absolute step

# =============================================================================
# Validation Constants
# =============================================================================

MIN_DATA_POINTS = 1  # Minimum number of data points

MIN_PARAMETERS = 1  # Minimum number of parameters

MAX_REASONABLE_PARAMETERS = 1000  # Warning threshold for parameter count

# =============================================================================
# Mixed Precision Fallback Constants
# =============================================================================

# Gradient explosion threshold for detecting numerical instability
DEFAULT_GRADIENT_EXPLOSION_THRESHOLD = 1e10

# Precision limit threshold - minimum meaningful parameter change in float32
DEFAULT_PRECISION_LIMIT_THRESHOLD = 1e-7

# Number of iterations to track for convergence stall detection
DEFAULT_STALL_WINDOW = 10

# Number of degradation iterations before upgrading from float32 to float64
DEFAULT_MAX_DEGRADATION_ITERATIONS = 5

# Factor to relax tolerances when float64 fails (fallback to float32)
DEFAULT_TOLERANCE_RELAXATION_FACTOR = 10.0

# =============================================================================
# Exported Constants (for backwards compatibility)
# =============================================================================

__all__ = [
    "DEFAULT_FTOL",
    "DEFAULT_F_SCALE",
    "DEFAULT_GRADIENT_EXPLOSION_THRESHOLD",
    "DEFAULT_GTOL",
    "DEFAULT_MAX_DEGRADATION_ITERATIONS",
    "DEFAULT_MAX_NFEV_MULTIPLIER",
    "DEFAULT_MEMORY_LIMIT_GB",
    "DEFAULT_PRECISION_LIMIT_THRESHOLD",
    "DEFAULT_STALL_WINDOW",
    "DEFAULT_TOLERANCE_RELAXATION_FACTOR",
    "DEFAULT_XTOL",
    "FINITE_DIFF_ABS_STEP_MIN",
    "FINITE_DIFF_REL_STEP",
    "HUBER_LOSS_THRESHOLD",
    "INITIAL_LEVENBERG_MARQUARDT_LAMBDA",
    "INITIAL_TRUST_RADIUS",
    "JACOBIAN_SPARSITY_THRESHOLD",
    "LARGE_DATASET_THRESHOLD",
    "MAX_CONDITION_NUMBER",
    "MAX_REASONABLE_PARAMETERS",
    "MAX_TRUST_RADIUS",
    "MEMORY_SAFETY_FACTOR",
    "MIN_DATA_POINTS",
    "MIN_PARAMETERS",
    "MIN_POSITIVE_VALUE",
    "MIN_TRUST_RADIUS",
    "SMALL_DATASET_THRESHOLD",
    "STEP_ACCEPTANCE_THRESHOLD",
    "STEP_QUALITY_EXCELLENT",
    "STEP_QUALITY_GOOD",
    "TERMINATION_FTOL",
    "TERMINATION_GTOL",
    "TERMINATION_MAX_NFEV",
    "TERMINATION_XTOL",
    "TRUST_RADIUS_DECREASE_FACTOR",
    "TRUST_RADIUS_INCREASE_FACTOR",
    "XLARGE_DATASET_THRESHOLD",
]
