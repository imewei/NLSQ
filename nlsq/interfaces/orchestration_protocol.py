"""Protocol definitions for CurveFit orchestration components.

This module defines protocols for the decomposed CurveFit components:
- DataPreprocessorProtocol: Input validation and array conversion
- OptimizationSelectorProtocol: Method selection and configuration
- CovarianceComputerProtocol: Covariance matrix computation
- StreamingCoordinatorProtocol: Streaming strategy selection

These protocols enable dependency injection and facilitate testing
of individual components in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from nlsq.result.optimize_result import OptimizeResult
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.types import ArrayLike


# =============================================================================
# DataPreprocessor Protocol and Entities
# =============================================================================


@dataclass(frozen=True, slots=True)
class PreprocessedData:
    """Result of data preprocessing.

    All arrays are validated and converted to JAX arrays.
    Padding may be applied for JAX compilation efficiency.

    Attributes:
        xdata: Independent variable data, shape (n,) or (k, n)
        ydata: Dependent variable data, shape (n,)
        sigma: Uncertainty/weights, shape (n,), (n, n), or None
        mask: Boolean mask for valid data points, shape (n,)
        n_points: Number of valid data points
        is_padded: Whether arrays were padded for fixed-size compilation
        original_length: Length before padding (equals n_points if not padded)
        has_nans_removed: True if NaN values were filtered during preprocessing
        has_infs_removed: True if Inf values were filtered during preprocessing
    """

    xdata: jax.Array
    ydata: jax.Array
    sigma: jax.Array | None
    mask: jax.Array
    n_points: int
    is_padded: bool
    original_length: int
    has_nans_removed: bool
    has_infs_removed: bool


@runtime_checkable
class DataPreprocessorProtocol(Protocol):
    """Protocol for data preprocessing component.

    Implementations handle:
    1. Input validation (type checking, finiteness)
    2. Array conversion (numpy/list to JAX)
    3. Length consistency checking
    4. Data masking for invalid points
    5. Padding for JAX compilation efficiency
    """

    def preprocess(
        self,
        f: Callable[..., ArrayLike],
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        sigma: ArrayLike | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        nan_policy: str = "raise",
        stability_check: bool = False,
    ) -> PreprocessedData:
        """Validate and preprocess input data for curve fitting.

        Args:
            f: Model function to fit (used for parameter count detection)
            xdata: Independent variable data
            ydata: Dependent variable data (observations)
            sigma: Uncertainty/weights for observations
            absolute_sigma: If True, sigma is absolute; else relative
            check_finite: If True, raise on NaN/Inf values
            nan_policy: How to handle NaN: 'raise', 'omit', or 'propagate'
            stability_check: If True, run additional stability checks

        Returns:
            PreprocessedData with validated, converted arrays

        Raises:
            ValueError: If inputs are invalid (wrong shape, non-finite, etc.)
            TypeError: If inputs have wrong types
        """
        ...

    def validate_sigma(
        self,
        sigma: ArrayLike | None,
        ydata_shape: tuple[int, ...],
    ) -> jax.Array | None:
        """Validate and convert sigma to appropriate format.

        Args:
            sigma: Input sigma (1D for diagonal, 2D for full covariance)
            ydata_shape: Shape of ydata for compatibility check

        Returns:
            Validated JAX array or None

        Raises:
            ValueError: If sigma shape is incompatible with ydata
        """
        ...


# =============================================================================
# OptimizationSelector Protocol and Entities
# =============================================================================


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Configuration for optimization execution.

    Contains all settings needed by LeastSquares optimizer.

    Attributes:
        method: Optimization algorithm ('trf', 'lm', 'dogbox')
        tr_solver: Trust region subproblem solver ('exact', 'lsmr', None)
        n_params: Number of parameters to fit
        p0: Initial parameter guess
        bounds: Lower and upper bounds as (lb, ub) tuple
        max_nfev: Maximum function evaluations
        ftol: Relative tolerance for cost function
        xtol: Relative tolerance for parameters
        gtol: Relative tolerance for gradient
        jac: Jacobian specification ('2-point', '3-point', callable, None)
        x_scale: Parameter scaling ('jac' or array)
    """

    method: Literal["trf", "lm", "dogbox"]
    tr_solver: Literal["exact", "lsmr"] | None
    n_params: int
    p0: jax.Array
    bounds: tuple[jax.Array, jax.Array]
    max_nfev: int
    ftol: float
    xtol: float
    gtol: float
    jac: str | Callable | None
    x_scale: jax.Array | str


@runtime_checkable
class OptimizationSelectorProtocol(Protocol):
    """Protocol for optimization method selection component.

    Implementations handle:
    1. Parameter count detection from function signature
    2. Method selection based on bounds and problem type
    3. Bounds validation and preparation
    4. Initial guess generation if not provided
    5. Solver configuration validation
    """

    def select(
        self,
        f: Callable[..., ArrayLike],
        xdata: jax.Array,
        ydata: jax.Array,
        *,
        p0: ArrayLike | None = None,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        method: str | None = None,
        jac: str | Callable | None = None,
        tr_solver: str | None = None,
        x_scale: ArrayLike | str | float = 1.0,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int | None = None,
    ) -> OptimizationConfig:
        """Select optimization method and prepare configuration.

        Args:
            f: Model function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess (auto-detected if None)
            bounds: Parameter bounds as (lower, upper)
            method: Optimization method ('trf', 'lm', 'dogbox', or None for auto)
            jac: Jacobian computation method
            tr_solver: Trust region solver ('exact', 'lsmr', or None for auto)
            x_scale: Parameter scaling
            ftol: Function tolerance
            xtol: Parameter tolerance
            gtol: Gradient tolerance
            max_nfev: Maximum function evaluations (auto if None)

        Returns:
            OptimizationConfig with all settings resolved

        Raises:
            ValueError: If configuration is invalid
        """
        ...

    def detect_parameter_count(
        self,
        f: Callable[..., ArrayLike],
        xdata: jax.Array,
    ) -> int:
        """Detect number of parameters from function signature.

        Uses inspection of function signature and optional probing
        with sample data to determine parameter count.

        Args:
            f: Model function to analyze
            xdata: Sample data for probing

        Returns:
            Number of parameters (excluding x)

        Raises:
            ValueError: If parameter count cannot be determined
        """
        ...

    def auto_initial_guess(
        self,
        n_params: int,
        bounds: tuple[jax.Array, jax.Array] | None,
    ) -> jax.Array:
        """Generate automatic initial parameter guess.

        Uses bounds midpoint if available, otherwise ones.

        Args:
            n_params: Number of parameters
            bounds: Parameter bounds or None

        Returns:
            Initial guess array of shape (n_params,)
        """
        ...


# =============================================================================
# CovarianceComputer Protocol and Entities
# =============================================================================


@dataclass(frozen=True, slots=True)
class CovarianceResult:
    """Result of covariance matrix computation.

    Attributes:
        pcov: Parameter covariance matrix, shape (n, n)
        perr: Parameter standard errors (sqrt of diagonal), shape (n,)
        method: Computation method used ('svd', 'cholesky', 'qr')
        condition_number: Condition number of Jacobian
        is_singular: True if Jacobian was singular/ill-conditioned
        sigma_used: True if sigma weights were applied
        absolute_sigma: True if sigma was treated as absolute
    """

    pcov: jax.Array
    perr: jax.Array
    method: Literal["svd", "cholesky", "qr"]
    condition_number: float
    is_singular: bool
    sigma_used: bool
    absolute_sigma: bool


@runtime_checkable
class CovarianceComputerProtocol(Protocol):
    """Protocol for covariance computation component.

    Implementations handle:
    1. Jacobian-based covariance via SVD
    2. Sigma transformation (1D and 2D)
    3. Absolute vs relative sigma handling
    4. Singularity detection and handling
    """

    def compute(
        self,
        result: OptimizeResult,
        n_data: int,
        *,
        sigma: jax.Array | None = None,
        absolute_sigma: bool = False,
        full_output: bool = False,
    ) -> CovarianceResult:
        """Compute parameter covariance from optimization result.

        Uses the Jacobian at the solution to compute covariance via:
        pcov = (J^T @ J)^(-1) * s_sq

        where s_sq is the residual variance.

        Args:
            result: OptimizeResult from LeastSquares
            n_data: Number of data points
            sigma: Observation uncertainties/weights
            absolute_sigma: If True, sigma is absolute uncertainty
            full_output: If True, include additional diagnostics

        Returns:
            CovarianceResult with covariance matrix and metadata

        Raises:
            ValueError: If Jacobian is unavailable or invalid
        """
        ...

    def create_sigma_transform(
        self,
        sigma: jax.Array,
        n_data: int,
    ) -> tuple[Callable, bool]:
        """Create sigma transformation function.

        Handles both 1D (diagonal) and 2D (full covariance) sigma.

        Args:
            sigma: Sigma array, shape (n,) or (n, n)
            n_data: Number of data points

        Returns:
            Tuple of (transform_func, is_2d)
            - transform_func: Function to apply sigma weighting
            - is_2d: True if sigma is full covariance matrix
        """
        ...

    def compute_condition_number(
        self,
        jacobian: jax.Array,
    ) -> float:
        """Compute condition number of Jacobian.

        Uses singular values: cond = max(s) / min(s)

        Args:
            jacobian: Jacobian matrix at solution

        Returns:
            Condition number (inf if singular)
        """
        ...


# =============================================================================
# StreamingCoordinator Protocol and Entities
# =============================================================================


@dataclass(frozen=True, slots=True)
class StreamingDecision:
    """Decision about streaming execution strategy.

    Attributes:
        strategy: Execution strategy to use
            - 'direct': Normal non-streaming fit
            - 'chunked': Simple chunked processing
            - 'hybrid': AdaptiveHybridStreamingOptimizer
            - 'auto_memory': Memory-aware automatic selection
        reason: Human-readable explanation of decision
        estimated_memory_mb: Estimated memory requirement
        available_memory_mb: Available system memory
        memory_pressure: Memory pressure ratio (0.0 to 1.0)
        chunk_size: Chunk size for chunked/hybrid strategies
        n_chunks: Number of chunks for chunked strategy
        hybrid_config: Configuration for hybrid strategy
    """

    strategy: Literal["direct", "chunked", "hybrid", "auto_memory"]
    reason: str
    estimated_memory_mb: float
    available_memory_mb: float
    memory_pressure: float
    chunk_size: int | None
    n_chunks: int | None
    hybrid_config: HybridStreamingConfig | None


@runtime_checkable
class StreamingCoordinatorProtocol(Protocol):
    """Protocol for streaming strategy coordination.

    Implementations handle:
    1. Memory estimation for dataset + Jacobian
    2. Available memory detection
    3. Strategy selection based on memory pressure
    4. Configuration of chunked/hybrid strategies
    """

    def decide(
        self,
        xdata: jax.Array,
        ydata: jax.Array,
        n_params: int,
        *,
        workflow: str = "auto",
        memory_limit_mb: float | None = None,
        force_streaming: bool = False,
    ) -> StreamingDecision:
        """Decide on streaming strategy for the dataset.

        Analyzes memory requirements and available resources to select
        the optimal execution strategy.

        Args:
            xdata: Independent variable data
            ydata: Dependent variable data
            n_params: Number of parameters
            workflow: Workflow hint ('auto', 'streaming', 'hybrid', 'normal')
            memory_limit_mb: Override for memory limit detection
            force_streaming: If True, always use streaming

        Returns:
            StreamingDecision with strategy and configuration

        Raises:
            MemoryError: If dataset too large even for streaming
        """
        ...

    def estimate_memory(
        self,
        n_data: int,
        n_params: int,
        dtype_bytes: int = 8,
    ) -> float:
        """Estimate memory requirement in MB.

        Accounts for:
        - Data arrays (x, y, residuals)
        - Jacobian matrix (n_data x n_params)
        - Working arrays for optimization
        - JAX compilation overhead

        Args:
            n_data: Number of data points
            n_params: Number of parameters
            dtype_bytes: Bytes per element (8 for float64)

        Returns:
            Estimated memory in MB
        """
        ...

    def get_available_memory(self) -> float:
        """Get available system memory in MB.

        Uses psutil with caching for efficiency.

        Returns:
            Available memory in MB
        """
        ...

    def configure_hybrid(
        self,
        n_data: int,
        n_params: int,
        available_memory_mb: float,
    ) -> HybridStreamingConfig:
        """Configure hybrid streaming for dataset.

        Calculates optimal chunk size and strategy parameters.

        Args:
            n_data: Number of data points
            n_params: Number of parameters
            available_memory_mb: Available memory

        Returns:
            HybridStreamingConfig for the dataset
        """
        ...


# Type aliases for documentation
DataPreprocessor = DataPreprocessorProtocol
OptimizationSelector = OptimizationSelectorProtocol
CovarianceComputer = CovarianceComputerProtocol
StreamingCoordinator = StreamingCoordinatorProtocol
