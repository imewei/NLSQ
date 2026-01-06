"""Workflow Configuration and Selection Module.

This module provides memory-based optimizer selection and adaptive tolerance
calculation for NLSQ curve fitting operations.

Key Components
--------------
- ``OptimizationGoal`` enum: Defines optimization priorities (FAST, ROBUST, QUALITY, etc.)
- ``MemoryBudget`` dataclass: Computes memory requirements for optimizer selection
- ``MemoryBudgetSelector`` class: Selects optimal optimizer strategy based on memory
- ``calculate_adaptive_tolerances()``: Returns size-appropriate convergence tolerances
- ``ClusterDetector`` class: Detects HPC cluster environments (PBS Pro)

Examples
--------
Memory-based optimizer selection:

>>> from nlsq.core.workflow import MemoryBudgetSelector
>>> selector = MemoryBudgetSelector(safety_factor=0.75)
>>> strategy, config = selector.select(n_points=5_000_000, n_params=10)
>>> if strategy == "streaming":
...     pass  # Use HybridStreamingOptimizer
>>> elif strategy == "chunked":
...     pass  # Use LargeDatasetFitter
>>> else:
...     pass  # Use standard curve_fit()

Adaptive tolerance calculation:

>>> from nlsq.core.workflow import calculate_adaptive_tolerances, OptimizationGoal
>>> tols = calculate_adaptive_tolerances(n_points=5_000_000, goal=OptimizationGoal.QUALITY)
>>> tols['gtol']  # Returns tighter tolerance for QUALITY goal
1e-08

Cluster detection for HPC environments:

>>> from nlsq.core.workflow import ClusterDetector
>>> detector = ClusterDetector()
>>> cluster_info = detector.detect()
>>> if cluster_info:
...     print(f"Running on cluster: {cluster_info.total_gpus} GPUs")
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LDMemoryConfig


class OptimizationGoal(Enum):
    """Optimization goals that influence workflow selection and tolerances.

    Each goal represents a different optimization priority, affecting:
    - Convergence tolerances (gtol, ftol, xtol)
    - Multi-start enablement
    - Memory/speed tradeoffs

    Attributes
    ----------
    FAST : auto
        Prioritize speed with local optimization only.
        Uses one tier looser tolerances, skips multi-start.
        Best for: quick exploration, well-conditioned problems.

    ROBUST : auto
        Standard tolerances with multi-start for better global optimum.
        Uses dataset-appropriate tolerances, enables multi-start via `MultiStartOrchestrator`.
        Best for: production use, unknown problem conditioning.

    GLOBAL : auto
        Synonym for ROBUST. Emphasizes global optimization.
        Same behavior as ROBUST, provided for semantic clarity.

    MEMORY_EFFICIENT : auto
        Minimize memory usage with standard tolerances.
        Prioritizes streaming/chunking with smaller chunk sizes.
        Best for: memory-constrained environments, very large datasets.

    QUALITY : auto
        Highest precision/accuracy as TOP PRIORITY.
        Uses one tier tighter tolerances, enables multi-start, runs validation passes.
        Best for: publication-quality results, critical applications.
    """

    FAST = auto()
    ROBUST = auto()
    GLOBAL = auto()  # Alias for ROBUST
    MEMORY_EFFICIENT = auto()
    QUALITY = auto()

    @classmethod
    def normalize(cls, goal: "OptimizationGoal") -> "OptimizationGoal":
        """Normalize GLOBAL to ROBUST since they have same behavior.

        Parameters
        ----------
        goal : OptimizationGoal
            The goal to normalize.

        Returns
        -------
        OptimizationGoal
            ROBUST if goal was GLOBAL, otherwise the original goal.
        """
        if goal == cls.GLOBAL:
            return cls.ROBUST
        return goal


# ============================================================================
# Memory Budget API (new unified memory-based optimizer selection)
# ============================================================================


@dataclass(slots=True, frozen=True)
class MemoryBudget:
    """Computed memory budget for optimizer selection.

    This immutable dataclass represents the computed memory requirements
    and available resources for automatic optimizer strategy selection.
    Use the `compute()` factory method to create instances.

    Attributes
    ----------
    available_gb : float
        Available system memory in GB (CPU or GPU depending on target).
    threshold_gb : float
        Safe memory threshold = available_gb × safety_factor.
    data_gb : float
        Memory required for data arrays (x_data, y_data).
    jacobian_gb : float
        Memory required for full Jacobian matrix.
    peak_gb : float
        Estimated peak memory = data_gb + 1.3 × jacobian_gb + solver overhead.

    Examples
    --------
    >>> budget = MemoryBudget.compute(n_points=10_000_000, n_params=10)
    >>> print(f"Available: {budget.available_gb:.1f} GB")
    >>> print(f"Peak estimate: {budget.peak_gb:.2f} GB")
    >>> print(f"Fits in memory: {budget.fits_in_memory}")
    """

    available_gb: float
    threshold_gb: float
    data_gb: float
    jacobian_gb: float
    peak_gb: float

    @property
    def fits_in_memory(self) -> bool:
        """Check if estimated peak memory fits within safe threshold.

        Returns
        -------
        bool
            True if peak_gb <= threshold_gb.
        """
        return self.peak_gb <= self.threshold_gb

    @property
    def data_fits(self) -> bool:
        """Check if data arrays alone fit within safe threshold.

        Returns
        -------
        bool
            True if data_gb <= threshold_gb.
        """
        return self.data_gb <= self.threshold_gb

    @classmethod
    def compute(
        cls,
        n_points: int,
        n_params: int,
        n_features: int = 1,
        dtype_bytes: int = 8,
        safety_factor: float = 0.75,
        memory_limit_gb: float | None = None,
        use_gpu: bool = False,
    ) -> "MemoryBudget":
        """Compute memory budget for a given dataset size.

        Parameters
        ----------
        n_points : int
            Number of data points.
        n_params : int
            Number of fit parameters.
        n_features : int, default=1
            Number of features in x_data (dimensions).
        dtype_bytes : int, default=8
            Bytes per element (8 for float64, 4 for float32).
        safety_factor : float, default=0.75
            Memory safety factor (0.75 means use 75% of available).
        memory_limit_gb : float | None, default=None
            Override memory limit in GB. If None, auto-detect.
        use_gpu : bool, default=False
            If True, use GPU memory instead of CPU memory.

        Returns
        -------
        MemoryBudget
            Computed memory budget with all fields populated.

        Raises
        ------
        ValueError
            If n_points <= 0, n_params <= 0, or safety_factor not in (0, 1].

        Examples
        --------
        >>> budget = MemoryBudget.compute(n_points=1_000_000, n_params=5)
        >>> budget.fits_in_memory
        True
        """
        # Validation
        if n_points <= 0:
            raise ValueError("n_points must be positive")
        if n_params <= 0:
            raise ValueError("n_params must be positive")
        if safety_factor <= 0 or safety_factor > 1.0:
            raise ValueError("safety_factor must be in (0, 1]")

        # Get available memory (lazy import to avoid circular dependency)
        if memory_limit_gb is not None:
            available_gb = memory_limit_gb
        else:
            available_gb = cls._detect_available_memory(use_gpu)

        # Compute memory estimates
        # Data arrays: x_data (n_points x n_features) + y_data (n_points)
        data_bytes = n_points * (n_features + 1) * dtype_bytes
        data_gb = data_bytes / (1024**3)

        # Jacobian: n_points x n_params
        jacobian_bytes = n_points * n_params * dtype_bytes
        jacobian_gb = jacobian_bytes / (1024**3)

        # Peak estimate: data + 1.3*jacobian (SVD working memory) + solver overhead
        # 1.3 factor accounts for SVD U, S, V matrices
        solver_overhead_gb = 0.1  # Fixed solver overhead (workspace, etc.)
        peak_gb = data_gb + 1.3 * jacobian_gb + solver_overhead_gb

        # Compute threshold
        threshold_gb = available_gb * safety_factor

        return cls(
            available_gb=available_gb,
            threshold_gb=threshold_gb,
            data_gb=data_gb,
            jacobian_gb=jacobian_gb,
            peak_gb=peak_gb,
        )

    @staticmethod
    def _detect_available_memory(use_gpu: bool = False) -> float:
        """Detect available system memory.

        Parameters
        ----------
        use_gpu : bool, default=False
            If True, detect GPU memory; otherwise detect CPU memory.

        Returns
        -------
        float
            Available memory in GB. Falls back to 8.0 GB if detection fails.
        """
        _FALLBACK_MEMORY_GB = 8.0  # Conservative fallback (FR-009)

        if use_gpu:
            try:
                from nlsq.streaming.large_dataset import GPUMemoryEstimator

                estimator = GPUMemoryEstimator()
                gpu_memory = estimator.get_available_gpu_memory_gb()
                if gpu_memory > 0:
                    return gpu_memory
                # No GPU available, fall back to CPU
                return MemoryBudget._detect_available_memory(use_gpu=False)
            except Exception:
                return _FALLBACK_MEMORY_GB
        else:
            try:
                from nlsq.streaming.large_dataset import MemoryEstimator

                return MemoryEstimator.get_available_memory_gb()
            except Exception:
                return _FALLBACK_MEMORY_GB


class MemoryBudgetSelector:
    """Selects optimal optimizer strategy based on memory budget.

    This class computes memory requirements and selects between STREAMING,
    CHUNKED, and STANDARD strategies based on three sequential memory
    comparisons.

    Decision Tree:
        1. data_gb > threshold_gb → STREAMING (data doesn't fit)
        2. peak_gb > threshold_gb → CHUNKED (Jacobian doesn't fit)
        3. else → STANDARD (everything fits)

    Parameters
    ----------
    safety_factor : float, default=0.75
        Memory safety factor (0.75 means use 75% of available memory).

    Examples
    --------
    >>> selector = MemoryBudgetSelector(safety_factor=0.75)
    >>> strategy, config = selector.select(n_points=5_000_000, n_params=10)
    >>> if strategy == "streaming":
    ...     # Use HybridStreamingOptimizer with config
    ...     pass
    >>> elif strategy == "chunked":
    ...     # Use LargeDatasetFitter with config
    ...     pass
    >>> else:
    ...     # Use standard curve_fit()
    ...     pass
    """

    def __init__(self, safety_factor: float = 0.75) -> None:
        """Initialize selector with safety factor.

        Parameters
        ----------
        safety_factor : float, default=0.75
            Memory safety factor (0.75 means use 75% of available memory).
        """
        self.safety_factor = safety_factor

    def select(
        self,
        n_points: int,
        n_params: int,
        n_features: int = 1,
        memory_limit_gb: float | None = None,
        goal: "OptimizationGoal | None" = None,
        use_gpu: bool = False,
        verbose: bool = False,
    ) -> tuple[str, "HybridStreamingConfig | LDMemoryConfig | None"]:
        """Select optimal optimizer strategy based on memory budget.

        Parameters
        ----------
        n_points : int
            Number of data points.
        n_params : int
            Number of fit parameters.
        n_features : int, default=1
            Number of features in x_data.
        memory_limit_gb : float | None, default=None
            Override memory limit in GB. If None, auto-detect.
        goal : OptimizationGoal | None, default=None
            Optimization goal (affects tolerances, not strategy selection).
        use_gpu : bool, default=False
            If True, use GPU memory instead of CPU memory.
        verbose : bool, default=False
            If True, log memory budget details and strategy selection reason.

        Returns
        -------
        tuple[str, config]
            - strategy: "streaming", "chunked", or "standard"
            - config: HybridStreamingConfig, LDMemoryConfig, or None

        Raises
        ------
        ValueError
            If n_points <= 0 or n_params <= 0.
        """
        import logging

        logger = logging.getLogger("nlsq")

        # Compute memory budget
        budget = MemoryBudget.compute(
            n_points=n_points,
            n_params=n_params,
            n_features=n_features,
            safety_factor=self.safety_factor,
            memory_limit_gb=memory_limit_gb,
            use_gpu=use_gpu,
        )

        # Log memory budget details if verbose
        if verbose:
            logger.info(
                f"[NLSQ] Memory budget: available={budget.available_gb:.1f} GB, "
                f"threshold={budget.threshold_gb:.1f} GB"
            )
            logger.info(
                f"[NLSQ] Estimates: data={budget.data_gb:.3f} GB, "
                f"jacobian={budget.jacobian_gb:.3f} GB, peak={budget.peak_gb:.3f} GB"
            )

        # Decision tree (FR-003):
        # 1. data_gb > threshold_gb → STREAMING
        if not budget.data_fits:
            if verbose:
                logger.info(
                    f"[NLSQ] Strategy: streaming (data {budget.data_gb:.2f} GB > "
                    f"threshold {budget.threshold_gb:.2f} GB)"
                )
            return self._create_streaming_config(budget, n_params, goal)

        # 2. peak_gb > threshold_gb → CHUNKED (but data fits)
        # Also apply 10% safety margin (FR-010)
        safety_margin_threshold = budget.threshold_gb * 0.9
        if budget.peak_gb > safety_margin_threshold:
            if verbose:
                logger.info(
                    f"[NLSQ] Strategy: chunked (peak {budget.peak_gb:.2f} GB > "
                    f"safety threshold {safety_margin_threshold:.2f} GB)"
                )
            return self._create_chunked_config(budget, n_params, goal)

        # 3. else → STANDARD
        if verbose:
            logger.info(
                f"[NLSQ] Strategy: standard (peak {budget.peak_gb:.2f} GB < "
                f"threshold {budget.threshold_gb:.2f} GB)"
            )
        return ("standard", None)

    def _create_streaming_config(
        self,
        budget: MemoryBudget,
        n_params: int,
        goal: "OptimizationGoal | None",
    ) -> tuple[str, "HybridStreamingConfig"]:
        """Create configuration for streaming strategy.

        Parameters
        ----------
        budget : MemoryBudget
            Computed memory budget.
        n_params : int
            Number of fit parameters.
        goal : OptimizationGoal | None
            Optimization goal.

        Returns
        -------
        tuple[str, HybridStreamingConfig]
            Strategy name and configuration.
        """
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Compute batch size based on available memory
        batch_size = self._compute_streaming_batch_size(budget, n_params)

        return (
            "streaming",
            HybridStreamingConfig(
                chunk_size=batch_size,  # HybridStreamingConfig uses chunk_size
                normalize=True,  # Always normalize for streaming
            ),
        )

    def _create_chunked_config(
        self,
        budget: MemoryBudget,
        n_params: int,
        goal: "OptimizationGoal | None",
    ) -> tuple[str, "LDMemoryConfig"]:
        """Create configuration for chunked strategy.

        Parameters
        ----------
        budget : MemoryBudget
            Computed memory budget.
        n_params : int
            Number of fit parameters.
        goal : OptimizationGoal | None
            Optimization goal.

        Returns
        -------
        tuple[str, LDMemoryConfig]
            Strategy name and configuration.
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        # Compute chunk size based on available memory
        chunk_size = self._compute_chunk_size(budget, n_params)

        return (
            "chunked",
            LDMemoryConfig(
                memory_limit_gb=budget.threshold_gb,
                safety_factor=self.safety_factor,
                min_chunk_size=1_000,
                max_chunk_size=1_000_000,
                streaming_batch_size=chunk_size,
            ),
        )

    def _compute_chunk_size(self, budget: MemoryBudget, n_params: int) -> int:
        """Compute optimal chunk size based on memory budget.

        Parameters
        ----------
        budget : MemoryBudget
            Computed memory budget.
        n_params : int
            Number of fit parameters.

        Returns
        -------
        int
            Optimal chunk size in data points.
        """
        # Target: use ~75% of threshold for chunk processing
        target_memory_gb = budget.threshold_gb * 0.75

        # Memory per point: data + jacobian row
        # data: (1 + 1) * 8 bytes = 16 bytes (x, y)
        # jacobian: n_params * 8 bytes
        bytes_per_point = (2 * 8) + (n_params * 8)
        gb_per_point = bytes_per_point / (1024**3)

        if gb_per_point > 0:
            computed_chunk = int(target_memory_gb / gb_per_point)
        else:
            computed_chunk = 100_000  # Default

        # Clamp to bounds (FR-007: 1K-1M range)
        return max(1_000, min(computed_chunk, 1_000_000))

    def _compute_streaming_batch_size(self, budget: MemoryBudget, n_params: int) -> int:
        """Compute optimal streaming batch size based on memory budget.

        Parameters
        ----------
        budget : MemoryBudget
            Computed memory budget.
        n_params : int
            Number of fit parameters.

        Returns
        -------
        int
            Optimal batch size in data points.
        """
        # Streaming needs smaller batches than chunked
        # Target: use ~50% of threshold for batch processing
        target_memory_gb = budget.threshold_gb * 0.5

        # Memory per point in streaming: lighter than chunked
        # data: (1 + 1) * 8 bytes = 16 bytes
        # gradient accumulation: n_params * 8 bytes
        bytes_per_point = (2 * 8) + (n_params * 8)
        gb_per_point = bytes_per_point / (1024**3)

        if gb_per_point > 0:
            computed_batch = int(target_memory_gb / gb_per_point)
        else:
            computed_batch = 50_000  # Default

        # Clamp to bounds
        return max(1_000, min(computed_batch, 1_000_000))


# Dataset size thresholds and their corresponding tolerances
# (max_points exclusive, tolerance)
_SIZE_TOLERANCE_TABLE: list[tuple[int, float]] = [
    (1_000, 1e-12),  # TINY: < 1K points
    (10_000, 1e-10),  # SMALL: 1K - 10K points
    (100_000, 1e-9),  # MEDIUM: 10K - 100K points
    (1_000_000, 1e-8),  # LARGE: 100K - 1M points
    (10_000_000, 1e-7),  # VERY_LARGE: 1M - 10M points
    (100_000_000, 1e-6),  # HUGE: 10M - 100M points
]
_MASSIVE_TOLERANCE = 1e-5  # > 100M points


def _get_size_tier_index(n_points: int) -> int:
    """Get the tier index for a given dataset size.

    Parameters
    ----------
    n_points : int
        Number of data points.

    Returns
    -------
    int
        Index into _SIZE_TOLERANCE_TABLE, or len(_SIZE_TOLERANCE_TABLE)
        for MASSIVE datasets.
    """
    for i, (max_points, _) in enumerate(_SIZE_TOLERANCE_TABLE):
        if n_points < max_points:
            return i
    return len(_SIZE_TOLERANCE_TABLE)  # MASSIVE


def _get_tolerance_by_index(index: int) -> float:
    """Get tolerance by tier index, clamped to valid range.

    Parameters
    ----------
    index : int
        Tier index (may be out of bounds, will be clamped).

    Returns
    -------
    float
        The tolerance for the (clamped) tier.
    """
    if index < 0:
        return _SIZE_TOLERANCE_TABLE[0][1]  # Tightest
    elif index >= len(_SIZE_TOLERANCE_TABLE):
        return _MASSIVE_TOLERANCE  # Loosest
    else:
        return _SIZE_TOLERANCE_TABLE[index][1]


def calculate_adaptive_tolerances(
    n_points: int,
    goal: OptimizationGoal | None = None,
) -> dict[str, float]:
    """Calculate adaptive tolerances based on dataset size and optimization goal.

    This function determines appropriate convergence tolerances (gtol, ftol, xtol)
    for the given dataset size, then applies goal-based adjustments:

    - "quality" goal: Use one tier tighter (lower) tolerances
    - "fast" goal: Use one tier looser (higher) tolerances
    - "robust"/"global"/"memory_efficient": Use standard tolerances for dataset size

    Parameters
    ----------
    n_points : int
        Number of data points in the dataset.
    goal : OptimizationGoal, optional
        Optimization goal to adjust tolerances. Default: None (use dataset-appropriate).

    Returns
    -------
    dict[str, float]
        Dictionary with 'gtol', 'ftol', 'xtol' keys and corresponding tolerance values.

    Examples
    --------
    >>> tols = calculate_adaptive_tolerances(5_000_000)
    >>> tols['gtol']
    1e-07

    >>> tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.QUALITY)
    >>> tols['gtol']  # One tier tighter
    1e-08

    >>> tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.FAST)
    >>> tols['gtol']  # One tier looser
    1e-06
    """
    # Get base tier index from dataset size
    tier_index = _get_size_tier_index(n_points)

    # Apply goal-based tier shifting
    if goal is not None:
        # Normalize GLOBAL to ROBUST
        goal = OptimizationGoal.normalize(goal)

        if goal == OptimizationGoal.QUALITY:
            # Use one tier tighter (shift toward smaller datasets)
            tier_index = tier_index - 1
        elif goal == OptimizationGoal.FAST:
            # Use one tier looser (shift toward larger datasets)
            tier_index = tier_index + 1
        # ROBUST, MEMORY_EFFICIENT: use base tier (no shift)

    # Get effective tolerance (clamped to valid range)
    tolerance = _get_tolerance_by_index(tier_index)

    return {
        "gtol": tolerance,
        "ftol": tolerance,
        "xtol": tolerance,
    }


# =============================================================================
# Cluster Detection and Distributed Processing
# =============================================================================


@dataclass(slots=True)
class ClusterInfo:
    """Information about detected cluster environment.

    This dataclass contains information about the cluster configuration,
    including node count, GPUs per node, and total resources available.

    Parameters
    ----------
    node_count : int
        Number of nodes in the cluster.
    gpus_per_node : int
        Number of GPUs per node.
    total_gpus : int
        Total number of GPUs across all nodes.
    node_list : list[str]
        List of node hostnames.
    scheduler : str
        Cluster scheduler type ('pbs', 'local', or 'unknown').
    job_id : str | None
        PBS job ID if available.
    interconnect : str | None
        Interconnect type if detectable (e.g., 'infiniband').

    Examples
    --------
    >>> cluster_info = ClusterInfo(
    ...     node_count=6,
    ...     gpus_per_node=8,
    ...     total_gpus=48,
    ...     node_list=["node01", "node02", "node03", "node04", "node05", "node06"],
    ...     scheduler="pbs",
    ...     job_id="12345.pbs_server",
    ... )
    >>> cluster_info.total_gpus
    48
    """

    node_count: int
    gpus_per_node: int
    total_gpus: int
    node_list: list[str]
    scheduler: str = "unknown"
    job_id: str | None = None
    interconnect: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize cluster info to dictionary.

        Returns
        -------
        dict
            Dictionary representation of cluster info.
        """
        return {
            "node_count": self.node_count,
            "gpus_per_node": self.gpus_per_node,
            "total_gpus": self.total_gpus,
            "node_list": self.node_list,
            "scheduler": self.scheduler,
            "job_id": self.job_id,
            "interconnect": self.interconnect,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClusterInfo":
        """Create ClusterInfo from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with cluster info fields.

        Returns
        -------
        ClusterInfo
            ClusterInfo instance.
        """
        return cls(
            node_count=d.get("node_count", 1),
            gpus_per_node=d.get("gpus_per_node", 0),
            total_gpus=d.get("total_gpus", 0),
            node_list=d.get("node_list", []),
            scheduler=d.get("scheduler", "unknown"),
            job_id=d.get("job_id"),
            interconnect=d.get("interconnect"),
        )


class ClusterDetector:
    """Detector for cluster environments and GPU configurations.

    This class auto-detects PBS cluster environments via $PBS_NODEFILE
    and single-node multi-GPU configurations via JAX's device API.

    Supports:
    - PBS Pro cluster manager
    - Single-node multi-GPU (2-8 GPUs)
    - Multi-node HPC clusters (10-100 nodes, 8x A100 GPUs per node)

    Examples
    --------
    >>> detector = ClusterDetector()
    >>> cluster_info = detector.detect()
    >>> if cluster_info is not None:
    ...     print(f"Cluster detected: {cluster_info.node_count} nodes")
    ...     print(f"Total GPUs: {cluster_info.total_gpus}")
    ... else:
    ...     print("Not in cluster environment")

    Check for PBS specifically:

    >>> if detector.is_pbs_environment():
    ...     cluster_info = detector.detect_pbs()
    ...     print(f"PBS Job ID: {cluster_info.job_id}")
    """

    # Default GPUs per node for HPC environments (A100 nodes)
    DEFAULT_GPUS_PER_NODE = 8

    def __init__(self, default_gpus_per_node: int = 8) -> None:
        """Initialize ClusterDetector.

        Parameters
        ----------
        default_gpus_per_node : int, optional
            Default number of GPUs per node when not auto-detectable.
            Default: 8 (for A100 HPC nodes).
        """
        self._default_gpus_per_node = default_gpus_per_node

    def detect(self) -> ClusterInfo | None:
        """Auto-detect cluster environment.

        Tries PBS first, then falls back to local multi-GPU detection.
        Returns None if not in a cluster environment (single CPU-only machine).

        Returns
        -------
        ClusterInfo or None
            ClusterInfo if cluster detected, None otherwise.

        Examples
        --------
        >>> detector = ClusterDetector()
        >>> info = detector.detect()
        >>> if info:
        ...     print(f"Running on {info.scheduler} with {info.total_gpus} GPUs")
        """
        # Try PBS environment first
        if self.is_pbs_environment():
            return self.detect_pbs()

        # Try local multi-GPU
        local_info = self.detect_local_gpus()
        if local_info and local_info.total_gpus > 0:
            return local_info

        # Not in cluster environment
        return None

    def is_pbs_environment(self) -> bool:
        """Check if running in PBS cluster environment.

        Returns
        -------
        bool
            True if PBS_NODEFILE environment variable is set.
        """
        return "PBS_NODEFILE" in os.environ

    def detect_pbs(self) -> ClusterInfo | None:
        """Detect PBS Pro cluster configuration.

        Parses PBS_NODEFILE to determine node count and list.
        GPU count per node is either auto-detected via JAX or uses default.

        Returns
        -------
        ClusterInfo or None
            ClusterInfo with PBS configuration, or None if not in PBS environment.

        Notes
        -----
        PBS_NODEFILE contains one line per allocated processor slot.
        For GPU jobs, typically each GPU gets one line per node.
        """
        nodefile_path = os.environ.get("PBS_NODEFILE")
        if not nodefile_path:
            return None

        try:
            # Parse PBS_NODEFILE
            nodefile = Path(nodefile_path)
            if not nodefile.exists():
                return None

            with open(nodefile) as f:
                lines = f.read().strip().split("\n")

            if not lines or not lines[0]:
                return None

            # Get unique nodes (PBS lists each slot, often duplicates)
            unique_nodes = list(dict.fromkeys(lines))  # Preserves order
            node_count = len(unique_nodes)

            # Try to detect GPUs per node via JAX
            gpus_per_node = self._detect_gpus_per_node()
            if gpus_per_node == 0:
                # Fallback to default
                gpus_per_node = self._default_gpus_per_node

            # Get PBS job ID
            job_id = os.environ.get("PBS_JOBID")

            # Detect interconnect (heuristic based on common setups)
            interconnect = self._detect_interconnect()

            return ClusterInfo(
                node_count=node_count,
                gpus_per_node=gpus_per_node,
                total_gpus=node_count * gpus_per_node,
                node_list=unique_nodes,
                scheduler="pbs",
                job_id=job_id,
                interconnect=interconnect,
            )

        except (OSError, ValueError):
            return None

    def detect_local_gpus(self) -> ClusterInfo | None:
        """Detect local multi-GPU configuration.

        Uses JAX's device API to enumerate available GPUs on the local node.

        Returns
        -------
        ClusterInfo or None
            ClusterInfo with local GPU configuration, or None if detection fails.
        """
        try:
            gpu_count = self._detect_gpus_per_node()
            if gpu_count == 0:
                return None

            import socket

            hostname = socket.gethostname()

            return ClusterInfo(
                node_count=1,
                gpus_per_node=gpu_count,
                total_gpus=gpu_count,
                node_list=[hostname],
                scheduler="local",
                job_id=None,
                interconnect=None,
            )

        except Exception:
            return None

    def _detect_gpus_per_node(self) -> int:
        """Detect number of GPUs on the local node via JAX.

        Returns
        -------
        int
            Number of GPU devices, or 0 if no GPUs or detection fails.
        """
        try:
            import jax

            devices = jax.devices()
            gpu_count = sum(
                1 for d in devices if getattr(d, "platform", "cpu") != "cpu"
            )
            return gpu_count
        except Exception:
            return 0

    def _detect_interconnect(self) -> str | None:
        """Detect interconnect type (heuristic).

        Returns
        -------
        str or None
            Interconnect type ('infiniband', 'ethernet') or None.
        """
        # Check for Infiniband indicators
        if Path("/sys/class/infiniband").exists():
            return "infiniband"

        # Check for common IB environment variables (OpenMPI)
        # Note: Environment variable names are case-sensitive and this one uses lowercase
        if os.environ.get("OMPI_MCA_btl_openib_allow_ib"):  # noqa: SIM112
            return "infiniband"

        return None


@dataclass(slots=True)
class MultiGPUConfig:
    """Configuration for multi-GPU data parallelism.

    This class holds configuration for distributing data across multiple GPUs
    using JAX's pmap/pjit primitives.

    Parameters
    ----------
    n_devices : int
        Number of GPU devices to use.
    shard_axis : int
        Axis along which to shard data. Default: 0 (batch dimension).
    use_pmap : bool
        Use pmap for data parallelism. Default: True.
    use_pjit : bool
        Use pjit for more flexible sharding. Default: False.
    per_device_batch_size : int
        Batch size per device. Default: 10000.

    Examples
    --------
    >>> config = MultiGPUConfig(n_devices=4, per_device_batch_size=5000)
    >>> config.total_batch_size
    20000
    """

    n_devices: int
    shard_axis: int = 0
    use_pmap: bool = True
    use_pjit: bool = False
    per_device_batch_size: int = 10000

    @property
    def total_batch_size(self) -> int:
        """Total batch size across all devices."""
        return self.n_devices * self.per_device_batch_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "n_devices": self.n_devices,
            "shard_axis": self.shard_axis,
            "use_pmap": self.use_pmap,
            "use_pjit": self.use_pjit,
            "per_device_batch_size": self.per_device_batch_size,
        }


def get_multi_gpu_config(
    cluster_info: ClusterInfo | None = None,
) -> MultiGPUConfig | None:
    """Generate multi-GPU sharding configuration.

    Creates a MultiGPUConfig based on detected cluster or local GPU setup.

    Parameters
    ----------
    cluster_info : ClusterInfo, optional
        Cluster information from ClusterDetector. If None, auto-detects.

    Returns
    -------
    MultiGPUConfig or None
        Configuration for multi-GPU processing, or None if no GPUs available.

    Examples
    --------
    >>> config = get_multi_gpu_config()
    >>> if config:
    ...     print(f"Using {config.n_devices} GPUs with batch size {config.total_batch_size}")
    """
    if cluster_info is None:
        detector = ClusterDetector()
        cluster_info = detector.detect()

    if cluster_info is None or cluster_info.total_gpus == 0:
        return None

    # For single-node, use all local GPUs
    if cluster_info.node_count == 1:
        n_devices = cluster_info.gpus_per_node
        per_device_batch = 10000
    else:
        # For multi-node, use GPUs on current node (pjit handles distribution)
        n_devices = cluster_info.gpus_per_node
        per_device_batch = 50000  # Larger batches for distributed

    return MultiGPUConfig(
        n_devices=n_devices,
        shard_axis=0,
        use_pmap=cluster_info.node_count == 1,  # pmap for single-node
        use_pjit=cluster_info.node_count > 1,  # pjit for multi-node
        per_device_batch_size=per_device_batch,
    )


def create_distributed_config(cluster_info: ClusterInfo) -> dict[str, Any]:
    """Create distributed processing configuration for HPC clusters.

    Generates configuration suitable for PBS Pro multi-node setup with
    appropriate chunk sizes, checkpointing, and memory settings.

    Parameters
    ----------
    cluster_info : ClusterInfo
        Cluster information from ClusterDetector.

    Returns
    -------
    dict
        Configuration dictionary for distributed processing.

    Examples
    --------
    >>> detector = ClusterDetector()
    >>> cluster_info = detector.detect()
    >>> if cluster_info:
    ...     dist_config = create_distributed_config(cluster_info)
    ...     print(f"Chunk size: {dist_config['chunk_size']}")
    """
    # Calculate memory per node (estimate based on A100 config)
    # A100 has 40GB or 80GB GPU memory; assume 80GB per GPU
    gpu_memory_per_node_gb = cluster_info.gpus_per_node * 80  # Conservative

    # For distributed, chunk size should be larger to amortize communication
    # But not so large that it overflows GPU memory
    chunk_size = min(
        1_000_000,  # Max 1M points per chunk
        max(
            100_000,  # Min 100K points per chunk
            int(gpu_memory_per_node_gb * 1e9 / (8 * 100)),  # ~100 bytes per point
        ),
    )

    # Enable checkpointing for fault tolerance in long-running distributed jobs
    enable_checkpoints = cluster_info.node_count > 1 or cluster_info.total_gpus > 4

    return {
        "tier": "STREAMING_CHECKPOINT",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": min(cluster_info.total_gpus, 20),  # Scale with GPUs
        "chunk_size": chunk_size,
        "enable_checkpoints": enable_checkpoints,
        "checkpoint_frequency": 50,  # Checkpoint every 50 iterations
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "distributed": True,
        "n_devices": cluster_info.total_gpus,
        "nodes": cluster_info.node_count,
        "gpus_per_node": cluster_info.gpus_per_node,
        "scheduler": cluster_info.scheduler,
    }


def create_checkpoint_directory(base_dir: str | Path | None = None) -> str:
    """Create a checkpoint directory with timestamp.

    Creates a directory at ./nlsq_checkpoints/YYYYMMDD_HHMMSS/ for storing
    optimization checkpoints. Integrates with HybridStreamingConfig.enable_checkpoints.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory for checkpoints. Default: ./nlsq_checkpoints

    Returns
    -------
    str
        Absolute path to the created checkpoint directory.

    Examples
    --------
    >>> checkpoint_dir = create_checkpoint_directory()
    >>> # Returns path like './nlsq_checkpoints/20251219_143052/'
    """
    if base_dir is None:
        base_dir = Path.cwd() / "nlsq_checkpoints"
    else:
        base_dir = Path(base_dir)

    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = base_dir / timestamp

    # Create directory (including parents if needed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return str(checkpoint_dir)


__all__ = [
    "ClusterDetector",
    "ClusterInfo",
    "MemoryBudget",
    "MemoryBudgetSelector",
    "MultiGPUConfig",
    "OptimizationGoal",
    "calculate_adaptive_tolerances",
    "create_checkpoint_directory",
    "create_distributed_config",
    "get_multi_gpu_config",
]
