"""Workflow Configuration and Selection Module.

This module provides workflow configuration infrastructure for the unified `fit()` entry
point, including enums for workflow tiers, optimization goals, dataset sizes, and memory
tiers, plus adaptive tolerance calculation.

The workflow system auto-selects optimal fitting strategies based on dataset size,
available memory (CPU and GPU), and user-specified goals.

Examples
--------
Basic usage with enums:

>>> from nlsq.core.workflow import WorkflowTier, OptimizationGoal, DatasetSizeTier
>>> tier = WorkflowTier.STREAMING
>>> goal = OptimizationGoal.QUALITY
>>> size_tier = DatasetSizeTier.LARGE

Adaptive tolerance calculation:

>>> from nlsq.core.workflow import calculate_adaptive_tolerances
>>> tols = calculate_adaptive_tolerances(n_points=5_000_000, goal=OptimizationGoal.QUALITY)
>>> tols['gtol']  # Returns appropriate tolerance for dataset size and goal
1e-08

WorkflowConfig dataclass:

>>> from nlsq.core.workflow import WorkflowConfig
>>> config = WorkflowConfig(tier=WorkflowTier.CHUNKED, goal=OptimizationGoal.ROBUST)
>>> config_dict = config.to_dict()
>>> restored = WorkflowConfig.from_dict(config_dict)

Using workflow presets:

>>> from nlsq.core.workflow import WorkflowConfig, WORKFLOW_PRESETS
>>> config = WorkflowConfig.from_preset("quality")
>>> config.enable_multistart
True

Loading YAML configuration:

>>> from nlsq.core.workflow import load_yaml_config
>>> config = load_yaml_config()  # Loads from ./nlsq.yaml if it exists

WorkflowSelector for automatic workflow selection:

>>> from nlsq.core.workflow import WorkflowSelector, auto_select_workflow
>>> selector = WorkflowSelector()
>>> config = selector.select(n_points=5_000_000, n_params=5, goal=OptimizationGoal.QUALITY)
>>> # Or use the convenience function:
>>> config = auto_select_workflow(n_points=5_000_000, n_params=5, goal=OptimizationGoal.QUALITY)

Cluster detection for HPC environments:

>>> from nlsq.core.workflow import ClusterDetector
>>> detector = ClusterDetector()
>>> cluster_info = detector.detect()
>>> if cluster_info:
...     print(f"Running on cluster: {cluster_info.total_gpus} GPUs")
"""

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from nlsq.global_optimization import GlobalOptimizationConfig
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LDMemoryConfig


class WorkflowTier(Enum):
    """Workflow processing tiers based on dataset size and memory constraints.

    The workflow system selects the appropriate tier based on dataset size
    and available memory to optimize both performance and memory usage.

    Attributes
    ----------
    STANDARD : auto
        Standard `curve_fit()` for small datasets that fit in memory.
        Use when: dataset < 10K points or fits comfortably in available memory.
        Memory complexity: O(N) where N is number of data points.

    CHUNKED : auto
        `LargeDatasetFitter` with automatic chunking for medium-to-large datasets.
        Use when: dataset 10K-10M points, requires memory management.
        Memory complexity: O(chunk_size), processes data in chunks sequentially.

    STREAMING : auto
        `AdaptiveHybridStreamingOptimizer` for huge datasets with O(batch_size) memory.
        Use when: dataset 10M-100M points, limited memory available.
        Memory complexity: O(batch_size), uses mini-batch gradient descent.

    STREAMING_CHECKPOINT : auto
        Streaming with automatic checkpointing for massive datasets.
        Use when: dataset > 100M points, fault tolerance required.
        Enables resume capability for multi-hour fits.
    """

    STANDARD = auto()
    CHUNKED = auto()
    STREAMING = auto()
    STREAMING_CHECKPOINT = auto()


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


class DatasetSizeTier(Enum):
    """Dataset size tiers with associated tolerance recommendations.

    Each tier represents a range of dataset sizes with corresponding
    recommended convergence tolerances. Larger datasets use progressively
    looser tolerances to balance precision with computation time.

    Attributes
    ----------
    TINY : tuple
        < 1K points. Tolerance: 1e-12.
        Maximum precision, negligible compute cost.

    SMALL : tuple
        1K - 10K points. Tolerance: 1e-10.
        High precision, minimal overhead.

    MEDIUM : tuple
        10K - 100K points. Tolerance: 1e-9.
        Balanced precision/performance.

    LARGE : tuple
        100K - 1M points. Tolerance: 1e-8.
        Standard tolerances (current NLSQ default).

    VERY_LARGE : tuple
        1M - 10M points. Tolerance: 1e-7.
        Reduced precision, chunked processing.

    HUGE : tuple
        10M - 100M points. Tolerance: 1e-6.
        Streaming mode, practical limits.

    MASSIVE : tuple
        > 100M points. Tolerance: 1e-5.
        Streaming with checkpoints, convergence-focused.
    """

    # (max_points, tolerance)
    # max_points is exclusive upper bound (except MASSIVE which has no upper bound)
    TINY = (1_000, 1e-12)
    SMALL = (10_000, 1e-10)
    MEDIUM = (100_000, 1e-9)
    LARGE = (1_000_000, 1e-8)
    VERY_LARGE = (10_000_000, 1e-7)
    HUGE = (100_000_000, 1e-6)
    MASSIVE = (float("inf"), 1e-5)

    @property
    def max_points(self) -> float:
        """Maximum number of points for this tier (exclusive)."""
        return self.value[0]

    @property
    def tolerance(self) -> float:
        """Recommended base tolerance for this tier."""
        return self.value[1]

    @classmethod
    def from_n_points(cls, n_points: int) -> "DatasetSizeTier":
        """Determine the dataset size tier from number of points.

        Parameters
        ----------
        n_points : int
            Number of data points in the dataset.

        Returns
        -------
        DatasetSizeTier
            The appropriate tier for the given dataset size.

        Examples
        --------
        >>> DatasetSizeTier.from_n_points(500)
        <DatasetSizeTier.TINY: (1000, 1e-12)>
        >>> DatasetSizeTier.from_n_points(5_000_000)
        <DatasetSizeTier.VERY_LARGE: (10000000, 1e-07)>
        """
        # Ordered from smallest to largest
        tiers = [
            cls.TINY,
            cls.SMALL,
            cls.MEDIUM,
            cls.LARGE,
            cls.VERY_LARGE,
            cls.HUGE,
            cls.MASSIVE,
        ]
        for tier in tiers:
            if n_points < tier.max_points:
                return tier
        return cls.MASSIVE  # Fallback for any edge case


class MemoryTier(Enum):
    """Memory availability tiers for workflow selection.

    Used to classify system memory and influence workflow decisions.
    Thresholds are set for modern computing environments.

    Attributes
    ----------
    LOW : tuple
        < 16GB available memory.
        Constrained environment, prioritize streaming/small chunks.

    MEDIUM : tuple
        16-64GB available memory.
        Standard workstation, moderate chunk sizes.

    HIGH : tuple
        64-128GB available memory.
        High-memory workstation, larger chunks possible.

    VERY_HIGH : tuple
        > 128GB available memory.
        HPC/server environment, can handle large in-memory operations.
    """

    # (max_memory_gb, description)
    LOW = (16.0, "Constrained memory (<16GB)")
    MEDIUM = (64.0, "Standard memory (16-64GB)")
    HIGH = (128.0, "High memory (64-128GB)")
    VERY_HIGH = (float("inf"), "Very high memory (>128GB)")

    @property
    def max_memory_gb(self) -> float:
        """Maximum memory in GB for this tier (exclusive upper bound)."""
        return self.value[0]

    @property
    def description(self) -> str:
        """Human-readable description of this tier."""
        return self.value[1]

    @classmethod
    def from_available_memory_gb(cls, available_memory_gb: float) -> "MemoryTier":
        """Determine the memory tier from available memory.

        Parameters
        ----------
        available_memory_gb : float
            Available system memory in gigabytes.

        Returns
        -------
        MemoryTier
            The appropriate tier for the given memory.

        Examples
        --------
        >>> MemoryTier.from_available_memory_gb(8.0)
        <MemoryTier.LOW: (16.0, 'Constrained memory (<16GB)')>
        >>> MemoryTier.from_available_memory_gb(32.0)
        <MemoryTier.MEDIUM: (64.0, 'Standard memory (16-64GB)')>
        """
        if available_memory_gb < cls.LOW.max_memory_gb:
            return cls.LOW
        elif available_memory_gb < cls.MEDIUM.max_memory_gb:
            return cls.MEDIUM
        elif available_memory_gb < cls.HIGH.max_memory_gb:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


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
    # Get ordered list of tiers for shifting
    tiers_ordered = [
        DatasetSizeTier.TINY,
        DatasetSizeTier.SMALL,
        DatasetSizeTier.MEDIUM,
        DatasetSizeTier.LARGE,
        DatasetSizeTier.VERY_LARGE,
        DatasetSizeTier.HUGE,
        DatasetSizeTier.MASSIVE,
    ]

    # Determine base tier from dataset size
    base_tier = DatasetSizeTier.from_n_points(n_points)
    tier_index = tiers_ordered.index(base_tier)

    # Apply goal-based tier shifting
    if goal is not None:
        # Normalize GLOBAL to ROBUST
        goal = OptimizationGoal.normalize(goal)

        if goal == OptimizationGoal.QUALITY:
            # Use one tier tighter (shift toward smaller datasets)
            tier_index = max(0, tier_index - 1)
        elif goal == OptimizationGoal.FAST:
            # Use one tier looser (shift toward larger datasets)
            tier_index = min(len(tiers_ordered) - 1, tier_index + 1)
        # ROBUST, MEMORY_EFFICIENT: use base tier (no shift)

    # Get effective tolerance
    effective_tier = tiers_ordered[tier_index]
    tolerance = effective_tier.tolerance

    return {
        "gtol": tolerance,
        "ftol": tolerance,
        "xtol": tolerance,
    }


# =============================================================================
# Cluster Detection and Distributed Processing (Task Group 6)
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


# =============================================================================
# Predefined Workflow Presets (Task Group 5)
# =============================================================================

# Following GlobalOptimizationConfig.PRESETS pattern
WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "description": "Standard curve_fit() with default tolerances",
        "tier": "STANDARD",
        "goal": "ROBUST",
        "enable_multistart": False,
        "n_starts": 0,
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
    },
    "quality": {
        "description": "Highest precision with multi-start and tighter tolerances",
        "tier": "STANDARD",
        "goal": "QUALITY",
        "enable_multistart": True,
        "n_starts": 20,
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "enable_checkpoints": False,
    },
    "fast": {
        "description": "Speed-optimized with looser tolerances",
        "tier": "STANDARD",
        "goal": "FAST",
        "enable_multistart": False,
        "n_starts": 0,
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "enable_checkpoints": False,
    },
    "large_robust": {
        "description": "Chunked processing with multi-start for large datasets",
        "tier": "CHUNKED",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
    },
    "streaming": {
        "description": "AdaptiveHybridStreamingOptimizer for huge datasets",
        "tier": "STREAMING",
        "goal": "ROBUST",
        "enable_multistart": False,
        "n_starts": 0,
        "gtol": 1e-7,
        "ftol": 1e-7,
        "xtol": 1e-7,
        "enable_checkpoints": False,
    },
    "hpc_distributed": {
        "description": "Multi-GPU/node configuration for HPC clusters (PBS Pro)",
        "tier": "STREAMING_CHECKPOINT",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "enable_checkpoints": True,
        "checkpoint_frequency": 50,
        # Distributed processing settings
        "distributed": True,
        "chunk_size": 500_000,  # Larger chunks for distributed memory
        # Auto-detect will override these based on actual cluster
        "auto_detect_cluster": True,
    },
    "memory_efficient": {
        "description": "Minimize memory usage with streaming/chunking",
        "tier": "STREAMING",
        "goal": "MEMORY_EFFICIENT",
        "enable_multistart": False,
        "n_starts": 0,
        "gtol": 1e-7,
        "ftol": 1e-7,
        "xtol": 1e-7,
        "enable_checkpoints": False,
        "chunk_size": 5000,
    },
    # ==========================================================================
    # Scientific Application Presets
    # ==========================================================================
    "spectroscopy": {
        "description": "Optimized for peak fitting (Gaussian/Lorentzian/Voigt)",
        "tier": "STANDARD",
        "goal": "QUALITY",
        "enable_multistart": True,
        "n_starts": 15,
        "sampler": "lhs",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "enable_checkpoints": False,
    },
    "xpcs": {
        "description": "X-ray photon correlation spectroscopy with multi-scale parameters",
        "tier": "STREAMING",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
        # Uses hybrid streaming for tau/beta scale differences
        "normalize": True,
        "normalization_strategy": "bounds",
    },
    "saxs": {
        "description": "Small-angle X-ray scattering form factor fitting",
        "tier": "STANDARD",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
    },
    "kinetics": {
        "description": "Chemical/enzyme kinetics rate fitting",
        "tier": "STANDARD",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-9,
        "ftol": 1e-9,
        "xtol": 1e-9,
        "enable_checkpoints": False,
    },
    "dose_response": {
        "description": "IC50/EC50 dose-response curves (4PL/5PL)",
        "tier": "STANDARD",
        "goal": "QUALITY",
        "enable_multistart": True,
        "n_starts": 20,
        "sampler": "lhs",
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "enable_checkpoints": False,
    },
    "imaging": {
        "description": "2D Gaussian PSF fitting for microscopy/astronomy",
        "tier": "CHUNKED",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 5,
        "sampler": "lhs",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
        "chunk_size": 100000,
    },
    "timeseries": {
        "description": "Long time series with streaming (oscillations, decay)",
        "tier": "STREAMING",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-7,
        "ftol": 1e-7,
        "xtol": 1e-7,
        "enable_checkpoints": True,
        "checkpoint_frequency": 100,
    },
    "materials": {
        "description": "Materials science (stress-strain, thermal analysis)",
        "tier": "STANDARD",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
    },
    "binding": {
        "description": "Binding isotherms and adsorption curves",
        "tier": "STANDARD",
        "goal": "ROBUST",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-9,
        "ftol": 1e-9,
        "xtol": 1e-9,
        "enable_checkpoints": False,
    },
    "multimodal": {
        "description": "Problems with multiple local minima (multi-peak, phase)",
        "tier": "STANDARD",
        "goal": "GLOBAL",
        "enable_multistart": True,
        "n_starts": 30,
        "sampler": "sobol",  # Better coverage for multimodal
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "enable_checkpoints": False,
    },
    "synchrotron": {
        "description": "Synchrotron beamline data (large, high-precision)",
        "tier": "STREAMING",
        "goal": "QUALITY",
        "enable_multistart": True,
        "n_starts": 10,
        "sampler": "lhs",
        "gtol": 1e-9,
        "ftol": 1e-9,
        "xtol": 1e-9,
        "enable_checkpoints": True,
        "checkpoint_frequency": 50,
        "chunk_size": 100000,
    },
}


@dataclass(slots=True)
class WorkflowConfig:
    """Configuration dataclass for workflow selection and execution.

    This dataclass encapsulates all workflow-related settings for the
    unified `fit()` entry point, supporting serialization for checkpoints
    and configuration files.

    Parameters
    ----------
    tier : WorkflowTier, optional
        Processing tier. Default: STANDARD.
    goal : OptimizationGoal, optional
        Optimization goal. Default: ROBUST.
    gtol : float, optional
        Gradient tolerance. Default: 1e-8.
    ftol : float, optional
        Function tolerance. Default: 1e-8.
    xtol : float, optional
        Parameter tolerance. Default: 1e-8.
    enable_multistart : bool, optional
        Enable multi-start optimization. Default: False.
    n_starts : int, optional
        Number of starting points for multi-start. Default: 10.
    sampler : str, optional
        Sampling strategy ('lhs', 'sobol', 'halton'). Default: 'lhs'.
    memory_limit_gb : float, optional
        Memory limit in GB. Default: None (auto-detect).
    chunk_size : int, optional
        Chunk size for chunked processing. Default: None (auto).
    enable_checkpoints : bool, optional
        Enable automatic checkpointing. Default: False.
    checkpoint_dir : str, optional
        Directory for checkpoints. Default: None.

    Examples
    --------
    >>> config = WorkflowConfig(
    ...     tier=WorkflowTier.CHUNKED,
    ...     goal=OptimizationGoal.QUALITY,
    ...     enable_multistart=True,
    ... )
    >>> config.to_dict()
    {'tier': 'CHUNKED', 'goal': 'QUALITY', ...}

    >>> config2 = WorkflowConfig.from_dict({'tier': 'STREAMING', 'goal': 'FAST'})
    >>> config2.tier
    <WorkflowTier.STREAMING: 3>

    Using presets:

    >>> config = WorkflowConfig.from_preset("quality")
    >>> config.enable_multistart
    True
    >>> config.gtol
    1e-10
    """

    tier: WorkflowTier = WorkflowTier.STANDARD
    goal: OptimizationGoal = OptimizationGoal.ROBUST

    # Tolerances
    gtol: float = 1e-8
    ftol: float = 1e-8
    xtol: float = 1e-8

    # Multi-start settings
    enable_multistart: bool = False
    n_starts: int = 10
    sampler: str = "lhs"
    center_on_p0: bool = True
    scale_factor: float = 1.0

    # Memory settings
    memory_limit_gb: float | None = None
    chunk_size: int | None = None

    # Checkpointing
    enable_checkpoints: bool = False
    checkpoint_dir: str | None = None

    # Private field for tracking preset origin
    _preset: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate sampler
        valid_samplers = ("lhs", "sobol", "halton")
        if self.sampler.lower() not in valid_samplers:
            raise ValueError(
                f"sampler must be one of {valid_samplers}, got '{self.sampler}'"
            )
        # Normalize sampler to lowercase
        object.__setattr__(self, "sampler", self.sampler.lower())

        # Validate tolerances
        if self.gtol <= 0:
            raise ValueError(f"gtol must be positive, got {self.gtol}")
        if self.ftol <= 0:
            raise ValueError(f"ftol must be positive, got {self.ftol}")
        if self.xtol <= 0:
            raise ValueError(f"xtol must be positive, got {self.xtol}")

        # Validate n_starts
        if self.n_starts < 0:
            raise ValueError(f"n_starts must be non-negative, got {self.n_starts}")

        # Validate scale_factor
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

        # Validate memory_limit_gb if provided
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError(
                f"memory_limit_gb must be positive, got {self.memory_limit_gb}"
            )

        # Validate chunk_size if provided
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization
            or checkpoint saving. Enum values are converted to strings.

        Examples
        --------
        >>> config = WorkflowConfig(tier=WorkflowTier.STREAMING)
        >>> d = config.to_dict()
        >>> d['tier']
        'STREAMING'
        """
        return {
            "tier": self.tier.name,
            "goal": self.goal.name,
            "gtol": self.gtol,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "enable_multistart": self.enable_multistart,
            "n_starts": self.n_starts,
            "sampler": self.sampler,
            "center_on_p0": self.center_on_p0,
            "scale_factor": self.scale_factor,
            "memory_limit_gb": self.memory_limit_gb,
            "chunk_size": self.chunk_size,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_dir": self.checkpoint_dir,
            "_preset": self._preset,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowConfig":
        """Deserialize configuration from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with configuration values. Enum values can be
            strings (names) or enum instances.

        Returns
        -------
        WorkflowConfig
            Configuration instance.

        Examples
        --------
        >>> d = {'tier': 'CHUNKED', 'goal': 'QUALITY', 'n_starts': 20}
        >>> config = WorkflowConfig.from_dict(d)
        >>> config.tier
        <WorkflowTier.CHUNKED: 2>
        >>> config.n_starts
        20
        """
        # Convert string enum values to enums
        tier_value = d.get("tier", "STANDARD")
        if isinstance(tier_value, str):
            tier = WorkflowTier[tier_value]
        elif isinstance(tier_value, WorkflowTier):
            tier = tier_value
        else:
            tier = WorkflowTier.STANDARD

        goal_value = d.get("goal", "ROBUST")
        if isinstance(goal_value, str):
            goal = OptimizationGoal[goal_value]
        elif isinstance(goal_value, OptimizationGoal):
            goal = goal_value
        else:
            goal = OptimizationGoal.ROBUST

        # Filter to known fields
        return cls(
            tier=tier,
            goal=goal,
            gtol=d.get("gtol", 1e-8),
            ftol=d.get("ftol", 1e-8),
            xtol=d.get("xtol", 1e-8),
            enable_multistart=d.get("enable_multistart", False),
            n_starts=d.get("n_starts", 10),
            sampler=d.get("sampler", "lhs"),
            center_on_p0=d.get("center_on_p0", True),
            scale_factor=d.get("scale_factor", 1.0),
            memory_limit_gb=d.get("memory_limit_gb"),
            chunk_size=d.get("chunk_size"),
            enable_checkpoints=d.get("enable_checkpoints", False),
            checkpoint_dir=d.get("checkpoint_dir"),
            _preset=d.get("_preset"),
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> "WorkflowConfig":
        """Create configuration from a named preset.

        Parameters
        ----------
        preset_name : str
            Name of the preset. One of: 'standard', 'quality', 'fast',
            'large_robust', 'streaming', 'hpc_distributed', 'memory_efficient'.

        Returns
        -------
        WorkflowConfig
            Configuration instance with preset values.

        Raises
        ------
        ValueError
            If preset_name is not a known preset.

        Examples
        --------
        >>> config = WorkflowConfig.from_preset('quality')
        >>> config.enable_multistart
        True
        >>> config.gtol
        1e-10

        >>> config = WorkflowConfig.from_preset('fast')
        >>> config.gtol
        1e-06
        """
        preset_name_lower = preset_name.lower()
        if preset_name_lower not in WORKFLOW_PRESETS:
            valid_presets = list(WORKFLOW_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Valid presets: {valid_presets}"
            )

        preset_values = WORKFLOW_PRESETS[preset_name_lower].copy()
        # Remove description field (not a config parameter)
        preset_values.pop("description", None)
        # Remove distributed-specific fields not in WorkflowConfig
        preset_values.pop("distributed", None)
        preset_values.pop("auto_detect_cluster", None)
        preset_values.pop("checkpoint_frequency", None)
        preset_values["_preset"] = preset_name_lower
        return cls.from_dict(preset_values)

    @property
    def preset(self) -> str | None:
        """The preset name if this config was created from a preset.

        Returns
        -------
        str or None
            Preset name ('standard', 'quality', etc.) or None if custom.
        """
        return self._preset

    def with_adaptive_tolerances(self, n_points: int) -> "WorkflowConfig":
        """Create a new config with tolerances adapted for dataset size.

        Parameters
        ----------
        n_points : int
            Number of data points in the dataset.

        Returns
        -------
        WorkflowConfig
            New configuration with adapted tolerances.

        Examples
        --------
        >>> config = WorkflowConfig(goal=OptimizationGoal.QUALITY)
        >>> adapted = config.with_adaptive_tolerances(5_000_000)
        >>> adapted.gtol  # Adapted for VERY_LARGE dataset + QUALITY goal shift
        1e-08
        """
        tolerances = calculate_adaptive_tolerances(n_points, self.goal)
        d = self.to_dict()
        d.update(tolerances)
        d["_preset"] = None  # Clear preset since we're modifying
        return self.from_dict(d)

    def with_overrides(self, **kwargs: Any) -> "WorkflowConfig":
        """Create a new config with specified overrides.

        Parameters
        ----------
        **kwargs
            Configuration fields to override.

        Returns
        -------
        WorkflowConfig
            New configuration with overrides applied.

        Examples
        --------
        >>> config = WorkflowConfig.from_preset('standard')
        >>> config2 = config.with_overrides(n_starts=20, enable_multistart=True)
        >>> config2.n_starts
        20
        """
        d = self.to_dict()
        d.update(kwargs)
        # Clear preset if we're overriding values
        if kwargs and "_preset" not in kwargs:
            d["_preset"] = None
        return self.from_dict(d)


# =============================================================================
# YAML Configuration Loading (Task Group 5)
# =============================================================================


def _check_yaml_available() -> bool:
    """Check if pyyaml is installed.

    Returns
    -------
    bool
        True if pyyaml is available, False otherwise.
    """
    try:
        import yaml

        return True
    except ImportError:
        return False


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any] | None:
    """Load workflow configuration from a YAML file.

    Loads configuration from ./nlsq.yaml in the current directory by default.
    Environment variables take precedence over YAML settings.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML configuration file. Default: ./nlsq.yaml

    Returns
    -------
    dict or None
        Configuration dictionary if file exists and is valid, None otherwise.

    Raises
    ------
    ImportError
        If pyyaml is not installed and a YAML file needs to be loaded.

    Notes
    -----
    pyyaml is an optional dependency. Install with: pip install pyyaml

    The YAML file should have the following structure::

        # Default workflow settings
        default_workflow: standard
        memory_limit_gb: 32.0

        # Custom workflow definitions
        workflows:
          my_custom_workflow:
            tier: CHUNKED
            goal: QUALITY
            enable_multistart: true
            n_starts: 15
            gtol: 1e-9
            ftol: 1e-9
            xtol: 1e-9

    Examples
    --------
    >>> config = load_yaml_config()  # Load from ./nlsq.yaml
    >>> config = load_yaml_config("/path/to/config.yaml")
    """
    if config_path is None:
        config_path = Path.cwd() / "nlsq.yaml"
    else:
        config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        return None

    # Check if pyyaml is available
    if not _check_yaml_available():
        raise ImportError(
            "pyyaml is required to load YAML configuration files. "
            "Install with: pip install pyyaml"
        )

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config if isinstance(config, dict) else None


def get_env_overrides() -> dict[str, Any]:
    """Get workflow configuration overrides from environment variables.

    Supported environment variables:
    - NLSQ_WORKFLOW_GOAL: Override the optimization goal
    - NLSQ_MEMORY_LIMIT_GB: Override the memory limit in GB
    - NLSQ_DEFAULT_WORKFLOW: Override the default workflow preset

    Returns
    -------
    dict
        Dictionary of configuration overrides from environment variables.

    Examples
    --------
    >>> import os
    >>> os.environ['NLSQ_WORKFLOW_GOAL'] = 'quality'
    >>> os.environ['NLSQ_MEMORY_LIMIT_GB'] = '16.0'
    >>> overrides = get_env_overrides()
    >>> overrides['goal']
    'QUALITY'
    >>> overrides['memory_limit_gb']
    16.0
    """
    overrides: dict[str, Any] = {}

    # NLSQ_WORKFLOW_GOAL
    goal_env = os.environ.get("NLSQ_WORKFLOW_GOAL")
    if goal_env:
        goal_upper = goal_env.upper()
        valid_goals = ["FAST", "ROBUST", "GLOBAL", "MEMORY_EFFICIENT", "QUALITY"]
        if goal_upper in valid_goals:
            overrides["goal"] = goal_upper
        else:
            warnings.warn(
                f"Invalid NLSQ_WORKFLOW_GOAL '{goal_env}'. "
                f"Valid values: {valid_goals}. Ignoring.",
                stacklevel=2,
            )

    # NLSQ_MEMORY_LIMIT_GB
    memory_env = os.environ.get("NLSQ_MEMORY_LIMIT_GB")
    if memory_env:
        try:
            memory_limit = float(memory_env)
            if memory_limit > 0:
                overrides["memory_limit_gb"] = memory_limit
            else:
                warnings.warn(
                    f"Invalid NLSQ_MEMORY_LIMIT_GB '{memory_env}'. Must be positive. Ignoring.",
                    stacklevel=2,
                )
        except ValueError:
            warnings.warn(
                f"Invalid NLSQ_MEMORY_LIMIT_GB '{memory_env}'. Must be a number. Ignoring.",
                stacklevel=2,
            )

    # NLSQ_DEFAULT_WORKFLOW
    workflow_env = os.environ.get("NLSQ_DEFAULT_WORKFLOW")
    if workflow_env:
        overrides["default_workflow"] = workflow_env.lower()

    return overrides


def load_config_with_overrides(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML with environment variable overrides.

    Environment variables take precedence over YAML settings.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML configuration file. Default: ./nlsq.yaml

    Returns
    -------
    dict
        Merged configuration dictionary.

    Examples
    --------
    >>> import os
    >>> os.environ['NLSQ_WORKFLOW_GOAL'] = 'quality'
    >>> config = load_config_with_overrides()
    >>> config.get('goal')
    'QUALITY'
    """
    # Start with empty config
    config: dict[str, Any] = {}

    # Load YAML config if available
    yaml_config = load_yaml_config(config_path)
    if yaml_config:
        config.update(yaml_config)

    # Apply environment variable overrides (takes precedence)
    env_overrides = get_env_overrides()
    config.update(env_overrides)

    return config


def get_custom_workflow(
    workflow_name: str,
    config_path: str | Path | None = None,
) -> WorkflowConfig | None:
    """Get a custom workflow definition from YAML configuration.

    Parameters
    ----------
    workflow_name : str
        Name of the custom workflow to retrieve.
    config_path : str or Path, optional
        Path to the YAML configuration file. Default: ./nlsq.yaml

    Returns
    -------
    WorkflowConfig or None
        WorkflowConfig if the custom workflow exists, None otherwise.

    Examples
    --------
    Given a nlsq.yaml file with::

        workflows:
          my_custom:
            tier: CHUNKED
            goal: QUALITY
            n_starts: 15

    >>> config = get_custom_workflow("my_custom")
    >>> config.tier
    <WorkflowTier.CHUNKED: 2>
    """
    yaml_config = load_yaml_config(config_path)
    if yaml_config is None:
        return None

    workflows = yaml_config.get("workflows", {})
    if not isinstance(workflows, dict):
        return None

    workflow_def = workflows.get(workflow_name)
    if workflow_def is None:
        return None

    if not isinstance(workflow_def, dict):
        return None

    # Validate custom workflow definition
    try:
        config = WorkflowConfig.from_dict(workflow_def)
        # Set the preset name to the custom workflow name
        object.__setattr__(config, "_preset", f"custom:{workflow_name}")
        return config
    except (ValueError, KeyError) as e:
        warnings.warn(
            f"Invalid custom workflow '{workflow_name}': {e}. Ignoring.",
            stacklevel=2,
        )
        return None


def validate_custom_workflow(workflow_def: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a custom workflow definition.

    Parameters
    ----------
    workflow_def : dict
        Dictionary containing workflow configuration.

    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message). error_message is None if valid.

    Examples
    --------
    >>> result, error = validate_custom_workflow({"tier": "CHUNKED", "goal": "QUALITY"})
    >>> result
    True

    >>> result, error = validate_custom_workflow({"tier": "INVALID"})
    >>> result
    False
    """
    try:
        WorkflowConfig.from_dict(workflow_def)
        return True, None
    except (ValueError, KeyError) as e:
        return False, str(e)


# Type alias for config return types from WorkflowSelector
ConfigType = Union[
    "LDMemoryConfig", "HybridStreamingConfig", "GlobalOptimizationConfig"
]


class WorkflowSelector:
    """Selector for automatic workflow selection based on dataset size and memory.

    This class implements the workflow tier selection matrix from the requirements,
    mapping dataset size and memory tier to the appropriate workflow tier and
    config class.

    The selection matrix is::

        Dataset Size    | Low (<16GB) | Medium (16-64GB) | High (64-128GB) | Very High (>128GB)
        ----------------|-------------|------------------|-----------------|-------------------
        Small (<10K)    | standard    | standard         | standard        | standard+quality
        Medium (10K-1M) | chunked     | standard         | standard+ms     | standard+ms
        Large (1M-10M)  | streaming   | chunked          | chunked+ms      | chunked+ms
        Huge (10M-100M) | stream+ckpt | streaming        | chunked         | chunked+ms
        Massive (>100M) | stream+ckpt | streaming+ckpt   | streaming       | streaming+ms

    Where:

    * ms = multi-start enabled
    * ckpt = checkpointing enabled

    Parameters
    ----------
    memory_limit_gb : float, optional
        Override memory limit in GB. If None, memory is auto-detected on each call.

    Examples
    --------
    >>> selector = WorkflowSelector()
    >>> config = selector.select(n_points=5_000, n_params=5)
    >>> # Returns LDMemoryConfig for small dataset with STANDARD tier behavior

    >>> config = selector.select(n_points=50_000_000, n_params=5)
    >>> # Returns HybridStreamingConfig for huge dataset

    >>> config = selector.select(n_points=5_000, n_params=5, goal=OptimizationGoal.QUALITY)
    >>> # Returns GlobalOptimizationConfig with multi-start enabled
    """

    def __init__(self, memory_limit_gb: float | None = None) -> None:
        """Initialize WorkflowSelector.

        Parameters
        ----------
        memory_limit_gb : float, optional
            Fixed memory limit in GB. If None (default), memory is
            auto-detected on each call using MemoryEstimator.
        """
        self._memory_limit_gb = memory_limit_gb

    def _get_available_memory_gb(self) -> float:
        """Get available memory in GB, re-evaluating on each call.

        Returns
        -------
        float
            Available memory in GB.
        """
        if self._memory_limit_gb is not None:
            return self._memory_limit_gb

        # Import here to avoid circular imports
        from nlsq.streaming.large_dataset import MemoryEstimator

        return MemoryEstimator.get_available_memory_gb()

    def _get_dataset_size_category(self, n_points: int) -> str:
        """Categorize dataset size for the selection matrix.

        Parameters
        ----------
        n_points : int
            Number of data points.

        Returns
        -------
        str
            One of: 'small', 'medium', 'large', 'huge', 'massive'
        """
        if n_points < 10_000:
            return "small"
        elif n_points < 1_000_000:
            return "medium"
        elif n_points < 10_000_000:
            return "large"
        elif n_points < 100_000_000:
            return "huge"
        else:
            return "massive"

    def _select_tier_and_options(
        self,
        size_category: str,
        memory_tier: MemoryTier,
        goal: OptimizationGoal | None,
    ) -> tuple[WorkflowTier, bool, bool]:
        """Select workflow tier and options from the matrix.

        Parameters
        ----------
        size_category : str
            Dataset size category: 'small', 'medium', 'large', 'huge', 'massive'
        memory_tier : MemoryTier
            Available memory tier.
        goal : OptimizationGoal, optional
            Optimization goal.

        Returns
        -------
        tuple[WorkflowTier, bool, bool]
            (tier, enable_multistart, enable_checkpoints)
        """
        # Normalize goal
        normalized_goal = OptimizationGoal.normalize(goal) if goal else None

        # Default options
        enable_multistart = False
        enable_checkpoints = False

        # Goal overrides: quality/robust enable multistart, fast disables
        if normalized_goal in (OptimizationGoal.QUALITY, OptimizationGoal.ROBUST):
            enable_multistart = True
        elif normalized_goal == OptimizationGoal.FAST:
            enable_multistart = False

        # Memory-efficient goal: prioritize streaming
        memory_efficient_mode = normalized_goal == OptimizationGoal.MEMORY_EFFICIENT

        # Selection matrix implementation
        if size_category == "small":
            # Small datasets: always STANDARD
            tier = WorkflowTier.STANDARD
            # Very high memory + quality: enable multi-start
            if (
                memory_tier == MemoryTier.VERY_HIGH
                and normalized_goal == OptimizationGoal.QUALITY
            ):
                enable_multistart = True
        elif size_category == "medium":
            # Medium datasets
            if memory_tier == MemoryTier.LOW or memory_efficient_mode:
                tier = WorkflowTier.CHUNKED
            else:
                tier = WorkflowTier.STANDARD
                # High/very high memory enables multi-start
                if memory_tier in (MemoryTier.HIGH, MemoryTier.VERY_HIGH):
                    enable_multistart = True
        elif size_category == "large":
            # Large datasets
            if memory_tier == MemoryTier.LOW or memory_efficient_mode:
                tier = WorkflowTier.STREAMING
            elif memory_tier == MemoryTier.MEDIUM:
                tier = WorkflowTier.CHUNKED
            else:  # HIGH or VERY_HIGH
                tier = WorkflowTier.CHUNKED
                enable_multistart = True
        elif size_category == "huge":
            # Huge datasets
            if memory_tier == MemoryTier.LOW or memory_efficient_mode:
                tier = WorkflowTier.STREAMING_CHECKPOINT
                enable_checkpoints = True
            elif memory_tier == MemoryTier.MEDIUM:
                tier = WorkflowTier.STREAMING
            elif memory_tier == MemoryTier.HIGH:
                tier = WorkflowTier.CHUNKED
            else:  # VERY_HIGH
                tier = WorkflowTier.CHUNKED
                enable_multistart = True
        # Massive datasets (>100M)
        elif (
            memory_tier in (MemoryTier.LOW, MemoryTier.MEDIUM) or memory_efficient_mode
        ):
            tier = WorkflowTier.STREAMING_CHECKPOINT
            enable_checkpoints = True
        elif memory_tier == MemoryTier.HIGH:
            tier = WorkflowTier.STREAMING
        else:  # VERY_HIGH
            tier = WorkflowTier.STREAMING
            enable_multistart = True

        # Fast goal always disables multi-start
        if normalized_goal == OptimizationGoal.FAST:
            enable_multistart = False

        return tier, enable_multistart, enable_checkpoints

    def select(
        self,
        n_points: int,
        n_params: int,
        goal: OptimizationGoal | None = None,
    ) -> ConfigType:
        """Select appropriate workflow configuration for the dataset.

        Re-evaluates memory on each call (no caching) per requirements.

        Parameters
        ----------
        n_points : int
            Number of data points in the dataset.
        n_params : int
            Number of parameters to fit.
        goal : OptimizationGoal, optional
            Optimization goal. Default: None (use ROBUST behavior).

        Returns
        -------
        LDMemoryConfig | HybridStreamingConfig | GlobalOptimizationConfig
            Configuration object for the selected workflow tier.

        Examples
        --------
        >>> selector = WorkflowSelector()
        >>> config = selector.select(5_000, 5, OptimizationGoal.FAST)
        >>> isinstance(config, LDMemoryConfig)
        True
        """
        # Get current memory availability (re-evaluate each call)
        available_memory_gb = self._get_available_memory_gb()
        memory_tier = MemoryTier.from_available_memory_gb(available_memory_gb)

        # Categorize dataset size
        size_category = self._get_dataset_size_category(n_points)

        # Select tier and options from matrix
        tier, enable_multistart, enable_checkpoints = self._select_tier_and_options(
            size_category, memory_tier, goal
        )

        # Calculate adaptive tolerances
        tolerances = calculate_adaptive_tolerances(n_points, goal)

        # Create appropriate config class
        return self._create_config(
            tier=tier,
            n_points=n_points,
            n_params=n_params,
            goal=goal,
            enable_multistart=enable_multistart,
            enable_checkpoints=enable_checkpoints,
            tolerances=tolerances,
            memory_limit_gb=available_memory_gb,
        )

    def _create_config(
        self,
        tier: WorkflowTier,
        n_points: int,
        n_params: int,
        goal: OptimizationGoal | None,
        enable_multistart: bool,
        enable_checkpoints: bool,
        tolerances: dict[str, float],
        memory_limit_gb: float,
    ) -> ConfigType:
        """Create the appropriate config class for the tier.

        Parameters
        ----------
        tier : WorkflowTier
            Selected workflow tier.
        n_points : int
            Number of data points.
        n_params : int
            Number of parameters.
        goal : OptimizationGoal, optional
            Optimization goal.
        enable_multistart : bool
            Whether multi-start is enabled.
        enable_checkpoints : bool
            Whether checkpointing is enabled.
        tolerances : dict
            Calculated tolerances (gtol, ftol, xtol).
        memory_limit_gb : float
            Available memory in GB.

        Returns
        -------
        LDMemoryConfig | HybridStreamingConfig | GlobalOptimizationConfig
            Appropriate config class instance.
        """
        # Import config classes (avoid circular imports)
        from nlsq.global_optimization import GlobalOptimizationConfig
        from nlsq.streaming.hybrid_config import HybridStreamingConfig
        from nlsq.streaming.large_dataset import LDMemoryConfig

        # Normalize goal for multi-start settings
        normalized_goal = OptimizationGoal.normalize(goal) if goal else None

        # Default n_starts based on goal
        n_starts = 10
        if normalized_goal == OptimizationGoal.QUALITY:
            n_starts = 20  # More starts for quality
        elif normalized_goal == OptimizationGoal.ROBUST:
            n_starts = 10
        elif normalized_goal == OptimizationGoal.FAST:
            n_starts = 0  # No multi-start for fast

        # If multi-start is enabled and it's the primary focus, return GlobalOptimizationConfig
        if enable_multistart and tier == WorkflowTier.STANDARD:
            return GlobalOptimizationConfig(
                n_starts=n_starts,
                sampler="lhs",
                center_on_p0=True,
                scale_factor=1.0,
                elimination_rounds=3 if n_starts > 10 else 2,
                elimination_fraction=0.5,
                batches_per_round=50,
            )

        # For STREAMING and STREAMING_CHECKPOINT, return HybridStreamingConfig
        if tier in (WorkflowTier.STREAMING, WorkflowTier.STREAMING_CHECKPOINT):
            # Calculate appropriate chunk size based on memory
            chunk_size = 10000
            if normalized_goal == OptimizationGoal.MEMORY_EFFICIENT:
                chunk_size = 5000

            # Set checkpoint_dir for STREAMING_CHECKPOINT tier
            checkpoint_dir = None
            if enable_checkpoints:
                checkpoint_dir = create_checkpoint_directory()

            return HybridStreamingConfig(
                normalize=True,
                normalization_strategy="auto",
                warmup_iterations=200,
                gauss_newton_tol=tolerances["gtol"],
                chunk_size=chunk_size,
                enable_checkpoints=enable_checkpoints,
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=100,
                precision="auto",
                enable_multistart=enable_multistart,
                n_starts=n_starts if enable_multistart else 0,
                multistart_sampler="lhs",
            )

        # For STANDARD and CHUNKED, return LDMemoryConfig
        # Calculate appropriate chunk size
        chunk_size = min(1_000_000, max(10_000, n_points // 10))
        if normalized_goal == OptimizationGoal.MEMORY_EFFICIENT:
            chunk_size = min(chunk_size, 100_000)

        return LDMemoryConfig(
            memory_limit_gb=memory_limit_gb,
            safety_factor=0.8,
            min_chunk_size=1000,
            max_chunk_size=chunk_size,
            use_streaming=tier
            in (WorkflowTier.STREAMING, WorkflowTier.STREAMING_CHECKPOINT),
            streaming_batch_size=50000,
            streaming_max_epochs=10,
        )


def auto_select_workflow(
    n_points: int,
    n_params: int,
    goal: OptimizationGoal | None = None,
    memory_limit_gb: float | None = None,
) -> ConfigType:
    """Automatically select workflow configuration for a dataset.

    This is a convenience wrapper around WorkflowSelector for simple use cases.
    Re-evaluates memory on each call (no caching) per requirements.

    Parameters
    ----------
    n_points : int
        Number of data points in the dataset.
    n_params : int
        Number of parameters to fit.
    goal : OptimizationGoal, optional
        Optimization goal. Default: None (use ROBUST behavior).
    memory_limit_gb : float, optional
        Override memory limit in GB. If None, memory is auto-detected.

    Returns
    -------
    LDMemoryConfig | HybridStreamingConfig | GlobalOptimizationConfig
        Configuration object for the selected workflow tier.

    Examples
    --------
    Basic usage:

    >>> config = auto_select_workflow(n_points=5_000, n_params=5)
    >>> # Returns config for small dataset with standard settings

    With goal specification:

    >>> config = auto_select_workflow(
    ...     n_points=5_000_000,
    ...     n_params=5,
    ...     goal=OptimizationGoal.QUALITY
    ... )
    >>> # Returns config with multi-start enabled, tighter tolerances

    With memory limit:

    >>> config = auto_select_workflow(
    ...     n_points=50_000_000,
    ...     n_params=5,
    ...     memory_limit_gb=8.0
    ... )
    >>> # Returns streaming config due to low memory
    """
    selector = WorkflowSelector(memory_limit_gb=memory_limit_gb)
    return selector.select(n_points, n_params, goal)


# =============================================================================
# Checkpoint System (Task Group 7)
# =============================================================================


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


def get_quality_n_starts(n_points: int) -> int:
    """Determine n_starts for quality goal based on dataset size.

    Smaller datasets can afford more starting points, while larger
    datasets use fewer to balance thoroughness with computation time.

    Parameters
    ----------
    n_points : int
        Number of data points in the dataset.

    Returns
    -------
    int
        Recommended number of starting points for multi-start optimization.

    Examples
    --------
    >>> get_quality_n_starts(5_000)
    20
    >>> get_quality_n_starts(5_000_000)
    10
    """
    size_tier = DatasetSizeTier.from_n_points(n_points)

    # Map dataset size to n_starts
    n_starts_map = {
        DatasetSizeTier.TINY: 25,
        DatasetSizeTier.SMALL: 20,
        DatasetSizeTier.MEDIUM: 15,
        DatasetSizeTier.LARGE: 10,
        DatasetSizeTier.VERY_LARGE: 10,
        DatasetSizeTier.HUGE: 8,
        DatasetSizeTier.MASSIVE: 5,
    }

    return n_starts_map.get(size_tier, 10)


# =============================================================================
# Validation Passes (Task Group 7)
# =============================================================================


def generate_perturbed_parameters(
    p0: list[float] | tuple[float, ...],
    n_perturbations: int = 5,
    perturbation_scale: float = 0.1,
) -> list[list[float]]:
    """Generate perturbed initial parameters for validation passes.

    Creates multiple perturbed versions of the initial parameters to test
    solution consistency. Used for quality goal validation.

    Parameters
    ----------
    p0 : list or tuple
        Initial parameters to perturb.
    n_perturbations : int, optional
        Number of perturbed parameter sets to generate. Default: 5.
    perturbation_scale : float, optional
        Scale of perturbation as fraction of parameter magnitude. Default: 0.1.

    Returns
    -------
    list[list[float]]
        List of perturbed parameter sets.

    Examples
    --------
    >>> p0 = [1.0, 2.0, 3.0]
    >>> perturbed = generate_perturbed_parameters(p0, n_perturbations=3)
    >>> len(perturbed)
    3
    """
    import random

    perturbed_params = []

    for _ in range(n_perturbations):
        perturbed = []
        for param in p0:
            # Scale perturbation by parameter magnitude (avoid division by zero)
            scale = (
                abs(param) * perturbation_scale if param != 0 else perturbation_scale
            )
            perturbation = random.uniform(-scale, scale)
            perturbed.append(param + perturbation)
        perturbed_params.append(perturbed)

    return perturbed_params


def compare_validation_results(
    results: list[dict[str, Any]],
    divergence_threshold: float = 0.1,
) -> tuple[bool, str | None]:
    """Compare validation pass results for consistency.

    Compares optimized parameters across validation passes and checks
    if solutions diverge significantly (>10% difference by default).

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries, each containing 'popt' key with optimized parameters.
    divergence_threshold : float, optional
        Maximum allowed relative difference between parameters. Default: 0.1 (10%).

    Returns
    -------
    tuple[bool, str | None]
        (is_converged, warning_message). is_converged is True if all results
        are consistent. warning_message is None if converged, otherwise contains
        warning details.

    Examples
    --------
    >>> results = [
    ...     {"popt": [1.0, 2.0], "cost": 0.01},
    ...     {"popt": [1.01, 2.01], "cost": 0.012},
    ... ]
    >>> is_converged, msg = compare_validation_results(results)
    >>> is_converged
    True
    """
    if len(results) < 2:
        return True, None

    # Extract optimized parameters from each result
    all_popt = [r.get("popt", []) for r in results]

    # Filter out empty results
    all_popt = [p for p in all_popt if p]
    if len(all_popt) < 2:
        return True, None

    # Check consistency between all pairs
    n_params = len(all_popt[0])
    max_divergence = 0.0
    divergent_param_idx = -1

    for i in range(len(all_popt)):
        for j in range(i + 1, len(all_popt)):
            for k in range(n_params):
                p1 = all_popt[i][k]
                p2 = all_popt[j][k]

                # Calculate relative difference
                if p1 == 0 and p2 == 0:
                    rel_diff = 0.0
                elif p1 == 0 or p2 == 0:
                    rel_diff = 1.0  # Max difference if one is zero
                else:
                    rel_diff = abs(p1 - p2) / max(abs(p1), abs(p2))

                if rel_diff > max_divergence:
                    max_divergence = rel_diff
                    divergent_param_idx = k

    if max_divergence > divergence_threshold:
        warning_msg = (
            f"Validation results differ by {max_divergence:.1%} on parameter {divergent_param_idx}. "
            f"Solutions may not have converged to the same optimum. "
            f"Consider increasing n_starts or checking initial parameter bounds."
        )
        return False, warning_msg

    return True, None


def get_quality_precision_config() -> dict[str, Any]:
    """Get precision configuration for quality goal.

    Returns configuration that leverages NLSQ's automatic float32->float64
    upgrade when precision issues are detected.

    Returns
    -------
    dict
        Precision configuration with 'precision' and 'description' keys.

    Examples
    --------
    >>> config = get_quality_precision_config()
    >>> config['precision']
    'auto'
    """
    return {
        "precision": "auto",
        "description": (
            "Automatic precision selection: Uses float32 for Phase 1 warmup, "
            "then upgrades to float64 for Phase 2+ when precision issues are "
            "detected. This provides optimal balance of speed and numerical stability."
        ),
    }


__all__ = [
    # Configuration
    "WORKFLOW_PRESETS",
    # Cluster Detection and Distributed Processing (Task Group 6)
    "ClusterDetector",
    "ClusterInfo",
    "DatasetSizeTier",
    "MemoryTier",
    "MultiGPUConfig",
    "OptimizationGoal",
    "WorkflowConfig",
    # Selection
    "WorkflowSelector",
    # Enums
    "WorkflowTier",
    "auto_select_workflow",
    "calculate_adaptive_tolerances",
    # Checkpointing and Quality Goal (Task Group 7)
    "compare_validation_results",
    "create_checkpoint_directory",
    "create_distributed_config",
    "generate_perturbed_parameters",
    # YAML Configuration (Task Group 5)
    "get_custom_workflow",
    "get_env_overrides",
    "get_multi_gpu_config",
    "get_quality_n_starts",
    "get_quality_precision_config",
    "load_config_with_overrides",
    "load_yaml_config",
    "validate_custom_workflow",
]
