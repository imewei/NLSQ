"""StreamingCoordinator component for CurveFit decomposition.

Handles memory analysis, streaming strategy selection, and configuration
for large-scale curve fitting operations.

Reference: specs/017-curve-fit-decomposition/spec.md FR-004
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from nlsq.interfaces.orchestration_protocol import StreamingDecision

if TYPE_CHECKING:
    import jax

    from nlsq.streaming.hybrid_config import HybridStreamingConfig


# Default fallback memory when detection fails (16 GB)
_DEFAULT_FALLBACK_MEMORY_MB = 16.0 * 1024


class StreamingCoordinator:
    """Coordinator for streaming strategy selection.

    Handles:
    1. Memory estimation for dataset + Jacobian
    2. Available memory detection
    3. Strategy selection based on memory pressure
    4. Configuration of chunked/hybrid strategies

    Example:
        >>> coordinator = StreamingCoordinator()
        >>> decision = coordinator.decide(
        ...     xdata=x_array,
        ...     ydata=y_array,
        ...     n_params=5,
        ... )
        >>> if decision.strategy == "hybrid":
        ...     config = decision.hybrid_config
        ...     # Use hybrid streaming optimizer
    """

    def __init__(self, safety_factor: float = 0.75) -> None:
        """Initialize StreamingCoordinator.

        Args:
            safety_factor: Memory safety factor (0.75 means use 75% of available)
        """
        self.safety_factor = safety_factor

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
        # Convert to numpy for size calculations
        ydata_np = np.asarray(ydata)
        n_data = len(ydata_np)

        # Estimate memory requirements
        estimated_mb = self.estimate_memory(n_data, n_params)

        # Get available memory
        if memory_limit_mb is not None:
            available_mb = memory_limit_mb
        else:
            available_mb = self.get_available_memory()

        # Apply safety factor
        usable_mb = available_mb * self.safety_factor

        # Calculate memory pressure
        memory_pressure = estimated_mb / usable_mb if usable_mb > 0 else 1.0
        memory_pressure = min(memory_pressure, 1.0)  # Cap at 1.0

        # Decide strategy
        if force_streaming:
            strategy, reason, chunk_size, n_chunks, hybrid_config = (
                self._decide_forced_streaming(n_data, n_params, usable_mb)
            )
        elif workflow == "streaming":
            strategy, reason, chunk_size, n_chunks, hybrid_config = (
                self._decide_streaming_hint(
                    n_data, n_params, usable_mb, memory_pressure
                )
            )
        else:
            strategy, reason, chunk_size, n_chunks, hybrid_config = self._decide_auto(
                n_data, n_params, usable_mb, memory_pressure
            )

        return StreamingDecision(
            strategy=strategy,
            reason=reason,
            estimated_memory_mb=estimated_mb,
            available_memory_mb=available_mb,
            memory_pressure=memory_pressure,
            chunk_size=chunk_size,
            n_chunks=n_chunks,
            hybrid_config=hybrid_config,
        )

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
        # Data arrays: x, y, residuals (3 arrays of n_data)
        data_bytes = 3 * n_data * dtype_bytes

        # Jacobian: n_data x n_params
        jacobian_bytes = n_data * n_params * dtype_bytes

        # Working arrays for optimization (estimate: 5x parameter arrays)
        working_bytes = 5 * n_params * n_params * dtype_bytes

        # JAX compilation overhead (estimate: 20% of data + jacobian)
        jax_overhead_bytes = 0.2 * (data_bytes + jacobian_bytes)

        # Total in MB
        total_mb = (
            data_bytes + jacobian_bytes + working_bytes + jax_overhead_bytes
        ) / (1024 * 1024)

        return total_mb

    @lru_cache(maxsize=1)
    def get_available_memory(self) -> float:
        """Get available system memory in MB.

        Uses psutil with caching for efficiency.

        Returns:
            Available memory in MB
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            return float(mem.available) / (1024 * 1024)
        except ImportError:
            # Fallback if psutil not available
            return _DEFAULT_FALLBACK_MEMORY_MB

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
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Calculate chunk size to fit in memory
        # Target: Jacobian chunk should use ~50% of available memory
        target_jacobian_mb = available_memory_mb * 0.5
        jacobian_per_point_mb = (n_params * 8) / (1024 * 1024)

        if jacobian_per_point_mb > 0:
            chunk_size = int(target_jacobian_mb / jacobian_per_point_mb)
        else:
            chunk_size = n_data

        # Ensure chunk_size is reasonable
        chunk_size = max(1000, min(chunk_size, n_data, 100_000))

        return HybridStreamingConfig(
            chunk_size=chunk_size,
        )

    def _decide_auto(
        self,
        n_data: int,
        n_params: int,
        usable_mb: float,
        memory_pressure: float,
    ) -> tuple[
        Literal["direct", "chunked", "hybrid", "auto_memory"],
        str,
        int | None,
        int | None,
        HybridStreamingConfig | None,
    ]:
        """Decide strategy automatically based on memory pressure.

        Returns:
            Tuple of (strategy, reason, chunk_size, n_chunks, hybrid_config)
        """
        # Calculate data and peak memory requirements
        data_mb = (3 * n_data * 8) / (1024 * 1024)
        jacobian_mb = (n_data * n_params * 8) / (1024 * 1024)
        peak_mb = data_mb + jacobian_mb

        # Decision tree from MemoryBudgetSelector
        if data_mb > usable_mb:
            # Data alone exceeds memory -> streaming
            config = self.configure_hybrid(n_data, n_params, usable_mb)
            n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size
            return (
                "hybrid",
                f"Data ({data_mb:.1f}MB) exceeds usable memory ({usable_mb:.1f}MB)",
                config.chunk_size,
                n_chunks,
                config,
            )
        elif peak_mb > usable_mb:
            # Peak memory (with Jacobian) exceeds memory -> chunked
            config = self.configure_hybrid(n_data, n_params, usable_mb)
            n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size
            return (
                "chunked",
                f"Peak memory ({peak_mb:.1f}MB) exceeds usable memory ({usable_mb:.1f}MB)",
                config.chunk_size,
                n_chunks,
                config,
            )
        else:
            # Everything fits -> direct
            return (
                "direct",
                f"Data fits in memory (peak {peak_mb:.1f}MB < usable {usable_mb:.1f}MB)",
                None,
                None,
                None,
            )

    def _decide_forced_streaming(
        self,
        n_data: int,
        n_params: int,
        usable_mb: float,
    ) -> tuple[
        Literal["direct", "chunked", "hybrid", "auto_memory"],
        str,
        int | None,
        int | None,
        HybridStreamingConfig | None,
    ]:
        """Decide strategy when streaming is forced.

        Returns:
            Tuple of (strategy, reason, chunk_size, n_chunks, hybrid_config)
        """
        config = self.configure_hybrid(n_data, n_params, usable_mb)
        n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size

        return (
            "hybrid",
            "Streaming forced by user request",
            config.chunk_size,
            n_chunks,
            config,
        )

    def _decide_streaming_hint(
        self,
        n_data: int,
        n_params: int,
        usable_mb: float,
        memory_pressure: float,
    ) -> tuple[
        Literal["direct", "chunked", "hybrid", "auto_memory"],
        str,
        int | None,
        int | None,
        HybridStreamingConfig | None,
    ]:
        """Decide strategy when streaming is hinted.

        The streaming hint suggests streaming but doesn't force it for small data.

        Returns:
            Tuple of (strategy, reason, chunk_size, n_chunks, hybrid_config)
        """
        # If data is very small, still use direct
        if n_data < 1000:
            return (
                "direct",
                "Data too small for streaming (< 1000 points)",
                None,
                None,
                None,
            )

        # Otherwise, prefer streaming
        config = self.configure_hybrid(n_data, n_params, usable_mb)
        n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size

        return (
            "hybrid",
            "Streaming strategy requested via workflow hint",
            config.chunk_size,
            n_chunks,
            config,
        )
