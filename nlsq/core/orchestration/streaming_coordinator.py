"""StreamingCoordinator component for CurveFit decomposition.

Handles memory analysis, streaming strategy selection, and configuration
for large-scale curve fitting operations.

Reference: specs/017-curve-fit-decomposition/spec.md FR-004
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

    .. warning::
       Not currently wired into the live ``fit()``/``curve_fit()`` workflow
       path -- ``workflow='auto'`` uses ``nlsq.core.workflow.MemoryBudgetSelector``
       instead, which is not the same code and uses a different, more
       conservative memory-decision formula (a 1.3x SVD-workspace multiplier,
       fixed solver overhead, and a 10% safety margin on top of the chunked
       threshold; see ``_decide_auto`` below, which intentionally does not
       replicate that). If this coordinator is ever wired into the live path,
       its strategy boundary will disagree with ``MemoryBudgetSelector``'s
       right at the point where the two diverge -- reconcile the formulas
       before doing so.

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

        Raises:
            ValueError: If safety_factor is not in (0, 1]
        """
        if not (0 < safety_factor <= 1):
            msg = f"safety_factor must be in (0, 1], got {safety_factor}"
            raise ValueError(msg)
        self.safety_factor = safety_factor
        self._cached_available_memory: float | None = None

    @staticmethod
    def _data_and_jacobian_bytes(
        n_data: int,
        n_params: int,
        dtype_bytes: int = 8,
        x_multiplier: int = 1,
    ) -> tuple[int, int]:
        """Shared byte-sizing for the data and Jacobian arrays.

        Used by both `estimate_memory` (reporting) and `_decide_auto`
        (strategy selection) so the two can't silently drift apart.

        x_multiplier accounts for multi-dimensional xdata: shape
        (k, n_data) contributes k * n_data elements, not n_data, so a
        multi-output x isn't silently undercounted.

        Raises:
            ValueError: If x_multiplier < 1 (would shrink/corrupt data_bytes)
        """
        if x_multiplier < 1:
            msg = f"x_multiplier must be >= 1, got {x_multiplier}"
            raise ValueError(msg)
        data_bytes = (x_multiplier + 2) * n_data * dtype_bytes  # x, y, residuals
        jacobian_bytes = n_data * n_params * dtype_bytes
        return data_bytes, jacobian_bytes

    @staticmethod
    def _x_multiplier(xdata: jax.Array, n_data: int) -> int:
        """Ratio of xdata's total element count to n_data, rounded up.

        1 for ordinary 1-D xdata; >1 for multi-dimensional/multi-output x
        (e.g. shape (k, n_data)). Reads only array metadata (`.size`), no
        device->host transfer.
        """
        if n_data <= 0:
            return 1
        x_elems = int(xdata.size) if hasattr(xdata, "size") else n_data
        return max(1, -(-x_elems // n_data))  # ceil division

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
            ValueError: If n_params is negative or memory_limit_mb is not positive
            MemoryError: If dataset too large even for streaming
        """
        if n_params < 0:
            msg = f"n_params must be non-negative, got {n_params}"
            raise ValueError(msg)
        if memory_limit_mb is not None and memory_limit_mb <= 0:
            msg = f"memory_limit_mb must be positive, got {memory_limit_mb}"
            raise ValueError(msg)

        # Dataset size only — read it from array metadata without forcing a
        # device->host copy of the (potentially huge) data array.
        n_data = ydata.shape[0] if hasattr(ydata, "shape") else len(ydata)
        x_multiplier = self._x_multiplier(xdata, n_data)

        # Estimate memory requirements
        estimated_mb = self.estimate_memory(n_data, n_params, x_multiplier=x_multiplier)

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
                    n_data,
                    n_params,
                    usable_mb,
                    memory_pressure,
                )
            )
        elif workflow == "hybrid":
            strategy, reason, chunk_size, n_chunks, hybrid_config = (
                self._decide_forced_streaming(
                    n_data,
                    n_params,
                    usable_mb,
                    reason="Hybrid strategy requested via workflow='hybrid'",
                )
            )
        elif workflow == "normal":
            strategy, reason, chunk_size, n_chunks, hybrid_config = (
                "direct",
                "Direct execution forced by workflow='normal'",
                None,
                None,
                None,
            )
        else:
            strategy, reason, chunk_size, n_chunks, hybrid_config = self._decide_auto(
                n_data,
                n_params,
                usable_mb,
                memory_pressure,
                x_multiplier=x_multiplier,
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
        x_multiplier: int = 1,
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
            x_multiplier: xdata element count as a multiple of n_data (>1
                for multi-dimensional/multi-output x); see `_x_multiplier`

        Returns:
            Estimated memory in MB
        """
        data_bytes, jacobian_bytes = self._data_and_jacobian_bytes(
            n_data, n_params, dtype_bytes, x_multiplier
        )

        # Working arrays for optimization (estimate: 5x parameter arrays)
        working_bytes = 5 * n_params * n_params * dtype_bytes

        # JAX compilation overhead (estimate: 20% of data + jacobian)
        jax_overhead_bytes = 0.2 * (data_bytes + jacobian_bytes)

        # Total in MB
        total_mb = (
            data_bytes + jacobian_bytes + working_bytes + jax_overhead_bytes
        ) / (1024 * 1024)

        return total_mb

    def get_available_memory(self) -> float:
        """Get available system memory in MB.

        Cached once per coordinator lifetime (one streaming decision per fit).

        Returns:
            Available memory in MB
        """
        if self._cached_available_memory is not None:
            return self._cached_available_memory

        try:
            import psutil

            mem = psutil.virtual_memory()
            self._cached_available_memory = float(mem.available) / (1024 * 1024)
            return self._cached_available_memory
        except ImportError:
            # Fallback if psutil not available
            self._cached_available_memory = _DEFAULT_FALLBACK_MEMORY_MB
            return self._cached_available_memory

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

        # Clamp to an upper ceiling only -- never floor above what the memory
        # budget actually computed. A floor here would let the chunk's
        # Jacobian exceed target_jacobian_mb whenever n_params is large
        # enough that the budget-safe size falls below it, which is exactly
        # the low-memory/high-n_params case this method exists to protect.
        # Then cap at n_data so chunk never exceeds data size.
        chunk_size = max(1, min(chunk_size, 100_000))
        chunk_size = max(1, min(chunk_size, n_data))

        return HybridStreamingConfig(
            chunk_size=chunk_size,
        )

    def _decide_auto(
        self,
        n_data: int,
        n_params: int,
        usable_mb: float,
        memory_pressure: float,
        x_multiplier: int = 1,
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
        # Calculate data and peak memory requirements. Shares the same
        # byte-sizing helper as estimate_memory() so the two can't drift
        # apart into inconsistent formulas.
        data_bytes, jacobian_bytes = self._data_and_jacobian_bytes(
            n_data, n_params, x_multiplier=x_multiplier
        )
        data_mb = data_bytes / (1024 * 1024)
        jacobian_mb = jacobian_bytes / (1024 * 1024)
        peak_mb = data_mb + jacobian_mb
        pressure_note = f" (memory pressure {memory_pressure:.2f})"

        # Decision tree independent of MemoryBudgetSelector -- see the
        # class-level warning above about the formula intentionally not
        # matching that class's.
        if data_mb > usable_mb:
            # Data alone exceeds memory -> streaming
            config = self.configure_hybrid(n_data, n_params, usable_mb)
            n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size
            return (
                "hybrid",
                f"Data ({data_mb:.1f}MB) exceeds usable memory "
                f"({usable_mb:.1f}MB){pressure_note}",
                config.chunk_size,
                n_chunks,
                config,
            )
        if peak_mb > usable_mb:
            # Peak memory (with Jacobian) exceeds memory -> chunked
            config = self.configure_hybrid(n_data, n_params, usable_mb)
            n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size
            return (
                "chunked",
                f"Peak memory ({peak_mb:.1f}MB) exceeds usable memory "
                f"({usable_mb:.1f}MB){pressure_note}",
                config.chunk_size,
                n_chunks,
                config,
            )
        # Everything fits -> direct
        return (
            "direct",
            f"Data fits in memory (peak {peak_mb:.1f}MB < usable "
            f"{usable_mb:.1f}MB){pressure_note}",
            None,
            None,
            None,
        )

    def _decide_forced_streaming(
        self,
        n_data: int,
        n_params: int,
        usable_mb: float,
        reason: str = "Streaming forced by user request",
    ) -> tuple[
        Literal["direct", "chunked", "hybrid", "auto_memory"],
        str,
        int | None,
        int | None,
        HybridStreamingConfig | None,
    ]:
        """Decide strategy when streaming is forced.

        Args:
            n_data: Number of data points
            n_params: Number of parameters
            usable_mb: Usable memory budget
            reason: Human-readable reason string, overridable so
                `workflow='hybrid'` (a soft request) and `force_streaming=True`
                (a hard requirement) report distinct reasons.

        Returns:
            Tuple of (strategy, reason, chunk_size, n_chunks, hybrid_config)
        """
        config = self.configure_hybrid(n_data, n_params, usable_mb)
        n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size

        return (
            "hybrid",
            reason,
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
                f"Data too small for streaming (< 1000 points, memory "
                f"pressure {memory_pressure:.2f})",
                None,
                None,
                None,
            )

        # Otherwise, prefer streaming
        config = self.configure_hybrid(n_data, n_params, usable_mb)
        n_chunks = (n_data + config.chunk_size - 1) // config.chunk_size

        return (
            "hybrid",
            f"Streaming strategy requested via workflow hint (memory "
            f"pressure {memory_pressure:.2f})",
            config.chunk_size,
            n_chunks,
            config,
        )
