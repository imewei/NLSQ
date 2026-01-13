"""Diagnostics facade for breaking circular dependencies.

This facade provides lazy access to diagnostics components,
breaking the circular import cycle between diagnostics.types and health_report.

Reference: specs/017-curve-fit-decomposition/spec.md FR-013
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlsq.diagnostics.types import DiagnosticLevel, DiagnosticsConfig
    from nlsq.utils.diagnostics import ConvergenceMonitor


class DiagnosticsFacade:
    """Facade for diagnostics components with lazy loading.

    This facade breaks the circular dependency between diagnostics.types
    and health_report by deferring all imports to method call time.

    Examples
    --------
    >>> facade = DiagnosticsFacade()
    >>> DiagnosticLevel = facade.get_diagnostic_level()
    >>> level = DiagnosticLevel.DETAILED
    """

    def get_diagnostic_level(self) -> type[DiagnosticLevel]:
        """Get the DiagnosticLevel enum.

        Returns
        -------
        type[DiagnosticLevel]
            The DiagnosticLevel enum for specifying diagnostic verbosity.
        """
        from nlsq.diagnostics.types import DiagnosticLevel

        return DiagnosticLevel

    def get_diagnostics_config(self) -> type[DiagnosticsConfig]:
        """Get the DiagnosticsConfig class.

        Returns
        -------
        type[DiagnosticsConfig]
            The DiagnosticsConfig class for configuring diagnostics.
        """
        from nlsq.diagnostics.types import DiagnosticsConfig

        return DiagnosticsConfig

    def get_convergence_monitor(self) -> type[ConvergenceMonitor]:
        """Get the ConvergenceMonitor class.

        Returns
        -------
        type[ConvergenceMonitor]
            The ConvergenceMonitor class for monitoring convergence patterns.
        """
        from nlsq.utils.diagnostics import ConvergenceMonitor

        return ConvergenceMonitor

    def create_diagnostics_config(self, **kwargs: Any) -> DiagnosticsConfig:
        """Create a DiagnosticsConfig instance.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to DiagnosticsConfig constructor.

        Returns
        -------
        DiagnosticsConfig
            A configured DiagnosticsConfig instance.
        """
        DiagnosticsConfig = self.get_diagnostics_config()
        return DiagnosticsConfig(**kwargs)

    def create_convergence_monitor(
        self, window_size: int = 10, sensitivity: float = 1.0
    ) -> ConvergenceMonitor:
        """Create a ConvergenceMonitor instance.

        Parameters
        ----------
        window_size : int
            Size of sliding window for pattern detection.
        sensitivity : float
            Sensitivity factor for pattern detection thresholds.

        Returns
        -------
        ConvergenceMonitor
            A configured ConvergenceMonitor instance.
        """
        ConvergenceMonitor = self.get_convergence_monitor()
        return ConvergenceMonitor(window_size=window_size, sensitivity=sensitivity)
