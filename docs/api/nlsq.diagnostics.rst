nlsq.diagnostics package
=========================

.. currentmodule:: nlsq.diagnostics

.. automodule:: nlsq.diagnostics
   :noindex:

Overview
--------

The ``nlsq.diagnostics`` package provides a comprehensive **Model Health Diagnostics System**
for nonlinear least squares curve fitting. It helps users understand why a fit may have
issues and provides actionable recommendations for improving results.

Key Capabilities
----------------

- **Identifiability Analysis** - Detect structural and practical parameter unidentifiability
- **Gradient Health Monitoring** - Track gradient behavior during optimization
- **Parameter Sensitivity Analysis** - Identify stiff vs sensitive parameter directions
- **Health Reports** - Aggregated diagnostics with severity-based issue categorization
- **Plugin System** - Extensible architecture for domain-specific diagnostics

Quick Start
-----------

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.diagnostics import DiagnosticsConfig

    # Enable diagnostics during fitting
    result = curve_fit(model, x, y, p0=p0, compute_diagnostics=True)

    # Access the health report
    if result.diagnostics is not None:
        print(result.diagnostics.summary())

Enumerations
------------

.. autosummary::
   :toctree: generated/

   HealthStatus
   IssueSeverity
   IssueCategory
   DiagnosticLevel

Types and Data Classes
----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ModelHealthIssue
   AnalysisResult
   IdentifiabilityReport
   GradientHealthReport
   ParameterSensitivityReport
   PluginResult
   DiagnosticsReport
   DiagnosticsConfig
   ModelHealthReport

Analyzer Classes
----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   IdentifiabilityAnalyzer
   GradientMonitor
   ParameterSensitivityAnalyzer

Plugin System
-------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   DiagnosticPlugin
   PluginRegistry

.. autosummary::
   :toctree: generated/

   run_plugins

Factory Functions
-----------------

.. autosummary::
   :toctree: generated/

   create_health_report

Recommendations
---------------

.. autosummary::
   :toctree: generated/

   RECOMMENDATIONS
   get_recommendation

Issue Codes Reference
---------------------

The diagnostics system uses structured issue codes to identify specific problems:

**Identifiability Issues (IDENT-xxx)**

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Code
     - Severity
     - Description
   * - IDENT-001
     - CRITICAL
     - Structural unidentifiability: FIM is rank-deficient
   * - IDENT-002
     - WARNING
     - Practical unidentifiability: FIM has high condition number

**Correlation Issues (CORR-xxx)**

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Code
     - Severity
     - Description
   * - CORR-001
     - WARNING
     - Highly correlated parameters detected

**Gradient Issues (GRAD-xxx)**

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Code
     - Severity
     - Description
   * - GRAD-001
     - WARNING
     - Vanishing gradients detected during optimization
   * - GRAD-002
     - WARNING
     - Gradient imbalance across parameters
   * - GRAD-003
     - WARNING
     - Gradient stagnation detected

**Parameter Sensitivity Issues (SENS-xxx)**

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Code
     - Severity
     - Description
   * - SENS-001
     - WARNING
     - Wide sensitivity spread detected (large eigenvalue range)
   * - SENS-002
     - INFO
     - Low effective dimensionality

Usage Examples
--------------

Identifiability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze parameter identifiability from a Jacobian matrix:

.. code-block:: python

    import numpy as np
    from nlsq.diagnostics import DiagnosticsConfig, IdentifiabilityAnalyzer

    # Configure analysis thresholds
    config = DiagnosticsConfig(
        condition_threshold=1e8,
        correlation_threshold=0.95,
    )

    # Create analyzer
    analyzer = IdentifiabilityAnalyzer(config)

    # Analyze Jacobian (typically from OptimizeResult.jac)
    jacobian = result.jac  # Shape: (n_data, n_params)
    report = analyzer.analyze(jacobian)

    print(f"Condition number: {report.condition_number:.2e}")
    print(f"Numerical rank: {report.numerical_rank}/{report.n_params}")
    print(f"Health status: {report.health_status.name}")

    # Check for issues
    for issue in report.issues:
        print(f"[{issue.severity.name}] {issue.code}: {issue.message}")
        print(f"  Recommendation: {issue.recommendation}")

Gradient Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor gradient behavior during optimization using callbacks:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.diagnostics import DiagnosticsConfig, GradientMonitor

    # Create gradient monitor
    config = DiagnosticsConfig(
        vanishing_threshold=1e-6,
        imbalance_threshold=1e6,
        stagnation_window=10,
    )
    monitor = GradientMonitor(config)

    # Create callback for curve_fit
    callback = monitor.create_callback()

    # Fit with gradient monitoring
    result = curve_fit(model, x, y, p0=p0, callback=callback)

    # Get gradient health report
    grad_report = monitor.get_report()

    print(f"Health score: {grad_report.health_score:.2f}")
    print(f"Iterations monitored: {grad_report.n_iterations}")
    print(f"Vanishing detected: {grad_report.vanishing_detected}")
    print(f"Imbalance detected: {grad_report.imbalance_detected}")

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze eigenvalue spectrum to identify sensitivity behavior:

.. code-block:: python

    from nlsq.diagnostics import (
        DiagnosticsConfig,
        DiagnosticLevel,
        ParameterSensitivityAnalyzer,
    )

    # Enable full analysis for sensitivity spectrum detection
    config = DiagnosticsConfig(
        level=DiagnosticLevel.FULL,
        sloppy_threshold=1e-6,
    )

    analyzer = ParameterSensitivityAnalyzer(config)
    report = analyzer.analyze(jacobian)

    print(f"Wide sensitivity spread: {report.is_sloppy}")
    print(f"Eigenvalue range: {report.eigenvalue_range:.1f} orders of magnitude")
    print(f"Effective dimensionality: {report.effective_dimensionality:.1f}")
    print(f"Stiff directions: {report.stiff_indices}")
    print(f"Sloppy directions: {report.sloppy_indices}")

Creating Aggregated Health Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine all analyses into a unified health report:

.. code-block:: python

    from nlsq.diagnostics import (
        DiagnosticsConfig,
        IdentifiabilityAnalyzer,
        GradientMonitor,
        ParameterSensitivityAnalyzer,
        create_health_report,
    )

    config = DiagnosticsConfig(verbose=False, emit_warnings=False)

    # Run individual analyses
    ident_analyzer = IdentifiabilityAnalyzer(config)
    ident_report = ident_analyzer.analyze(jacobian)

    sens_analyzer = ParameterSensitivityAnalyzer(config)
    sens_report = sens_analyzer.analyze(jacobian)

    # Create aggregated report
    health_report = create_health_report(
        identifiability=ident_report,
        gradient_health=grad_report,  # From GradientMonitor
        sensitivity=sens_report,
        config=config,
    )

    # Access aggregated results
    print(f"Overall status: {health_report.status.name}")
    print(f"Health score: {health_report.health_score:.2f}")
    print(f"Total issues: {len(health_report.all_issues)}")

    # Get formatted summary
    print(health_report.summary())

    # Convert to dictionary for serialization
    data = health_report.to_dict()

Custom Diagnostic Plugins
~~~~~~~~~~~~~~~~~~~~~~~~~

Create domain-specific diagnostics that integrate with the health report:

.. code-block:: python

    import numpy as np
    from nlsq.diagnostics import (
        DiagnosticPlugin,
        PluginRegistry,
        PluginResult,
        ModelHealthIssue,
        IssueCategory,
        IssueSeverity,
    )
    from nlsq.diagnostics.recommendations import get_recommendation


    class OpticalScatteringPlugin:
        """Plugin for optical scattering parameter validation."""

        @property
        def name(self) -> str:
            return "optical-scattering"

        def analyze(
            self,
            jacobian: np.ndarray,
            parameters: np.ndarray,
            residuals: np.ndarray,
            **context,
        ) -> PluginResult:
            issues = []

            # Domain-specific validation: scattering coefficients must be positive
            if any(parameters < 0):
                issues.append(
                    ModelHealthIssue(
                        category=IssueCategory.IDENTIFIABILITY,
                        severity=IssueSeverity.CRITICAL,
                        code="OPTICAL-001",
                        message="Negative scattering coefficient detected",
                        affected_parameters=tuple(np.where(parameters < 0)[0]),
                        details={"negative_values": parameters[parameters < 0].tolist()},
                        recommendation="Ensure bounds enforce non-negative coefficients",
                    )
                )

            return PluginResult(
                plugin_name=self.name,
                data={"custom_metric": np.mean(np.abs(parameters))},
                issues=issues,
            )


    # Register the plugin
    PluginRegistry.register(OpticalScatteringPlugin())

    # Plugins are automatically included in health reports
    from nlsq.diagnostics import run_plugins

    plugin_results = run_plugins(jacobian, parameters, residuals)
    for name, result in plugin_results.items():
        print(f"{name}: {len(result.issues)} issues")

Configuration Reference
-----------------------

The ``DiagnosticsConfig`` dataclass controls all diagnostic thresholds:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``level``
     - BASIC
     - Diagnostic depth: BASIC or FULL (includes sensitivity analysis)
   * - ``condition_threshold``
     - 1e8
     - FIM condition number threshold for practical identifiability
   * - ``correlation_threshold``
     - 0.95
     - Correlation coefficient threshold for high correlation warning
   * - ``imbalance_threshold``
     - 1e6
     - Gradient imbalance ratio threshold
   * - ``vanishing_threshold``
     - 1e-6
     - Relative gradient magnitude threshold for vanishing detection
   * - ``sloppy_threshold``
     - 1e-6
     - Eigenvalue ratio threshold for sensitivity classification
   * - ``gradient_window_size``
     - 100
     - Sliding window size for gradient norm history
   * - ``stagnation_window``
     - 10
     - Iterations to check for gradient stagnation
   * - ``stagnation_tolerance``
     - 0.01
     - Relative tolerance for stagnation detection (1%)
   * - ``verbose``
     - True
     - Print diagnostic summary to console
   * - ``emit_warnings``
     - True
     - Emit Python warnings for critical issues

Memory Efficiency
-----------------

The diagnostics system is designed for memory efficiency:

- **GradientMonitor**: Uses bounded memory (<1KB) regardless of iteration count:
  - Sliding window (deque) for gradient norm history
  - Welford's online algorithm for running mean/variance per parameter

- **IdentifiabilityAnalyzer**: Computes SVD once, reuses results

- **ParameterSensitivityAnalyzer**: Uses eigenvalue decomposition from SVD

See Also
--------

- :doc:`nlsq.stability` - Numerical stability analysis (SVD fallback, condition monitoring)
- :doc:`nlsq.callbacks` - Callback functions for optimization monitoring
- :func:`nlsq.curve_fit` - Main curve fitting function with ``compute_diagnostics`` option
