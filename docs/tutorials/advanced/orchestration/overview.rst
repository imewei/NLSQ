Orchestration Overview
======================

.. versionadded:: 0.6.4

This page explains the v0.6.4 decomposition of the CurveFit class.

The Decomposition
-----------------

Before v0.6.4, ``CurveFit`` was a "god class" with 2500+ lines handling:

- Data validation
- Parameter detection
- Bounds processing
- Initial guess generation
- Method selection
- Memory strategy
- Optimization execution
- Covariance computation

This was split into four focused components:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Responsibility
     - Lines
   * - DataPreprocessor
     - Input validation, conversion
     - <300
   * - OptimizationSelector
     - Method selection, config
     - <350
   * - CovarianceComputer
     - Post-fit covariance
     - <350
   * - StreamingCoordinator
     - Memory strategy
     - <350

Data Flow
---------

.. code-block:: text

   Input (model, x, y, p0)
           │
           ▼
   ┌───────────────────────┐
   │   DataPreprocessor    │
   │   - Type conversion   │
   │   - NaN handling      │
   │   - Validation        │
   └───────────┬───────────┘
               │ PreprocessedData
               ▼
   ┌───────────────────────┐
   │  OptimizationSelector │
   │   - Param count       │
   │   - Bounds            │
   │   - Initial guess     │
   │   - Method selection  │
   └───────────┬───────────┘
               │ OptimizationConfig
               ▼
   ┌───────────────────────┐
   │ StreamingCoordinator  │
   │   - Memory analysis   │
   │   - Strategy select   │
   └───────────┬───────────┘
               │ StreamingDecision
               ▼
   ┌───────────────────────┐
   │    LeastSquares/TRF   │
   │   - Optimization      │
   └───────────┬───────────┘
               │ OptimizeResult
               ▼
   ┌───────────────────────┐
   │  CovarianceComputer   │
   │   - Jacobian @ soln   │
   │   - SVD decomp        │
   │   - Sigma transform   │
   └───────────┬───────────┘
               │
               ▼
         (popt, pcov)

Feature Flags
-------------

For gradual rollout, components can be toggled:

.. code-block:: bash

   # Use new DataPreprocessor
   export NLSQ_PREPROCESSOR_IMPL=new

   # Use legacy implementation
   export NLSQ_PREPROCESSOR_IMPL=legacy

This allows testing new implementations without breaking existing code.

Protocol Compliance
-------------------

Each component implements a protocol from ``nlsq/interfaces/orchestration_protocol.py``:

.. code-block:: python

   from nlsq.interfaces.orchestration_protocol import (
       DataPreprocessorProtocol,
       OptimizationSelectorProtocol,
       CovarianceComputerProtocol,
       StreamingCoordinatorProtocol,
   )


   # Components implement these protocols
   class DataPreprocessor:
       def preprocess(self, f, xdata, ydata, **kwargs) -> PreprocessedData: ...

Benefits
--------

**For users:**

- Same high-level API (``fit()``, ``curve_fit()``)
- No code changes required
- Performance maintained (actually 48% faster)

**For developers:**

- Easier to understand (smaller files)
- Easier to test (isolated components)
- Easier to extend (inject custom components)
- Easier to debug (clear boundaries)

Next Steps
----------

- :doc:`data_preprocessor` - Data preprocessing
- :doc:`optimization_selector` - Optimization configuration
- :doc:`covariance_computer` - Covariance estimation
- :doc:`streaming_coordinator` - Memory management
