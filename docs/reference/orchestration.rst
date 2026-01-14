Orchestration Components
========================

.. versionadded:: 0.6.4

The orchestration module provides decomposed components for curve fitting,
enabling modular testing, customization, and gradual migration via feature flags.

Overview
--------

The monolithic ``CurveFit`` class has been decomposed into four focused components:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Responsibility
   * - :class:`~nlsq.core.orchestration.DataPreprocessor`
     - Input validation, array conversion, NaN handling
   * - :class:`~nlsq.core.orchestration.OptimizationSelector`
     - Method selection, bounds preparation, initial guess
   * - :class:`~nlsq.core.orchestration.CovarianceComputer`
     - SVD-based covariance, sigma transformation
   * - :class:`~nlsq.core.orchestration.StreamingCoordinator`
     - Memory analysis, streaming strategy selection

DataPreprocessor
----------------

.. autoclass:: nlsq.core.orchestration.DataPreprocessor
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.core.orchestration import DataPreprocessor

   preprocessor = DataPreprocessor()
   data = preprocessor.preprocess(
       f=my_model,
       xdata=raw_x,
       ydata=raw_y,
       sigma=uncertainties,
       check_finite=True,
       nan_policy="omit",
   )

   print(f"Valid points: {data.n_points}")
   print(f"NaNs removed: {data.has_nans_removed}")

OptimizationSelector
--------------------

.. autoclass:: nlsq.core.orchestration.OptimizationSelector
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.core.orchestration import OptimizationSelector

   selector = OptimizationSelector()
   config = selector.select(
       f=my_model,
       xdata=data.xdata,
       ydata=data.ydata,
       p0=None,  # Auto-detect
       bounds=([0, 0], [10, 10]),
       method=None,  # Auto-select
   )

   print(f"Method: {config.method}")
   print(f"Parameters: {config.n_params}")

CovarianceComputer
------------------

.. autoclass:: nlsq.core.orchestration.CovarianceComputer
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.core.orchestration import CovarianceComputer

   computer = CovarianceComputer()
   cov_result = computer.compute(
       result=optimize_result,
       n_data=len(ydata),
       sigma=uncertainties,
       absolute_sigma=True,
   )

   print(f"Covariance:\n{cov_result.pcov}")
   print(f"Errors: {cov_result.perr}")

StreamingCoordinator
--------------------

.. autoclass:: nlsq.core.orchestration.StreamingCoordinator
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.core.orchestration import StreamingCoordinator

   coordinator = StreamingCoordinator()
   decision = coordinator.decide(
       xdata=large_x,
       ydata=large_y,
       n_params=5,
       workflow="auto",
   )

   print(f"Strategy: {decision.strategy}")
   print(f"Reason: {decision.reason}")
   print(f"Memory pressure: {decision.memory_pressure:.1%}")

Data Classes
------------

PreprocessedData
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.interfaces.orchestration_protocol.PreprocessedData
   :members:

OptimizationConfig
~~~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.interfaces.orchestration_protocol.OptimizationConfig
   :members:

CovarianceResult
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.interfaces.orchestration_protocol.CovarianceResult
   :members:

StreamingDecision
~~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.interfaces.orchestration_protocol.StreamingDecision
   :members:

Feature Flags
-------------

Control implementation selection via environment variables for gradual rollout:

.. code-block:: bash

   # Use old implementation (for rollback)
   export NLSQ_PREPROCESSOR_IMPL=old

   # Use new implementation (default after rollout)
   export NLSQ_PREPROCESSOR_IMPL=new

   # Global rollout percentage (0-100)
   export NLSQ_REFACTOR_ROLLOUT_PERCENT=50

.. autoclass:: nlsq.core.feature_flags.FeatureFlags
   :members:
   :show-inheritance:

See Also
--------

- :doc:`core_api` - Main curve fitting API
- :doc:`large_data` - Large dataset handling
- :doc:`/tutorials/advanced/architecture/index` - Architecture overview
