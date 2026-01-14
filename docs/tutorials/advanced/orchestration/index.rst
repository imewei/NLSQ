Orchestration Components (v0.6.4)
=================================

.. versionadded:: 0.6.4

The monolithic ``CurveFit`` class was decomposed into four focused components.
This chapter covers each component and how to use them directly.

.. toctree::
   :maxdepth: 1

   overview
   data_preprocessor
   optimization_selector
   covariance_computer
   streaming_coordinator
   facades

Chapter Overview
----------------

**Overview** (5 min)
   Why decomposition and the component architecture.

**DataPreprocessor** (10 min)
   Input validation, array conversion, NaN handling.

**OptimizationSelector** (10 min)
   Parameter detection, method selection, bounds processing.

**CovarianceComputer** (10 min)
   Post-fit covariance via SVD.

**StreamingCoordinator** (10 min)
   Memory analysis and strategy selection.

**Facades** (5 min)
   Lazy-loading wrappers for breaking circular dependencies.

Component Architecture
----------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                         CurveFit                             │
   │  ┌─────────────────┐  ┌─────────────────────────────────┐  │
   │  │ DataPreprocessor│  │ OptimizationSelector            │  │
   │  │ - validate      │  │ - detect_parameter_count        │  │
   │  │ - convert       │  │ - process_bounds                │  │
   │  │ - handle_nan    │  │ - generate_initial_guess        │  │
   │  └─────────────────┘  └─────────────────────────────────┘  │
   │                                                             │
   │  ┌─────────────────┐  ┌─────────────────────────────────┐  │
   │  │ CovarianceComp. │  │ StreamingCoordinator            │  │
   │  │ - svd_compute   │  │ - memory_analysis               │  │
   │  │ - sigma_xform   │  │ - strategy_selection            │  │
   │  └─────────────────┘  └─────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────┘

Why Decomposition?
------------------

1. **Single Responsibility**: Each component does one thing well
2. **Testability**: Components can be tested in isolation
3. **Flexibility**: Swap components without affecting others
4. **Code Size**: Each component is <350 lines (maintainable)
5. **Feature Flags**: Gradual rollout via environment variables

Quick Examples
--------------

.. code-block:: python

   from nlsq.core.orchestration import (
       DataPreprocessor,
       OptimizationSelector,
       CovarianceComputer,
       StreamingCoordinator,
   )

   # Use components directly
   preprocessor = DataPreprocessor()
   preprocessed = preprocessor.preprocess(f=model, xdata=x, ydata=y)

   selector = OptimizationSelector()
   config = selector.select(f=model, xdata=x, ydata=y, p0=[1, 0.5])

   coordinator = StreamingCoordinator()
   decision = coordinator.decide(xdata=x, ydata=y, n_params=2)

   computer = CovarianceComputer()
   cov_result = computer.compute(result=opt_result, n_data=len(y))
