StreamingCoordinator
====================

.. versionadded:: 0.6.4

The ``StreamingCoordinator`` analyzes memory requirements and selects the
optimal processing strategy.

Basic Usage
-----------

.. code-block:: python

   from nlsq.core.orchestration import StreamingCoordinator

   coordinator = StreamingCoordinator()
   decision = coordinator.decide(
       xdata=x, ydata=y, n_params=3, workflow="auto", memory_limit_mb=None
   )

   # Access decision
   strategy = decision.strategy  # 'standard', 'chunked', 'streaming'
   memory_pressure = decision.memory_pressure
   chunk_size = decision.chunk_size

StreamingDecision
-----------------

The ``decide()`` method returns a ``StreamingDecision`` object:

.. code-block:: python

   @dataclass
   class StreamingDecision:
       strategy: str  # 'standard', 'chunked', 'streaming'
       memory_pressure: float  # Estimated memory usage ratio
       chunk_size: int | None  # Chunk size for chunked/streaming
       config: dict  # Strategy-specific configuration
       reason: str  # Human-readable explanation

Memory Analysis
---------------

The coordinator estimates memory requirements:

.. code-block:: python

   memory_gb = coordinator.estimate_memory(n_data=1000000, n_params=5)
   print(f"Estimated peak memory: {memory_gb:.2f} GB")

.. code-block:: text

   Memory Estimation:
   - data_gb = n_points × 2 × 8 bytes (x + y)
   - jacobian_gb = n_points × n_params × 8 bytes
   - peak_gb = data_gb + 1.3 × jacobian_gb + solver_overhead

Strategy Selection
------------------

**STANDARD (default):**

.. code-block:: python

   # All data and Jacobian fit in memory
   # Most efficient option

   decision = coordinator.decide(xdata=x_small, ydata=y_small, n_params=3)
   assert decision.strategy == "standard"

**CHUNKED:**

.. code-block:: python

   # Data fits, but full Jacobian exceeds memory
   # Jacobian computed in chunks

   decision = coordinator.decide(xdata=x_medium, ydata=y_medium, n_params=50)
   if decision.strategy == "chunked":
       print(f"Chunk size: {decision.chunk_size}")

**STREAMING:**

.. code-block:: python

   # Data itself exceeds memory
   # Adaptive batch processing

   decision = coordinator.decide(xdata=x_large, ydata=y_large, n_params=3)
   if decision.strategy == "streaming":
       print(f"Memory pressure: {decision.memory_pressure:.1%}")

Memory Limit Override
---------------------

Force a specific memory limit:

.. code-block:: python

   # Pretend only 4GB available
   decision = coordinator.decide(xdata=x, ydata=y, n_params=3, memory_limit_mb=4000)

   # Forces chunked/streaming for smaller datasets

Workflow Integration
--------------------

Different workflows have different defaults:

.. code-block:: python

   # auto: memory-aware selection
   decision = coordinator.decide(..., workflow="auto")

   # auto_global: same + global optimization support
   decision = coordinator.decide(..., workflow="auto_global")

   # hpc: optimized for cluster environments
   decision = coordinator.decide(..., workflow="hpc")

Decision Reasons
----------------

.. code-block:: python

   decision = coordinator.decide(xdata=x, ydata=y, n_params=5)
   print(f"Strategy: {decision.strategy}")
   print(f"Reason: {decision.reason}")

Example reasons:

- "Data and Jacobian fit in memory"
- "Jacobian exceeds 75% of available memory"
- "Data exceeds available memory"

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from nlsq.core.orchestration import StreamingCoordinator

   coordinator = StreamingCoordinator()

   # Test different data sizes
   sizes = [1_000, 100_000, 1_000_000, 10_000_000]

   for n in sizes:
       x = np.linspace(0, 10, n)
       y = np.random.randn(n)

       decision = coordinator.decide(xdata=x, ydata=y, n_params=5, workflow="auto")

       memory_gb = coordinator.estimate_memory(n, 5)

       print(
           f"Points: {n:>10,} | Strategy: {decision.strategy:10} | "
           f"Memory: {memory_gb:.2f} GB | Pressure: {decision.memory_pressure:.1%}"
       )

Example output:

.. code-block:: text

   Points:      1,000 | Strategy: standard   | Memory: 0.00 GB | Pressure: 0.0%
   Points:    100,000 | Strategy: standard   | Memory: 0.04 GB | Pressure: 0.5%
   Points:  1,000,000 | Strategy: chunked    | Memory: 0.40 GB | Pressure: 5.0%
   Points: 10,000,000 | Strategy: streaming  | Memory: 4.00 GB | Pressure: 50.0%

Next Steps
----------

- :doc:`facades` - Lazy-loading wrappers
- :doc:`../custom_workflows/index` - Building custom pipelines
