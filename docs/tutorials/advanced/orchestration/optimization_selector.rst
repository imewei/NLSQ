OptimizationSelector
====================

.. versionadded:: 0.6.4

The ``OptimizationSelector`` handles parameter detection, bounds processing,
initial guess generation, and method selection.

Basic Usage
-----------

.. code-block:: python

   from nlsq.core.orchestration import OptimizationSelector
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   selector = OptimizationSelector()
   config = selector.select(
       f=model, xdata=x, ydata=y, p0=[1.0, 0.5, 0.0], bounds=None, method="trf"
   )

   # Access configuration
   n_params = config.n_params
   initial_guess = config.p0
   method = config.method

OptimizationConfig
------------------

The ``select()`` method returns an ``OptimizationConfig`` object:

.. code-block:: python

   @dataclass
   class OptimizationConfig:
       n_params: int  # Number of parameters
       p0: np.ndarray  # Initial parameter guess
       bounds: tuple | None  # (lower, upper) bounds
       method: str  # Optimization method
       jac: str | Callable  # Jacobian computation
       tr_solver: str  # Trust region solver
       ftol: float  # Function tolerance
       xtol: float  # Parameter tolerance
       gtol: float  # Gradient tolerance

Parameter Detection
-------------------

Automatically detect number of parameters from model signature:

.. code-block:: python

   def model(x, a, b, c):  # 3 parameters: a, b, c
       return a * jnp.exp(-b * x) + c


   n_params = selector.detect_parameter_count(model, xdata)
   print(f"Detected {n_params} parameters")  # 3

This uses Python's ``inspect`` module to analyze the function signature.

Initial Guess Generation
------------------------

If ``p0`` is not provided:

.. code-block:: python

   config = selector.select(
       f=model, xdata=x, ydata=y, p0=None  # Auto-generate initial guess
   )
   print(f"Generated p0: {config.p0}")

The selector uses heuristics based on data range and model type.

Bounds Processing
-----------------

Process user-provided bounds:

.. code-block:: python

   # Tuple format
   bounds = ([0, 0, -1], [10, 5, 1])

   config = selector.select(f=model, xdata=x, ydata=y, p0=[1, 0.5, 0], bounds=bounds)

   # Unbounded parameters
   bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

Method Selection
----------------

Select optimization method:

.. code-block:: python

   # Trust Region Reflective (default)
   config = selector.select(..., method="trf")

   # Dogbox (for small problems)
   config = selector.select(..., method="dogbox")

   # Levenberg-Marquardt (unbounded only)
   config = selector.select(..., method="lm")

Solver Selection
----------------

Select trust region solver:

.. code-block:: python

   # Exact (SVD-based, for small problems)
   config = selector.select(..., tr_solver="exact")

   # LSMR (iterative, for large problems)
   config = selector.select(..., tr_solver="lsmr")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.core.orchestration import OptimizationSelector


   def gaussian(x, amplitude, center, width, offset):
       return amplitude * jnp.exp(-0.5 * ((x - center) / width) ** 2) + offset


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 3.0 * np.exp(-0.5 * ((x - 5) / 1.2) ** 2) + 0.5

   # Configure optimization
   selector = OptimizationSelector()
   config = selector.select(
       f=gaussian,
       xdata=x,
       ydata=y,
       p0=[2.5, 5.0, 1.0, 0.5],
       bounds=([0, 0, 0.1, 0], [10, 10, 5, 2]),
       method="trf",
       ftol=1e-10,
       xtol=1e-10,
       gtol=1e-10,
   )

   print(f"Parameters: {config.n_params}")
   print(f"Initial guess: {config.p0}")
   print(f"Method: {config.method}")
   print(f"TR solver: {config.tr_solver}")
   print(f"Tolerances: ftol={config.ftol}, xtol={config.xtol}")

Next Steps
----------

- :doc:`covariance_computer` - Covariance estimation
- :doc:`streaming_coordinator` - Memory strategy
