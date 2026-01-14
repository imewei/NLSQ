Protocols
=========

Protocols define interface contracts that enable loose coupling and
dependency injection.

What Are Protocols?
-------------------

Python protocols (PEP 544) define structural subtyping - any class
that implements the required methods satisfies the protocol:

.. code-block:: python

   from typing import Protocol


   class MyProtocol(Protocol):
       def do_something(self, x: int) -> str: ...


   # Any class with do_something(int) -> str satisfies this

NLSQ Protocols
--------------

Located in ``nlsq/interfaces/``:

OptimizerProtocol
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.interfaces.optimizer_protocol import OptimizerProtocol


   class OptimizerProtocol(Protocol):
       def optimize(
           self, fun: Callable, x0: ArrayLike, args: tuple = (), **kwargs
       ) -> OptimizeResult: ...

CurveFitProtocol
~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol


   class CurveFitProtocol(Protocol):
       def curve_fit(
           self,
           f: Callable,
           xdata: ArrayLike,
           ydata: ArrayLike,
           p0: ArrayLike | None = None,
           sigma: ArrayLike | None = None,
           **kwargs
       ) -> tuple[ArrayLike, ArrayLike]: ...

CacheProtocol
~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.interfaces.cache_protocol import CacheProtocol


   class CacheProtocol(Protocol):
       def get(self, key: str, default: Any = None) -> Any: ...

       def set(self, key: str, value: Any) -> None: ...

       def clear(self) -> None: ...

Orchestration Protocols (v0.6.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq.interfaces.orchestration_protocol import (
       DataPreprocessorProtocol,
       OptimizationSelectorProtocol,
       CovarianceComputerProtocol,
       StreamingCoordinatorProtocol,
   )

Implementing Protocols
----------------------

**Simple implementation:**

.. code-block:: python

   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol


   class MyCurveFitter:
       """Custom implementation of CurveFitProtocol."""

       def curve_fit(self, f, xdata, ydata, p0=None, sigma=None, **kwargs):
           # Your custom fitting logic
           from scipy.optimize import curve_fit as scipy_fit

           return scipy_fit(f, xdata, ydata, p0=p0, sigma=sigma, **kwargs)


   # Type check
   fitter: CurveFitProtocol = MyCurveFitter()  # OK

**With type hints:**

.. code-block:: python

   from typing import Callable
   import numpy as np
   from numpy.typing import ArrayLike
   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol


   class TypedFitter:
       def curve_fit(
           self,
           f: Callable,
           xdata: ArrayLike,
           ydata: ArrayLike,
           p0: ArrayLike | None = None,
           sigma: ArrayLike | None = None,
           **kwargs
       ) -> tuple[np.ndarray, np.ndarray]:
           # Implementation
           pass

Protocol Usage
--------------

**Function accepting protocol:**

.. code-block:: python

   def run_analysis(
       fitter: CurveFitProtocol, data: list[tuple], model: Callable
   ) -> list[np.ndarray]:
       """Run analysis with any CurveFitProtocol implementation."""
       results = []
       for x, y in data:
           popt, pcov = fitter.curve_fit(model, x, y)
           results.append(popt)
       return results


   # Works with any compatible fitter
   from nlsq import CurveFit

   results = run_analysis(CurveFit(), data_list, my_model)

   # Also works with custom implementation
   results = run_analysis(MyCurveFitter(), data_list, my_model)

**Testing with mocks:**

.. code-block:: python

   class MockFitter:
       def curve_fit(self, f, xdata, ydata, **kwargs):
           # Return fixed values for testing
           return np.array([1.0, 2.0]), np.eye(2)


   # Use mock in tests
   results = run_analysis(MockFitter(), test_data, model)

Runtime Checking
----------------

.. code-block:: python

   from typing import runtime_checkable, Protocol


   @runtime_checkable
   class CurveFitProtocol(Protocol):
       def curve_fit(self, f, xdata, ydata, **kwargs): ...


   # Now isinstance works
   fitter = MyCurveFitter()
   assert isinstance(fitter, CurveFitProtocol)  # True

Next Steps
----------

- :doc:`adapters` - Protocol implementations
- :doc:`dependency_injection` - Using protocols with DI
