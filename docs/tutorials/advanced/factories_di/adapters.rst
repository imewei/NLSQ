Adapters
========

Adapters implement protocols to bridge different components.

CurveFitAdapter
---------------

Located in ``nlsq/core/adapters/curve_fit_adapter.py``:

.. code-block:: python

   from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter

   # Basic usage
   adapter = CurveFitAdapter()
   popt, pcov = adapter.curve_fit(model, x, y, p0=[...])

**With dependency injection:**

.. code-block:: python

   from nlsq.caching.unified_cache import get_global_cache
   from nlsq.stability.guard import NumericalStabilityGuard

   adapter = CurveFitAdapter(
       cache=get_global_cache(),
       stability_guard=NumericalStabilityGuard(),
       diagnostics_config=None,
   )

**Factory methods:**

.. code-block:: python

   # Standard adapter
   adapter = CurveFitAdapter()

   # With global optimization
   adapter = CurveFitAdapter.with_global_optimization()

   # With full stability
   adapter = CurveFitAdapter.with_stability()

Why Adapters?
-------------

1. **Breaking Circular Dependencies**: Adapters can import lazily
2. **Interface Compliance**: Ensure implementations match protocols
3. **Configuration**: Centralize component wiring
4. **Testing**: Easy to mock or replace

Creating Custom Adapters
------------------------

**Adapter for external optimizer:**

.. code-block:: python

   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol
   import numpy as np


   class ScipyAdapter:
       """Adapter wrapping SciPy's curve_fit."""

       def curve_fit(self, f, xdata, ydata, p0=None, sigma=None, **kwargs):
           from scipy.optimize import curve_fit as scipy_fit

           # Convert JAX function if needed
           import jax.numpy as jnp

           def numpy_f(x, *params):
               result = f(jnp.array(x), *params)
               return np.array(result)

           return scipy_fit(numpy_f, xdata, ydata, p0=p0, sigma=sigma, **kwargs)


   # Use like NLSQ
   adapter = ScipyAdapter()
   popt, pcov = adapter.curve_fit(model, x, y, p0=[...])

**Adapter with logging:**

.. code-block:: python

   import logging
   from nlsq import CurveFit
   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol


   class LoggingAdapter:
       """Adapter that logs all fitting calls."""

       def __init__(self, inner: CurveFitProtocol = None):
           self.inner = inner or CurveFit()
           self.logger = logging.getLogger(__name__)

       def curve_fit(self, f, xdata, ydata, **kwargs):
           self.logger.info(f"Starting fit: {len(xdata)} points")
           popt, pcov = self.inner.curve_fit(f, xdata, ydata, **kwargs)
           self.logger.info(f"Fit complete: popt={popt}")
           return popt, pcov

**Adapter with caching:**

.. code-block:: python

   import hashlib
   import numpy as np


   class CachingAdapter:
       """Adapter that caches fit results."""

       def __init__(self, inner: CurveFitProtocol = None):
           self.inner = inner or CurveFit()
           self.cache = {}

       def _make_key(self, xdata, ydata, p0):
           data = np.concatenate([xdata.flatten(), ydata.flatten(), p0])
           return hashlib.md5(data.tobytes()).hexdigest()

       def curve_fit(self, f, xdata, ydata, p0=None, **kwargs):
           key = self._make_key(xdata, ydata, p0)

           if key in self.cache:
               return self.cache[key]

           result = self.inner.curve_fit(f, xdata, ydata, p0=p0, **kwargs)
           self.cache[key] = result
           return result

Adapter Pattern Benefits
------------------------

.. code-block:: text

   External System          Adapter              NLSQ Core
   ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐
   │ Custom      │    │  MyAdapter      │    │ CurveFit    │
   │ Interface   │───►│  - translates   │───►│ - actual    │
   │             │    │  - wraps        │    │   fitting   │
   └─────────────┘    └─────────────────┘    └─────────────┘

- **Decoupling**: External code doesn't depend on NLSQ internals
- **Flexibility**: Swap implementations without changing code
- **Testing**: Mock adapters for unit tests

Next Steps
----------

- :doc:`dependency_injection` - Full DI patterns
- :doc:`../orchestration/index` - Component-based design
