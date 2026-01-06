nlsq.core.factories
===================

Factory functions for creating and configuring NLSQ optimization components.

This module provides builder functions that simplify the creation of optimizers
and configuration objects with sensible defaults.

.. automodule:: nlsq.core.factories
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

create_optimizer
~~~~~~~~~~~~~~~~

Create an optimizer instance with the specified configuration.

.. autofunction:: nlsq.core.factories.create_optimizer
   :no-index:

configure_curve_fit
~~~~~~~~~~~~~~~~~~~

Configure curve_fit with custom settings.

.. autofunction:: nlsq.core.factories.configure_curve_fit
   :no-index:

Usage Example
-------------

.. code-block:: python

   from nlsq.core.factories import create_optimizer, configure_curve_fit

   # Create a streaming optimizer
   optimizer = create_optimizer("streaming", memory_limit_gb=16.0)

   # Configure curve_fit with defaults
   curve_fit_fn = configure_curve_fit(
       method="trf",
       max_nfev=1000,
       verbose=2,
   )
   popt, pcov = curve_fit_fn(model, xdata, ydata, p0=p0)
