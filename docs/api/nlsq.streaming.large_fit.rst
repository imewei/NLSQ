nlsq.streaming.large_fit module
===============================

.. currentmodule:: nlsq.streaming

Strategy-based entry point behind :func:`nlsq.curve_fit_large`. Each strategy is a frozen
dataclass that names one way to run a large fit; pass one to
``curve_fit_large(..., strategy=...)``. There are five strategy types (``Auto``,
``InMemory``, ``Chunked``, ``Hybrid``, ``Exact``) plus the ``Multistart`` settings object,
which is nested in ``InMemory`` and ``Chunked`` and is not accepted by ``strategy=``.
``curve_fit_large`` builds the same strategy from
its legacy keywords (``method``, ``size_threshold``, ``chunk_size``, ``multistart`` and
the rest) when ``strategy`` is omitted, so both spellings run the same code.

When ``strategy`` is given, ``curve_fit_large`` raises ``TypeError`` for any legacy path
option set to a non-default value and for any extra keyword (``ftol``, ``xtol``, ``gtol``,
``loss`` and similar). Set them on the strategy instead: ``InMemory.options`` and
``Chunked.options`` take engine keywords, ``Exact`` has the ``ftol``, ``xtol``, ``gtol``,
``max_nfev`` and ``n_points`` fields, and ``Hybrid`` has ``config`` and ``verbose``. A key in
``options`` that the strategy or the fit sets itself (a strategy field, ``sigma``,
``bounds``, ``absolute_sigma``, ``check_finite`` or a multi-start keyword) raises
``TypeError``.
:func:`~nlsq.streaming.large_fit.fit_with_strategy`, called directly, does accept extra
engine keywords: for ``InMemory`` and ``Chunked`` they are merged over the strategy's
``options``, ``Exact`` merges them over its solver settings, and ``Hybrid`` accepts none.

.. code-block:: python

   from nlsq import curve_fit_large
   from nlsq.streaming import Chunked

   popt, pcov = curve_fit_large(model, x, y, p0=p0, strategy=Chunked(chunk_size=500_000))

Known limitation: the chunked path ignores ``stability``, ``rescale_data``,
``max_jacobian_elements_for_svd`` and multi-start. A chunked fit that produces no
covariance raises ``RuntimeError`` instead of returning an identity matrix.

Strategies
----------

.. autoclass:: nlsq.streaming.Auto
   :members:

.. autoclass:: nlsq.streaming.InMemory
   :members:

.. autoclass:: nlsq.streaming.Chunked
   :members:

.. autoclass:: nlsq.streaming.Hybrid
   :members:

.. autoclass:: nlsq.streaming.Exact
   :members:

Settings
--------

.. autoclass:: nlsq.streaming.Multistart
   :members:

Runner
------

.. autofunction:: nlsq.streaming.large_fit.fit_with_strategy
