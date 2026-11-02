nlsq.streaming.exact\_lm module
================================

.. currentmodule:: nlsq.streaming.exact_lm

Exact chunk-accumulated Levenberg-Marquardt behind
``curve_fit_large(method="exact_streaming")``. Each trial point costs one pass over the
data that accumulates ``J^T J``, ``J^T r`` and the cost in float64, so solver state is
O(p\ :sup:`2`) and the result equals a full-batch fit regardless of chunking.

Use it through :func:`nlsq.curve_fit_large`; the argument rules, model contract, ``sigma``
and ``bounds`` handling, out-of-memory sources and covariance caveat are in
:ref:`the large dataset reference <exact-streaming>`.

ExactSolverError
----------------

.. autoclass:: nlsq.streaming.exact_lm.ExactSolverError
   :members:
   :show-inheritance:

Raised when the fit stalls (``termination="stalled"``) or exhausts ``max_nfev``
(``termination="max_nfev"``). The full ``OptimizeResult`` is on ``.result``:

.. code-block:: python

   from nlsq import curve_fit_large
   from nlsq.streaming import ExactSolverError

   try:
       popt, pcov = curve_fit_large(model, x, y, p0=p0, method="exact_streaming")
   except ExactSolverError as e:
       print(e, e.result["x"], e.result["nfev"])
