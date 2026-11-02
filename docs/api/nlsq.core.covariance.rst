nlsq.core.covariance module
===========================

Parameter covariance shared by the fitting paths: ``pcov_from_jacobian`` (SVD of the
Jacobian, used by ``curve_fit``) and ``pcov_from_normal`` (eigen-decomposition of
``J^T J``, used by ``exact_streaming``). Their rank cutoffs differ on purpose and are
documented in the module docstring.

.. automodule:: nlsq.core.covariance
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:
