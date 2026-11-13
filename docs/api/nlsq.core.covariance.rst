nlsq.core.covariance module
===========================

Parameter covariance shared by the fitting paths: ``pcov_from_jacobian`` (SVD of the
Jacobian, used by ``curve_fit``) and ``pcov_from_normal`` (eigen-decomposition of
``J^T J``, used by ``exact_streaming``). Their rank cutoffs differ on purpose and are
documented in the module docstring. ``pcov_from_normal`` Jacobi-equilibrates ``J^T J`` to a
unit diagonal before its cutoff when the diagonal spans more than 1e8 and the equilibrated
matrix is full rank, so parameter units do not decide which directions are dropped; a
rank-deficient matrix keeps the unscaled pseudo-inverse.

.. automodule:: nlsq.core.covariance
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:
