Troubleshooting
===============

This chapter covers common issues and their solutions.

.. toctree::
   :maxdepth: 1

   common_issues
   getting_help

Quick Fixes
-----------

**Fit doesn't converge:**

- Improve initial guess ``p0``
- Add bounds to constrain parameters
- Use ``workflow='auto_global'`` for global search

**Wrong results:**

- Check model function uses ``jax.numpy``
- Verify data is correct (plot it first)
- Try different initial guess

**Memory errors:**

- Reduce ``memory_limit_gb``
- Use float32 data for large datasets
- Close other applications

**Slow performance:**

- Use GPU acceleration
- Loosen tolerances (``ftol=1e-6``)
- Start with data subset

Error Messages
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - "p0 is required"
     - Provide initial parameter guess
   * - "bounds required for auto_global"
     - Add ``bounds=([lower], [upper])``
   * - "Model must use jax.numpy"
     - Replace ``np`` with ``jnp`` in model
   * - "Covariance could not be estimated"
     - Check model/data fit, improve p0
   * - "Out of memory"
     - Reduce memory_limit_gb or data size
