Concepts & Explanations
=======================

Understand how NLSQ works and why it's designed the way it is.
These guides explain the theory, architecture, and design decisions behind NLSQ.

Fundamentals
------------

.. toctree::
   :maxdepth: 1

   how_fitting_works
   trust_region
   jax_autodiff

Numerical Stability
-------------------

.. toctree::
   :maxdepth: 1

   numerical_stability
   regularization

Advanced Topics
---------------

.. toctree::
   :maxdepth: 1

   streaming
   gpu_architecture
   workflows

Overview
--------

**How Curve Fitting Works**
    :doc:`how_fitting_works` explains the mathematical foundation of
    nonlinear least squares optimization - what it means to "fit" a model
    to data and how the algorithm finds optimal parameters.

**Trust Region Reflective Algorithm**
    :doc:`trust_region` provides a deep dive into the TRF algorithm that
    NLSQ uses for optimization, including how it handles bounds and
    ensures convergence.

**JAX and Automatic Differentiation**
    :doc:`jax_autodiff` explains how NLSQ uses JAX for GPU acceleration
    and automatic Jacobian computation, and why this is faster than
    finite differences.

**Numerical Stability**
    :doc:`numerical_stability` covers the 4-layer defense strategy that
    prevents divergence and ensures robust optimization even with
    challenging data.

**Streaming Optimization**
    :doc:`streaming` explains how NLSQ handles datasets too large to fit
    in memory using streaming optimization techniques.

**GPU Architecture**
    :doc:`gpu_architecture` describes how NLSQ leverages GPU hardware
    for massive speedups and when GPU acceleration is most beneficial.

See Also
--------

- :doc:`/tutorials/index` - Learn by doing
- :doc:`/howto/index` - Solve specific problems
- :doc:`/reference/index` - API reference
