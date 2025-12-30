.. NLSQ documentation master file

NLSQ: GPU/TPU-Accelerated Curve Fitting
=======================================

**Fast, production-ready nonlinear least squares for scientific computing**

NLSQ is a JAX-powered library that brings GPU/TPU acceleration to curve fitting.
It provides a drop-in replacement for SciPy's ``curve_fit`` with 150-270x speedups
on modern hardware.

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5, 0.0])

----

Documentation
-------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      **Learn NLSQ step by step**

      Start here if you're new. Six progressive tutorials that teach you
      curve fitting from basics to GPU acceleration.

      +++
      :doc:`Start Learning → <tutorials/index>`

   .. grid-item-card:: How-To Guides
      :link: howto/index
      :link-type: doc

      **Solve specific problems**

      Practical recipes for common tasks: migrating from SciPy, handling
      large datasets, troubleshooting bad fits.

      +++
      :doc:`Find Solutions → <howto/index>`

   .. grid-item-card:: Explanation
      :link: explanation/index
      :link-type: doc

      **Understand concepts**

      Learn how curve fitting works, why JAX enables GPU acceleration,
      and what makes the Trust Region algorithm robust.

      +++
      :doc:`Explore Concepts → <explanation/index>`

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      **Look up details**

      Complete API documentation, configuration options, CLI commands,
      and built-in model functions.

      +++
      :doc:`View Reference → <reference/index>`

----

Quick Start
-----------

**Install**:

.. code-block:: bash

   pip install nlsq

**Fit some data**:

.. code-block:: python

   import numpy as np
   from nlsq import curve_fit
   import jax.numpy as jnp


   # Define your model (use jax.numpy for GPU acceleration)
   def exponential(x, a, tau, c):
       return a * jnp.exp(-x / tau) + c


   # Generate example data
   x = np.linspace(0, 10, 10000)
   y = 2.5 * np.exp(-x / 3.0) + 0.5 + 0.1 * np.random.randn(10000)

   # Fit! (automatically uses GPU if available)
   popt, pcov = curve_fit(exponential, x, y, p0=[1.0, 1.0, 0.0])

   print(f"amplitude = {popt[0]:.3f}")  # ~2.5
   print(f"tau = {popt[1]:.3f}")  # ~3.0
   print(f"offset = {popt[2]:.3f}")  # ~0.5

**Next step**: :doc:`tutorials/01_first_fit`

----

Why NLSQ?
---------

.. list-table::
   :widths: 30 70
   :class: borderless

   * - **GPU Acceleration**
     - 270x faster than SciPy on large datasets. Same code runs on CPU or GPU.
   * - **Drop-in API**
     - Minimal changes from ``scipy.optimize.curve_fit``. Familiar interface.
   * - **Automatic Jacobians**
     - JAX automatic differentiation computes exact derivatives.
   * - **Large Datasets**
     - Automatic chunking and streaming for datasets up to 100M+ points.
   * - **Production Ready**
     - 3280+ tests, 100% pass rate, comprehensive error handling, numerical stability.

Performance
~~~~~~~~~~~

**GPU Benchmarks** (NVIDIA Tesla V100):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Dataset Size
     - SciPy (CPU)
     - NLSQ (GPU)
     - Speedup
   * - 10,000
     - 0.18s
     - 0.04s
     - 4.5x
   * - 100,000
     - 2.1s
     - 0.09s
     - 23x
   * - 1,000,000
     - 40.5s
     - 0.15s
     - 270x

----

Interactive GUI
---------------

No code required? Use the interactive GUI:

.. code-block:: bash

   nlsq gui

Load data, select a model, click fit. See :doc:`gui/index` for details.

----

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Learn

   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Use

   howto/index
   gui/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Understand

   explanation/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   reference/index
   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   developer/index
   CHANGELOG

----

Resources
---------

- **GitHub**: https://github.com/imewei/NLSQ
- **PyPI**: https://pypi.org/project/nlsq/
- **Issues**: https://github.com/imewei/NLSQ/issues

Citation
~~~~~~~~

If you use NLSQ in your research, please cite:

   Hofer, L. R., Krstajić, M., & Smith, R. P. (2022). JAXFit: Fast Nonlinear
   Least Squares Fitting in JAX. *arXiv preprint arXiv:2208.12187*.
   https://doi.org/10.48550/arXiv.2208.12187

Acknowledgments
~~~~~~~~~~~~~~~

NLSQ is an enhanced fork of `JAXFit <https://github.com/Dipolar-Quantum-Gases/JAXFit>`_,
originally developed by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith.

Current maintainer: **Wei Chen** (Argonne National Laboratory)

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
