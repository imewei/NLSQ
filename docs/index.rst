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

Choose your path:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Routine Analysis
      :link: routine_guide
      :link-type: doc

      **Fast, Standardized Fitting**

      Best for:
      - Standard data analysis
      - Using the CLI or GUI
      - Pre-defined workflows
      - Quick results

      +++
      :doc:`Routine User Guide → <routine_guide>`

   .. grid-item-card:: Advanced Development
      :link: advanced_guide
      :link-type: doc

      **Custom Pipelines & Scale**

      Best for:
      - Python API integration
      - Custom models & algorithms
      - HPC & Graphics Cards
      - Debugging & Diagnostics

      +++
      :doc:`Advanced User Guide → <advanced_guide>`

----

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Guides

   routine_guide
   advanced_guide

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Project Info

   reference/index
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
