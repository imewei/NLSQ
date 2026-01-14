Advanced User Guide
===================

This guide is for developers and scientists who need to leverage the full power
of NLSQ: custom optimization pipelines, protocol-based design, orchestration
components, and library extension.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Advanced Tutorials
      :link: tutorials/advanced/index
      :link-type: doc

      Design custom workflows using NLSQ's API layer.

   .. grid-item-card:: Architecture
      :link: tutorials/advanced/architecture/index
      :link-type: doc

      Package structure and optimization pipeline.

API Levels
----------

NLSQ provides multiple API levels for different needs:

.. code-block:: python

   # High Level: fit() function
   from nlsq import fit

   popt, pcov = fit(model, x, y, p0=[...])

   # Mid Level: CurveFit class
   from nlsq import CurveFit

   fitter = CurveFit()
   popt, pcov = fitter.curve_fit(model, x, y, p0=[...])

   # Low Level: LeastSquares class
   from nlsq.core.least_squares import LeastSquares

   optimizer = LeastSquares()
   result = optimizer.least_squares(fun=residuals, x0=p0)

   # Orchestration: Decomposed components (v0.6.4)
   from nlsq.core.orchestration import DataPreprocessor, OptimizationSelector

   preprocessor = DataPreprocessor()
   preprocessed = preprocessor.preprocess(f=model, xdata=x, ydata=y)

.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   tutorials/advanced/index
   tutorials/advanced/architecture/index
   tutorials/advanced/core_apis/index
   tutorials/advanced/orchestration/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/core_api
   reference/orchestration
   reference/facades
   howto/advanced_api

.. toctree::
   :maxdepth: 2
   :caption: Factories & DI

   tutorials/advanced/factories_di/index

.. toctree::
   :maxdepth: 2
   :caption: Performance & Scale

   tutorials/advanced/performance/index
   howto/handle_large_data
   howto/optimize_performance
   howto/streaming_checkpoints
   reference/large_data

.. toctree::
   :maxdepth: 2
   :caption: Stability & Extension

   tutorials/advanced/stability/index
   tutorials/advanced/extension/index
   howto/debug_bad_fits
   howto/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Conceptual Deep Dives

   explanation/index
