Advanced User Guide
===================

This guide is for developers and scientists who need to leverage the full power of NLSQ: custom models, complex optimization pipelines, large-scale data processing, and integration into other software.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Core API
      :link: reference/core_api
      :link-type: doc

      Deep dive into the Python API.

   .. grid-item-card:: Performance
      :link: howto/optimize_performance
      :link-type: doc

      Tuning for maximum speed and scale.

.. toctree::
   :maxdepth: 2
   :caption: Advanced Development

   tutorials/03_fitting_with_bounds
   tutorials/04_multiple_parameters
   tutorials/05_large_datasets
   tutorials/06_gpu_acceleration
   reference/core_api
   howto/advanced_api
   reference/functions

.. toctree::
   :maxdepth: 2
   :caption: Performance & Scale

   howto/handle_large_data
   howto/optimize_performance
   howto/streaming_checkpoints
   reference/large_data

.. toctree::
   :maxdepth: 2
   :caption: Diagnosis & Internals

   howto/debug_bad_fits
   howto/troubleshooting
   explanation/index
