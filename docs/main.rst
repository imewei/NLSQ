.. figure:: images/NLSQ_logo.png

NLSQ: Nonlinear least squares curve fitting for the GPU/TPU
=============================================================

**Note:** NLSQ is forked from `JAXFit <https://github.com/Dipolar-Quantum-Gases/JAXFit>`__ with significant optimizations and improvements.

`Quickstart <#quickstart-colab-in-the-cloud>`__ \| `Install
guide <#installation>`__ \| `ArXiv
Paper <https://doi.org/10.48550/arXiv.2208.12187>`__ \| :doc:`API Docs <autodoc/modules>`

What is NLSQ?
---------------

NLSQ is a fork of JAXFit that implements SciPy's nonlinear least squares
curve fitting algorithms using
`JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`__
for GPU/TPU acceleration. This fork includes significant optimizations,
enhanced testing, and improved API design. Fit functions are written
in Python without CUDA programming. Performance improvements over
SciPy/Gpufit are documented in
`this paper <https://doi.org/10.48550/arXiv.2208.12187>`__.

NLSQ also improves on SciPy’s algorithm by taking advantage of JAX’s
in-built `automatic
differentiation <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`__
(autodiff) of Python functions. We use JAX’s autodiff to calculate the
Jacobians in the NLSQ algorithms rather than requiring the user to give
analytic partial derivatives or using numeric approximation techniques.

We’ve designed NLSQ to be a drop-in replacement for SciPy’s curve_fit
function. Below we show how to fit a linear function with some data

.. code:: python

   import numpy as np
   from nlsq import CurveFit


   def linear(x, m, b):  # fit function
       return m * x + b


   x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   y = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

   cf = CurveFit()
   popt, pcov = cf.curve_fit(linear, x, y)

NLSQ takes advantage of JAX’s just-in-time compilation (JIT) of Python
code to `XLA <https://www.tensorflow.org/xla>`__ which runs on GPU or
TPU hardware. This means the fit functions you define must be JIT
compilable. For basic fit functions this should cause no issues as we
simply replace NumPy functions with their drop-in JAX equivalents. For
example we show an exponential fit function

.. code:: python

   import jax.numpy as jnp


   def exponential(x, a, b):  # fit function
       return jnp.exp(a * x) + b

For more complex fit functions there are a few JIT function caveats (see
`Current gotchas <#current-gotchas>`__) such as avoiding control code
within the fit function (see `JAX’s sharp
edges <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__
article for a more in-depth look at JAX specific caveats).

Contents
~~~~~~~~

-  `Quickstart: Colab in the Cloud <#quickstart-colab-in-the-cloud>`__
-  `Large Dataset Support <#large-dataset-support>`__
-  `Current gotchas <#current-gotchas>`__
-  `Installation <#installation>`__
-  `Citing NLSQ <#citing-nlsq>`__
-  :doc:`Reference documentation <autodoc/modules>`.

Quickstart: Colab in the Cloud
------------------------------

The easiest way to test out NLSQ is using a Colab notebook connected
to a Google Cloud GPU. JAX comes pre-installed so you’ll be able to
start fitting right away.

Example notebooks are available in the `examples/ directory <https://github.com/imewei/NLSQ/tree/main/examples>`__ of the repository:

- **NLSQ Quickstart**: Learn the basics of fitting functions with NLSQ
- **NLSQ 2D Gaussian Demo**: Advanced example for fitting 2D images
- **Large Dataset Demo**: Fitting datasets with millions of points

You can run these notebooks on Google Colab by opening them directly from the GitHub repository.

Large Dataset Support
---------------------

NLSQ includes comprehensive support for handling very large datasets (20M+ points) that may exceed available memory. These features are designed to handle scientific and engineering datasets with millions to billions of data points.

Automatic Large Dataset Handling with curve_fit_large
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``curve_fit_large`` function provides automatic dataset size detection and intelligent chunking:

.. code:: python

   from nlsq import curve_fit_large, estimate_memory_requirements
   import jax.numpy as jnp
   import numpy as np

   # Check memory requirements for 50 million points
   stats = estimate_memory_requirements(50_000_000, n_params=3)
   print(f"Memory required: {stats.total_memory_estimate_gb:.2f} GB")
   print(f"Recommended chunks: {stats.n_chunks}")

   # Generate large dataset (50M points)
   x = np.linspace(0, 10, 50_000_000)
   y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, len(x))


   # Define fit function using JAX numpy
   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Use curve_fit_large - automatic chunking if needed
   popt, pcov = curve_fit_large(
       exponential,
       x,
       y,
       p0=[2.5, 0.6, 0.2],
       memory_limit_gb=4.0,  # Automatic chunking if needed
       show_progress=True,  # Progress bar for large datasets
   )

   print(f"Fitted parameters: {popt}")
   print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")

Advanced Options
~~~~~~~~~~~~~~~~

For more control, use the ``LargeDatasetFitter`` class or ``fit_large_dataset`` function:

.. code:: python

   from nlsq import LargeDatasetFitter, fit_large_dataset, LDMemoryConfig
   import jax.numpy as jnp

   # Option 1: Use the convenience function
   result = fit_large_dataset(
       exponential,
       x,
       y,
       p0=[2.5, 0.6, 0.2],
       memory_limit_gb=4.0,
       show_progress=True,
   )

   # Option 2: Use LargeDatasetFitter for more control
   config = LDMemoryConfig(
       memory_limit_gb=4.0,
       min_chunk_size=10000,
       max_chunk_size=1000000,
       enable_sampling=True,  # For datasets > 100M points
   )

   fitter = LargeDatasetFitter(config=config)
   result = fitter.fit_with_progress(
       exponential,
       x,
       y,
       p0=[2.5, 0.6, 0.2],
   )

Sparse Jacobian Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For problems with sparse Jacobian structure:

.. code:: python

   from nlsq import SparseJacobianComputer, SparseOptimizer

   # Detect sparsity pattern
   sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)
   sparsity_pattern = sparse_computer.detect_sparsity(func, x_sample, p0)

   # Optimize with sparse methods if beneficial
   if sparse_computer.is_sparse(sparsity_pattern):
       optimizer = SparseOptimizer()
       result = optimizer.optimize_with_sparsity(func, x, y, p0, sparsity_pattern)

Streaming Optimizer
~~~~~~~~~~~~~~~~~~~

For datasets that don't fit in memory or are generated on-the-fly:

.. code:: python

   from nlsq import StreamingOptimizer, StreamingConfig
   from nlsq import create_hdf5_dataset

   # Create or load HDF5 dataset
   create_hdf5_dataset(
       "large_data.h5", func, params, n_samples=100_000_000, chunk_size=10000
   )

   # Configure streaming
   config = StreamingConfig(batch_size=10000, max_epochs=100, convergence_tol=1e-6)

   optimizer = StreamingOptimizer(config)
   result = optimizer.fit_from_hdf5("large_data.h5", func, p0)

Memory-Efficient Solvers
~~~~~~~~~~~~~~~~~~~~~~~~

NLSQ includes iterative solvers that reduce memory usage:

.. code:: python

   from nlsq import CurveFit

   cf = CurveFit()

   # Use conjugate gradient solver (memory efficient)
   popt, pcov = cf.curve_fit(func, x, y, p0, solver="cg")  # Or 'lsqr' for sparse problems

Key Features:

- **Automatic Size Detection**: ``curve_fit_large`` automatically switches between standard and chunked fitting
- **Intelligent Chunking**: Improved algorithm with <1% error for well-conditioned problems
- **JAX Tracing Support**: Compatible with functions having 15+ parameters
- **Memory Estimation**: Predict memory requirements before fitting
- **Progress Reporting**: Real-time progress for long-running fits
- **Sparse Optimization**: Exploits sparsity in Jacobian matrices
- **Streaming Support**: Process data that doesn't fit in memory
- **HDF5 Integration**: Work with datasets stored on disk
- **Iterative Solvers**: CG and LSQR solvers for reduced memory footprint
- **Adaptive Convergence**: Early stopping when parameters stabilize

For detailed information, see the :doc:`Large Dataset Guide <large_dataset_guide>` and :doc:`API documentation <autodoc/modules>`.

Current gotchas
---------------

Full disclosure we’ve copied most of this from the `JAX
repo <https://github.com/google/jax#current-gotchas>`__, but NLSQ
inherits JAX’s idiosyncrasies and so the “gotchas” are mostly the same.

Double precision required
~~~~~~~~~~~~~~~~~~~~~~~~~

NLSQ requires double precision (64-bit, ``float64``) for numerical stability.
By default, JAX uses single precision (32-bit, ``float32``).

NLSQ **automatically enables double precision** when imported. However, if you
import JAX before NLSQ, you must enable it manually:

.. code:: python

   # If importing JAX first (not recommended)
   from jax import config

   config.update("jax_enable_x64", True)

   import jax.numpy as jnp
   from nlsq import CurveFit

   # Recommended: Import NLSQ first (auto-enables double precision)
   from nlsq import CurveFit
   import jax.numpy as jnp

Other caveats
~~~~~~~~~~~~~

Below are some more things to be careful of, but a full list can be
found in `JAX’s Gotchas
Notebook <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.
Some standouts:

1. JAX transformations only work on `pure
   functions <https://en.wikipedia.org/wiki/Pure_function>`__, which
   don’t have side-effects and respect `referential
   transparency <https://en.wikipedia.org/wiki/Referential_transparency>`__
   (i.e. object identity testing with ``is`` isn’t preserved). If you
   use a JAX transformation on an impure Python function, you might see
   an error like ``Exception: Can't lift Traced...`` or
   ``Exception: Different traces at same level``.
2. `In-place mutating updates of
   arrays <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates>`__,
   like ``x[i] += y``, aren’t supported, but `there are functional
   alternatives <https://jax.readthedocs.io/en/latest/jax.ops.html>`__.
   Under a ``jit``, those functional alternatives will reuse buffers
   in-place automatically.
3. Some transformations, like ``jit``, `constrain how you can use Python
   control
   flow <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow>`__.
   You’ll always get loud errors if something goes wrong. You might have
   to use `jit’s static_argnums
   parameter <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__,
   `structured control flow
   primitives <https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators>`__
   like
   `lax.scan <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan>`__.
4. Some of NumPy’s dtype promotion semantics involving a mix of Python
   scalars and NumPy types aren’t preserved, namely
   ``np.add(1, np.array([2], np.float32)).dtype`` is ``float64`` rather
   than ``float32``.
5. If you’re looking for `convolution
   operators <https://jax.readthedocs.io/en/latest/notebooks/convolutions.html>`__,
   they’re in the ``jax.lax`` package.

Installation
------------

Requirements
~~~~~~~~~~~~

NLSQ has been tested with the following versions:

- **Python**: 3.12 or higher (3.13 also supported)
- **JAX**: 0.4.20 to 0.7.2
- **NumPy**: 1.26.0 or higher
- **SciPy**: 1.11.0 or higher
- **Operating Systems**: Linux (recommended), macOS, Windows (via WSL2 or native)
- **Hardware**: CPU, NVIDIA GPU (CUDA 12+), Google TPU

Quick Install
~~~~~~~~~~~~~

**Linux/macOS (Recommended):**

::

   # For CPU-only
   pip install --upgrade "jax[cpu]>=0.4.20" nlsq

   # For GPU with CUDA 12
   pip install --upgrade "jax[cuda12]>=0.4.20" nlsq

**Development Installation:**

::

   git clone https://github.com/imewei/NLSQ.git
   cd nlsq
   pip install -e ".[dev,test,docs]"

Windows JAX install
~~~~~~~~~~~~~~~~~~~

If you are installing JAX on a Windows machine with a CUDA compatible
GPU then you’ll need to read the first part. If you’re only installing
the CPU version

Installing CUDA Toolkit
^^^^^^^^^^^^^^^^^^^^^^^

If you’ll be running JAX on a CUDA compatible GPU you’ll need a CUDA
toolkit and CUDnn. We recommend using an Anaconda environment to do all
this installation.

First make sure your GPU driver is CUDA compatible and that the latest
NVIDIA driver has been installed.

To create a Conda environment with Python 3.12 open up Anaconda Prompt
and do the following:

::

   conda create -n nlsq python=3.12

Now activate the environment

::

   conda activate nlsq

For CUDA 12 support, install the toolkit:

::

   conda install -c conda-forge cuda-toolkit=12.1

Installing JAX and NLSQ
^^^^^^^^^^^^^^^^^^^^^^^

Install JAX with CUDA support using the standard pip packages:

::

   # For CPU-only
   pip install "jax[cpu]>=0.4.20"

   # For GPU with CUDA 12
   pip install "jax[cuda12_local]>=0.4.20"

   # Then install NLSQ
   pip install nlsq

For the latest JAX installation instructions, see the `official JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`__.

.. raw:: html

   <!--For more detail on using these pre-built wheels please see the docs.-->

Citing NLSQ
-------------

If you use NLSQ consider citing the `introductory
paper <https://doi.org/10.48550/arXiv.2208.12187>`__:

::

   @article{NLSQ,
     title={NLSQ: Trust Region Method for Nonlinear Least-Squares Curve Fitting on the {GPU}},
     author={Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
     journal={arXiv preprint arXiv:2208.12187},
     year={2022}
     url={https://doi.org/10.48550/arXiv.2208.12187}
   }

API Documentation
-----------------------

For details about the NLSQ API, see the :doc:`reference documentation <autodoc/modules>`.
