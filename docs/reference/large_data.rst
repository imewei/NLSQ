Large Dataset API Reference
===========================

APIs for handling datasets that exceed GPU memory.

Overview
--------

NLSQ provides three approaches for large datasets:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - API
     - Best For
     - Memory Behavior
   * - ``curve_fit_large``
     - Datasets up to ~100M points
     - Automatic chunking, fits in memory
   * - ``LargeDatasetFitter``
     - Fine-grained control over chunking
     - Automatic memory management
   * - ``AdaptiveHybridStreamingOptimizer``
     - Huge datasets and production pipelines
     - Four-phase optimization

curve_fit_large
---------------

.. autofunction:: nlsq.curve_fit_large
   :no-index:

Automatically chunks data to fit within GPU memory.

**Key Parameters:**

- ``memory_limit_gb``: Maximum GPU memory to use
- ``chunk_size``: Override automatic chunk sizing
- ``progress``: Show progress bar during fitting

**Example:**

.. code-block:: python

   from nlsq import curve_fit_large

   # 50 million data points
   x = jnp.linspace(0, 100, 50_000_000)
   y = 2.5 * jnp.exp(-0.1 * x) + 0.5 + noise

   popt, pcov = curve_fit_large(
       model, x, y, p0=[1.0, 0.1, 0.0], memory_limit_gb=8.0  # Limit to 8 GB
   )

Exact streaming (``method="exact_streaming"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default chunked path fits each chunk separately and combines the results,
which is exact only for linear models. ``method="exact_streaming"`` (opt-in)
instead runs one Levenberg-Marquardt fit over the whole dataset: each trial
point costs one pass that accumulates ``J^T J``, ``J^T r`` and the cost in
float64 chunk by chunk, so the result does not depend on the chunk size (up to rounding). It
runs at any dataset size (``size_threshold`` is not consulted) and returns
``(popt, pcov)``.

.. code-block:: python

   from nlsq import curve_fit_large
   from nlsq.streaming import ExactSolverError

   try:
       popt, pcov = curve_fit_large(
           model, x, y, p0=[1.0, 0.1, 0.0], method="exact_streaming", ftol=1e-10
       )
   except ExactSolverError as err:
       print(err.result["termination"], err.result["nfev"])

- **Model contract:** ``model(x, *params)`` must be written with
  ``jax.numpy`` and be pointwise and deterministic in ``x`` (row ``i`` of the
  output depends only on ``x[i]``); a probe on the first chunk checks this.
- **Out-of-memory sources:** ``xdata``, ``ydata`` and ``sigma`` may be
  ``np.memmap``, ``h5py.Dataset``, ``zarr.Array`` or any array-like with
  ``shape``, ``len`` and slicing. They are validated from ``shape`` and read by
  slice, one chunk at a time and in order, on every pass: each pass reads the
  data once (plus one chunk of ``xdata`` for the model probe), and nothing is
  loaded or placed on a device whole. With the default ``check_finite=True``,
  non-finite ``xdata``/``ydata`` in any chunk is caught during the first pass.
  In-memory NumPy/JAX arrays are placed on an accelerator once only if they fit
  50% of free device memory and, with ``memory_limit_gb``, the data plus chunk
  workspace fit that budget; the chunk is sized from the budget first, so under
  a limit placement happens only for small inputs. A ``jax.Array`` already on a
  device stays there regardless of the limit.
- **Chunk factories:** pass a callable as ``xdata`` and ``None`` as ``ydata``:
  ``curve_fit_large(model, factory, None, p0=..., method="exact_streaming",
  n_points=N)``. ``factory(chunk_size)`` returns an iterator of ``(x, y)`` or
  ``(x, y, sigma)`` blocks (every block the same form). Contract: it is
  **replayable** (called once per full-data pass, each call yields the same
  data in the same deterministic order) and each block holds at most
  ``chunk_size`` points; blocks may differ in size and are padded to one
  compiled shape. ``chunk_size`` is the derived chunk (``chunk_size`` if
  given, else ``1 << 22`` reduced to fit ``memory_limit_gb``; clamped to
  ``n_points`` when that is given). ``n_points`` is optional: when omitted the
  points are counted on the first pass; every pass (the first too, when
  ``n_points`` is given) verifies the count and a mismatch raises
  ``ValueError``. Each block is validated as it arrives (1-D parts of equal,
  non-zero length, at most ``chunk_size`` points, finite on the first pass
  unless ``check_finite=False``, ``sigma`` finite and positive); the model
  probe uses the first block. ``sigma`` must then come from the blocks
  (``sigma=`` raises ``ValueError``), and ``n_points`` with an array source
  raises ``ValueError``. Exceptions raised by the factory propagate unchanged.
  With neither ``n_points`` nor ``chunk_size``, every block is padded to the
  derived chunk (default 4,194,304 rows), so pass one of them for small
  factories. The model-contract probe runs on the first block, so it should
  hold at least 2 points; fewer than 2 points in total raises ``ValueError``.
- **Accepted arguments:** ``sigma`` (1-D only), ``bounds``, ``chunk_size``,
  ``check_finite``, ``show_progress``, ``absolute_sigma``, ``memory_limit_gb``, and the keyword arguments ``ftol``,
  ``xtol``, ``gtol``, ``max_nfev`` (full-data passes, default
  ``100 * len(p0)``) and ``n_points`` (chunk factories only). Any other
  keyword argument raises ``TypeError``.
- **Weights:** a 1-D ``sigma`` divides residuals and Jacobian rows, as in
  ``curve_fit``. A ``sigma`` that is 2-D (correlated errors), of the wrong
  length, non-finite or non-positive raises ``ValueError``; finiteness and
  positivity are checked during the first pass, whatever ``check_finite`` says.
- **Bounds:** box ``bounds`` as in ``curve_fit`` (scalars or length-``p``
  arrays, ``lb < ub`` elementwise; equal, inverted or NaN bounds raise
  ``ValueError``). They are handled by an active set, not Coleman-Li TRF:
  ``p0`` is clipped into the box, a parameter on a bound whose gradient points
  outward is held, the rest take a damped step that is projected back onto the
  box. The fit is exact for the box-constrained problem and the returned point
  satisfies the KKT conditions; it differs from ``method="trf"`` in the path
  taken, not in what a solution is (on a non-convex problem the two may still
  reach different local minima). While a bound is active, stationarity is
  certified by the exact maximum decrease of the Gauss-Newton model over the
  box (a small host-side bounded least-squares solve, no extra data pass).
- **Not yet supported (``ValueError`` naming the argument):**
  ``multistart=True`` / ``global_search=True``, 2-D ``xdata``, and
  non-default ``stability``, ``rescale_data``,
  ``max_jacobian_elements_for_svd``, ``auto_size_detection`` or
  ``size_threshold``. ``nlsq.fit`` does not dispatch to this method.
- **Failure behaviour:** a fit that stalls or runs out of passes raises
  ``nlsq.streaming.ExactSolverError`` (a ``RuntimeError``) with the full
  result, including ``termination`` (``"stalled"`` or ``"max_nfev"``), as
  ``.result``.
- **Covariance caveat:** ``pcov`` follows the ``curve_fit`` contract
  (pseudo-inverse, scaled by the residual variance unless ``absolute_sigma``)
  but is computed from the normal equations ``J^T J``, which squares
  ``cond(J)``. Expect about ``1e-6`` relative agreement with ``curve_fit`` for
  ``cond(J) <= 1e4`` and ``1e-3`` for ``cond(J) <= 1e6``; a ``UserWarning`` is
  issued beyond ``1e4``.

LargeDatasetFitter
------------------

.. autoclass:: nlsq.LargeDatasetFitter
   :members:
   :special-members: __init__
   :no-index:

Class-based interface for large dataset fitting with more control.

**Example:**

.. code-block:: python

   from nlsq import LargeDatasetFitter

   fitter = LargeDatasetFitter(memory_limit_gb=8.0)

   result = fitter.fit(model, x, y, p0=p0)

Chunk sizing (``min_chunk_size``, ``max_chunk_size``, etc.) is controlled via
an ``LDMemoryConfig`` passed as ``config=``, not a ``chunk_overlap`` argument:

.. code-block:: python

   from nlsq import LargeDatasetFitter
   from nlsq.streaming.large_dataset import LDMemoryConfig

   fitter = LargeDatasetFitter(
       config=LDMemoryConfig(memory_limit_gb=8.0, max_chunk_size=500_000)
   )

AdaptiveHybridStreamingOptimizer
--------------------------------

.. autoclass:: nlsq.AdaptiveHybridStreamingOptimizer
   :members:
   :special-members: __init__
   :no-index:

Production-grade optimizer with four-phase optimization:

1. **Parameter normalization**: Scales parameters for stability
2. **L-BFGS warmup**: Fast initial convergence
3. **Streaming Gauss-Newton**: Precise refinement
4. **Exact covariance**: Accurate uncertainty estimates

**Example:**

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer
   from nlsq import HybridStreamingConfig

   config = HybridStreamingConfig.scientific_default()

   optimizer = AdaptiveHybridStreamingOptimizer(config)

   result = optimizer.fit((x, y), model, p0=p0)

HybridStreamingConfig
---------------------

.. autoclass:: nlsq.HybridStreamingConfig
   :members:
   :no-index:

Configuration for the hybrid streaming optimizer.

**Presets:**

- ``scientific_default()``: High precision for scientific computing
- ``memory_optimized()``: Reduced memory footprint
- ``aggressive()``: Fast convergence, less conservative
- ``conservative()``: Maximum reliability
- ``defense_strict()``: Strict numerical guards

**Example:**

.. code-block:: python

   from nlsq import HybridStreamingConfig

   # From preset
   config = HybridStreamingConfig.scientific_default()

   # Custom configuration
   config = HybridStreamingConfig(
       warmup_iterations=50,
       gauss_newton_max_iterations=20,
       chunk_size=50_000,
       checkpoint_frequency=10,
   )

Memory Estimation
-----------------

.. autofunction:: nlsq.estimate_memory_requirements
   :no-index:

Estimate memory requirements before fitting:

.. code-block:: python

   from nlsq import estimate_memory_requirements

   stats = estimate_memory_requirements(n_points=10_000_000, n_params=5)

   print(f"Estimated memory: {stats.total_memory_estimate_gb:.2f} GB")
   print(f"Recommended chunk size: {stats.recommended_chunk_size:,}")
   print(f"Number of chunks: {stats.n_chunks}")

Checkpointing
-------------

AdaptiveHybridStreamingOptimizer supports checkpointing for long-running fits:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

   config = HybridStreamingConfig(
       checkpoint_dir="./checkpoints",
       checkpoint_frequency=100,
   )
   optimizer = AdaptiveHybridStreamingOptimizer(config)

   # Checkpoints are written every checkpoint_frequency iterations, but
   # automatic resume from a saved checkpoint is not yet implemented --
   # see :doc:`/howto/streaming_checkpoints` for details.
   result = optimizer.fit((x, y), model, p0=p0, verbose=1)

See Also
--------

- :doc:`/tutorials/routine/data_handling/large_datasets` - Tutorial on large data handling
- :doc:`/howto/streaming_checkpoints` - Checkpoint and resume guide
- :doc:`/explanation/streaming` - Streaming concepts
