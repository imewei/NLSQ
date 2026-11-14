Performance Benchmarks
======================

This page records measured fit times for NLSQ and four other libraries, and states what those
measurements do and do not show. All numbers come from one run of the library-comparison harness
(``benchmarks/compare/``), recorded in
``benchmarks/baselines/2026-11-13-library-comparison-1k-100M.md``.

.. important::

   **Every comparison on this page is made on one device.** CPU times are compared with CPU times and
   GPU times with GPU times. SciPy and lmfit have no GPU path, so they appear only in the CPU comparison.
   The only CPU-versus-GPU comparison is :ref:`NLSQ on CPU against NLSQ on GPU <bench-device-scaling>`,
   which measures what the GPU adds to the same code, not how NLSQ compares with another library.

How to read the tables
----------------------

- Times are the warm median of end-to-end fits (data transfer included, first-call compilation excluded).
- **Bold** marks the fastest entry in a row.
- ``—`` means the combination was not run: ``curve_fit`` was not run at 10M points and above,
  ``curve_fit_large`` was not run below 1M points, and JAXopt and Optimistix cannot fit the bounded problem.
- ``fails`` means the run did not complete (Optimistix on GPU at 100M points).
- ``curve_fit`` is the in-memory solver. ``curve_fit_large`` runs exact full-data streaming at or above its
  ``size_threshold`` (1M points). Only ``curve_fit_large`` was run at 10M points and above. At 1M points both
  are shown.

Summary
-------

Ratios below are *other library time / NLSQ time*, taken over the four models (three for JAXopt and
Optimistix), using ``curve_fit`` up to 100K points and ``curve_fit_large`` from 1M points. A ratio above 1
means NLSQ was faster.

**CPU against CPU**

- **SciPy with** ``method="trf"`` (the method the harness runs) was faster than NLSQ at 1K and 10K points (by
  1.7x to 9.7x). NLSQ was faster at 100K points (1.7x to 3.6x) and from 1M points up (2.4x to 12.6x).
- **SciPy with its default** ``method="lm"`` (MINPACK), measured only in a :ref:`separate spot check <bench-scipy-lm>`
  on the exponential decay, was faster than NLSQ ``curve_fit`` at every size from 1K to 1M points (by 12x at 1K,
  3.0x at 100K and 1.8x at 1M). NLSQ ``curve_fit_large`` was faster than SciPy ``lm`` at 1M points (2.3x) and at 10M
  points (4.2x).
- **lmfit** was faster than NLSQ in 10 of the 12 cases from 1K to 100K points (by 1.4x to 5.9x). NLSQ was faster
  in all 12 cases from 1M points up (1.4x to 10.8x).
- **JAXopt** was faster than NLSQ from 1K to 100K points (by 1.8x to 10x). From 1M points up the two were
  within about 1.5x of each other (JAXopt up to 1.2x faster, NLSQ up to 1.5x faster).
- **Optimistix** was faster than NLSQ in all 9 cases from 1K to 100K points (by 1.1x to 8.3x). NLSQ was faster
  from 1M points up (1.8x to 5.7x).

**GPU against GPU** (JAXopt and Optimistix only)

- **JAXopt** was faster than NLSQ from 1K to 100K points (by 2.2x to 12.5x) and at 1M points (by 1.03x to 1.6x).
  At 10M points the two were level to 1.5x in NLSQ's favour. At 100M points NLSQ was faster in all three
  models (1.3x to 1.9x).
- **Optimistix** was faster than NLSQ at 1K and 10K points (by 1.1x to 8.3x). NLSQ was faster at 1M and 10M
  points (3.4x to 14.7x). Optimistix failed on GPU at 100M points in all three models.

**NLSQ on CPU against NLSQ on GPU**

- The GPU was slower than the CPU at 1K and 10K points (by 1.4x to 7.1x) and between 1.2x slower and 3.2x
  faster at 100K points.
- At 1M points ``curve_fit_large`` ran 4.9x to 6.8x faster on the GPU; at 10M and 100M points, 9.8x to 20.7x.

What the data do **not** support: a general claim that NLSQ is faster than other libraries, or that a GPU makes
small fits faster. On the CPU, NLSQ was faster than SciPy ``trf`` and lmfit from about 100K to 1M points, but
against SciPy's default ``lm`` only ``curve_fit_large`` at 1M points and above was faster. JAXopt was
faster than or level with NLSQ in most cases below 100M points (NLSQ led in 4 of the 12 cases from 1M to 10M
points and in all 100M-point cases). NLSQ's practical advantages are elsewhere: the SciPy-compatible ``curve_fit`` interface, bounds, covariance estimates, and fits to
datasets too large to hold in memory.

Methodology
-----------

**Hardware and software.** Intel Core i9-13900H (20 logical cores, 62 GB RAM) and NVIDIA RTX 4090 Laptop GPU
(16 GB), Python 3.13, JAX 0.12.0.1, NumPy 2.5, SciPy 1.18 (versions installed in the project environment when this page
was written; the baseline file does not record them). CPU runs use
``JAX_PLATFORMS=cpu``. The CPU governor was ``powersave``. Commit ``343af740``.

**Problems.** Four models, each fitted to ``n`` points with ``x`` evenly spaced, Gaussian noise of standard
deviation 0.05 and seed 0:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Function
     - Start point
     - Bounds
   * - ``exp_decay``
     - ``a*exp(-b*x) + c`` on ``x`` in [0, 4]
     - (1.0, 1.0, 0.0)
     - none
   * - ``gaussian``
     - ``a*exp(-(x-mu)**2 / (2*s**2))`` on ``x`` in [-3, 5]
     - (1.0, 0.0, 1.0)
     - none
   * - ``two_exp``
     - ``a1*exp(-k1*x) + a2*exp(-k2*x)`` on ``x`` in [0, 5]
     - (1.0, 0.5, 1.0, 2.0)
     - none
   * - ``bounded_exp``
     - same as ``exp_decay``
     - (1.0, 1.0, 0.0)
     - ``a`` ≤ 2.0 (the unconstrained optimum is 2.5, so the bound is active), 0 ≤ ``b`` ≤ 3, 0 ≤ ``c`` ≤ 1

**Protocol.**

- Each case runs in a fresh process. The first (cold) call is discarded, and the median of the following
  warm calls is reported: 7 repeats from 1K to 1M points, 3 at 10M, and at 100M points 3 on GPU, 2 for the
  JAX-based libraries on CPU and 1 for SciPy and lmfit.
- Every library was given a tolerance of 1e-8 and at most 1000 iterations (``max_nfev`` or ``maxiter``). SciPy,
  lmfit and NLSQ received ``ftol``, ``xtol`` and ``gtol``. JAXopt received ``xtol`` and ``gtol`` (its ``tol`` is
  unused with the Madsen-Nielsen stopping criterion that the adapter selects), and Optimistix received ``rtol`` and
  ``atol`` only, because it has no separate ``gtol``. The stopping rules therefore differ between libraries.
- **Solvers.** NLSQ ``curve_fit``: trust-region reflective (TRF). NLSQ ``curve_fit_large``: exact full-data
  Levenberg-Marquardt, with bounds handled by an active set. SciPy: ``curve_fit(method="trf")``, chosen to match
  NLSQ's algorithm. SciPy's default for unbounded problems, ``method="lm"`` (MINPACK), is faster than ``trf`` and is
  measured only in the :ref:`spot check <bench-scipy-lm>` below. lmfit: ``leastsq`` (MINPACK Levenberg-Marquardt) for the three unbounded problems and
  ``least_squares`` (TRF) for the bounded one. JAXopt and Optimistix: their ``LevenbergMarquardt`` solvers, jitted
  as a single program.
- The final cost is recomputed from the returned parameters and compared with the best cost found by any
  library (relative tolerance 1e-6). The largest excess cost for NLSQ in this run was 7e-9.
- NLSQ was run through ``curve_fit`` and ``curve_fit_large`` (default exact streaming). Both return a
  covariance matrix.

Comparison on CPU
-----------------

NLSQ (both APIs), SciPy, lmfit, JAXopt and Optimistix, all on the CPU.

Exponential decay (3 parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - SciPy (``trf``)
     - lmfit
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 1.1 ms
     - 4.6 ms
     - 4.0 ms
     - —
     - **0.4 ms**
     - 0.5 ms
   * - 10,000
     - 3.9 ms
     - 3.8 ms
     - 11.8 ms
     - —
     - **1.4 ms**
     - 3.8 ms
   * - 100,000
     - 111.2 ms
     - 24.5 ms
     - 44.4 ms
     - —
     - **8.6 ms**
     - 19.3 ms
   * - 1,000,000
     - 1.14 s
     - 464.0 ms
     - 362.3 ms
     - 90.8 ms
     - **82.8 ms**
     - 159.7 ms
   * - 10,000,000
     - 5.26 s
     - 4.45 s
     - —
     - 977.2 ms
     - **945.4 ms**
     - 2.25 s
   * - 100,000,000
     - 65.94 s
     - 53.46 s
     - —
     - **6.08 s**
     - 7.70 s
     - 12.91 s

Gaussian (3 parameters)
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - SciPy (``trf``)
     - lmfit
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 1.7 ms
     - 2.4 ms
     - 5.3 ms
     - —
     - **0.9 ms**
     - 1.2 ms
   * - 10,000
     - 6.3 ms
     - 5.3 ms
     - 10.8 ms
     - —
     - **4.0 ms**
     - 6.3 ms
   * - 100,000
     - 164.1 ms
     - 39.1 ms
     - 64.7 ms
     - —
     - **37.2 ms**
     - 58.5 ms
   * - 1,000,000
     - 1.53 s
     - 432.4 ms
     - 562.0 ms
     - **252.5 ms**
     - 320.9 ms
     - 577.0 ms
   * - 10,000,000
     - 9.05 s
     - 5.40 s
     - —
     - 3.83 s
     - **3.15 s**
     - 7.67 s
   * - 100,000,000
     - 120.34 s
     - 54.72 s
     - —
     - **23.61 s**
     - 24.41 s
     - 59.69 s

Two exponentials (4 parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - SciPy (``trf``)
     - lmfit
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 3.1 ms
     - 3.5 ms
     - 7.0 ms
     - —
     - **1.0 ms**
     - 2.0 ms
   * - 10,000
     - 8.3 ms
     - 7.9 ms
     - 15.1 ms
     - —
     - **4.5 ms**
     - 4.8 ms
   * - 100,000
     - 312.8 ms
     - 64.4 ms
     - 87.0 ms
     - —
     - **42.3 ms**
     - 82.2 ms
   * - 1,000,000
     - 1.56 s
     - 737.0 ms
     - 930.4 ms
     - **328.2 ms**
     - 482.9 ms
     - 984.4 ms
   * - 10,000,000
     - 14.20 s
     - 9.43 s
     - —
     - 4.26 s
     - **3.74 s**
     - 11.12 s
   * - 100,000,000
     - 139.54 s
     - 107.25 s
     - —
     - **28.22 s**
     - 41.55 s
     - 159.60 s

Bounded exponential decay (3 parameters, active upper bound)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - SciPy (``trf``)
     - lmfit
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - **4.4 ms**
     - 7.2 ms
     - 42.8 ms
     - —
     - —
     - —
   * - 10,000
     - **12.1 ms**
     - 14.6 ms
     - 51.2 ms
     - —
     - —
     - —
   * - 100,000
     - 524.1 ms
     - 454.2 ms
     - **311.4 ms**
     - —
     - —
     - —
   * - 1,000,000
     - 3.21 s
     - 3.18 s
     - 1.28 s
     - **293.3 ms**
     - —
     - —
   * - 10,000,000
     - 19.43 s
     - 19.97 s
     - —
     - **3.92 s**
     - —
     - —
   * - 100,000,000
     - 170.56 s
     - 342.39 s
     - —
     - **34.09 s**
     - —
     - —


Comparison on GPU
-----------------

NLSQ, JAXopt and Optimistix on the GPU. SciPy and lmfit are CPU-only and are not part of this comparison.

Exponential decay (3 parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 24.8 ms
     - —
     - **2.0 ms**
     - 3.0 ms
   * - 10,000
     - 30.9 ms
     - —
     - 5.0 ms
     - **3.9 ms**
   * - 100,000
     - 48.0 ms
     - —
     - **4.1 ms**
     - 11.1 ms
   * - 1,000,000
     - 140.5 ms
     - 18.5 ms
     - **11.6 ms**
     - 62.6 ms
   * - 10,000,000
     - —
     - 78.1 ms
     - **77.3 ms**
     - 657.0 ms
   * - 100,000,000
     - —
     - **618.1 ms**
     - 801.0 ms
     - fails

Gaussian (3 parameters)
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 39.2 ms
     - —
     - 5.5 ms
     - **5.0 ms**
   * - 10,000
     - 24.8 ms
     - —
     - **6.1 ms**
     - 7.1 ms
   * - 100,000
     - 27.8 ms
     - —
     - **10.7 ms**
     - 29.6 ms
   * - 1,000,000
     - 165.8 ms
     - 43.5 ms
     - **36.7 ms**
     - 196.1 ms
   * - 10,000,000
     - —
     - **217.4 ms**
     - 334.9 ms
     - 2.11 s
   * - 100,000,000
     - —
     - **1.63 s**
     - 2.84 s
     - fails

Two exponentials (4 parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - 27.1 ms
     - —
     - **6.8 ms**
     - 25.6 ms
   * - 10,000
     - 20.5 ms
     - —
     - **8.4 ms**
     - 11.8 ms
   * - 100,000
     - 27.6 ms
     - —
     - **12.6 ms**
     - 52.0 ms
   * - 1,000,000
     - 115.5 ms
     - 48.4 ms
     - **47.1 ms**
     - 405.3 ms
   * - 10,000,000
     - —
     - **296.5 ms**
     - 437.9 ms
     - 4.35 s
   * - 100,000,000
     - —
     - **2.22 s**
     - 4.24 s
     - fails

Bounded exponential decay (3 parameters, active upper bound)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
     - JAXopt
     - Optimistix
   * - 1,000
     - **139.2 ms**
     - —
     - —
     - —
   * - 10,000
     - **111.6 ms**
     - —
     - —
     - —
   * - 100,000
     - **383.6 ms**
     - —
     - —
     - —
   * - 1,000,000
     - 538.6 ms
     - **49.0 ms**
     - —
     - —
   * - 10,000,000
     - —
     - **256.7 ms**
     - —
     - —
   * - 100,000,000
     - —
     - **1.65 s**
     - —
     - —


.. _bench-scipy-lm:

SciPy's default method (spot check)
-----------------------------------

The harness runs SciPy with ``method="trf"``. SciPy's default for unbounded problems is ``method="lm"``, which is
faster. This table measures all four variants back to back in one process on the CPU, for the exponential decay
only. It is a separate, smaller run than the tables above, with a load average of about 4.6, and is recorded in
``benchmarks/baselines/2026-11-14-scipy-lm-spot-check.md``. Its SciPy ``trf`` times differ from the main run
(for example 652 ms against 1.14 s at 1M points), so compare within a row only.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - SciPy ``lm`` (default)
     - SciPy ``trf``
     - NLSQ ``curve_fit``
     - NLSQ ``curve_fit_large``
   * - 1,000
     - **0.3 ms**
     - 1.0 ms
     - 3.6 ms
     - —
   * - 10,000
     - **1.7 ms**
     - 2.8 ms
     - 5.2 ms
     - —
   * - 100,000
     - **18.3 ms**
     - 67.5 ms
     - 55.1 ms
     - —
   * - 1,000,000
     - 212.7 ms
     - 651.7 ms
     - 373.9 ms
     - **93.8 ms**
   * - 10,000,000
     - 2823.4 ms
     - 5321.4 ms
     - —
     - **670.7 ms**

.. _bench-device-scaling:

NLSQ on CPU against NLSQ on GPU
-------------------------------

The same NLSQ API run on each device. "CPU time / GPU time" above 1 means the GPU was faster; below 1 means
the CPU was faster. This is a statement about NLSQ's scaling with hardware, not a library comparison.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Points
     - ``curve_fit`` CPU
     - ``curve_fit`` GPU
     - CPU time / GPU time
     - ``curve_fit_large`` CPU
     - ``curve_fit_large`` GPU
     - CPU time / GPU time
   * - Exponential decay
     - 1,000
     - 4.0 ms
     - 24.8 ms
     - 0.16x
     - —
     - —
     - —
   * - Exponential decay
     - 10,000
     - 11.8 ms
     - 30.9 ms
     - 0.38x
     - —
     - —
     - —
   * - Exponential decay
     - 100,000
     - 44.4 ms
     - 48.0 ms
     - 0.93x
     - —
     - —
     - —
   * - Exponential decay
     - 1,000,000
     - 362.3 ms
     - 140.5 ms
     - 2.6x
     - 90.8 ms
     - 18.5 ms
     - 4.9x
   * - Exponential decay
     - 10,000,000
     - —
     - —
     - —
     - 977.2 ms
     - 78.1 ms
     - 12.5x
   * - Exponential decay
     - 100,000,000
     - —
     - —
     - —
     - 6.08 s
     - 618.1 ms
     - 9.8x
   * - Gaussian
     - 1,000
     - 5.3 ms
     - 39.2 ms
     - 0.14x
     - —
     - —
     - —
   * - Gaussian
     - 10,000
     - 10.8 ms
     - 24.8 ms
     - 0.44x
     - —
     - —
     - —
   * - Gaussian
     - 100,000
     - 64.7 ms
     - 27.8 ms
     - 2.3x
     - —
     - —
     - —
   * - Gaussian
     - 1,000,000
     - 562.0 ms
     - 165.8 ms
     - 3.4x
     - 252.5 ms
     - 43.5 ms
     - 5.8x
   * - Gaussian
     - 10,000,000
     - —
     - —
     - —
     - 3.83 s
     - 217.4 ms
     - 17.6x
   * - Gaussian
     - 100,000,000
     - —
     - —
     - —
     - 23.61 s
     - 1.63 s
     - 14.5x
   * - Two exponentials
     - 1,000
     - 7.0 ms
     - 27.1 ms
     - 0.26x
     - —
     - —
     - —
   * - Two exponentials
     - 10,000
     - 15.1 ms
     - 20.5 ms
     - 0.74x
     - —
     - —
     - —
   * - Two exponentials
     - 100,000
     - 87.0 ms
     - 27.6 ms
     - 3.2x
     - —
     - —
     - —
   * - Two exponentials
     - 1,000,000
     - 930.4 ms
     - 115.5 ms
     - 8.1x
     - 328.2 ms
     - 48.4 ms
     - 6.8x
   * - Two exponentials
     - 10,000,000
     - —
     - —
     - —
     - 4.26 s
     - 296.5 ms
     - 14.4x
   * - Two exponentials
     - 100,000,000
     - —
     - —
     - —
     - 28.22 s
     - 2.22 s
     - 12.7x
   * - Bounded exponential decay
     - 1,000
     - 42.8 ms
     - 139.2 ms
     - 0.31x
     - —
     - —
     - —
   * - Bounded exponential decay
     - 10,000
     - 51.2 ms
     - 111.6 ms
     - 0.46x
     - —
     - —
     - —
   * - Bounded exponential decay
     - 100,000
     - 311.4 ms
     - 383.6 ms
     - 0.81x
     - —
     - —
     - —
   * - Bounded exponential decay
     - 1,000,000
     - 1.28 s
     - 538.6 ms
     - 2.4x
     - 293.3 ms
     - 49.0 ms
     - 6.0x
   * - Bounded exponential decay
     - 10,000,000
     - —
     - —
     - —
     - 3.92 s
     - 256.7 ms
     - 15.3x
   * - Bounded exponential decay
     - 100,000,000
     - —
     - —
     - —
     - 34.09 s
     - 1.65 s
     - 20.7x


Choosing a device and an API
----------------------------

These recommendations apply to NLSQ on the hardware above; the crossover points move with the model and the
machine.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - Recommendation
     - Measured basis
   * - Up to 10K
     - CPU, ``curve_fit``
     - The GPU run was 1.4x to 7.1x slower than the CPU run, and SciPy (both ``lm`` and ``trf``) was faster than NLSQ on CPU.
   * - About 100K
     - CPU or GPU, ``curve_fit``
     - GPU time ranged from 1.2x slower to 3.2x faster than CPU time, depending on the model.
   * - 1M and above
     - GPU, ``curve_fit_large``
     - The GPU run was 4.9x to 20.7x faster than the CPU run. At 1M points ``curve_fit_large`` was 2.2x to
       11x faster than ``curve_fit``.

Limitations
-----------

- **One machine, one run.** The numbers come from a laptop, with no confidence intervals. Treat differences
  smaller than about 25% as ties.
- **The host was not quiet.** The load average was 9.9 at the start of the run and 31 at the end, and six
  unrelated Julia processes were running afterwards. Host-driven loops such as NLSQ's are more sensitive to
  this than single-program GPU solvers. Four NLSQ ``curve_fit`` GPU cells (100K to 1M points) were 1.9x to
  2.8x slower than in the previous run (2026-11-11), while JAXopt and Optimistix on GPU were unchanged
  (median ratio 0.97 and 1.0). On CPU, the median run-to-run ratio for SciPy, lmfit, JAXopt and Optimistix was
  0.4 to 0.56, so absolute CPU times differ between runs. Compare entries within one table only.
- **Not like for like.** The libraries run different solvers (see Methodology) with different stopping rules, and
  the SciPy comparison depends on which SciPy method is chosen (see the :ref:`spot check <bench-scipy-lm>`). JAXopt and Optimistix return only the fitted parameters. They do not support bounds,
  and in these adapters they return no covariance matrix, whereas NLSQ computes one. JAXopt's repository states
  that it is no longer maintained.
- **Warm timings only.** The first fit of a new problem shape also compiles the kernel and is not part of
  these numbers. The persistent JAX cache (``~/.cache/nlsq/jax_cache``) was disabled for the benchmark workers.
- **Four smooth models with 3 to 4 parameters.** Results for many-parameter models, ill-conditioned
  problems or other noise levels may differ.

Reproducing the benchmarks
--------------------------

From the repository root, with the project environment (each command writes to its own ``--out``):

.. code-block:: bash

   R() { .venv/bin/python -m benchmarks.compare.run_library_comparison --no-batch \
         --out benchmarks/results/2026-11-13/$1 "${@:2}"; }

   # GPU
   R gpu-small    --sizes 1000 10000 100000 1000000 --libraries nlsq jaxopt optimistix --repeats 7
   R gpu-1M-large --sizes 1000000   --libraries nlsq_large --repeats 7
   R gpu-10M      --sizes 10000000  --libraries nlsq_large jaxopt optimistix --repeats 3
   R gpu-100M     --sizes 100000000 --libraries nlsq_large jaxopt optimistix --repeats 3

   # CPU
   export JAX_PLATFORMS=cpu
   R cpu-small    --sizes 1000 10000 100000 1000000 --libraries nlsq scipy lmfit jaxopt optimistix --repeats 7
   R cpu-1M-large --sizes 1000000   --libraries nlsq_large --repeats 7
   R cpu-10M      --sizes 10000000  --libraries nlsq_large scipy lmfit jaxopt optimistix --repeats 3
   R cpu-100M-jax --sizes 100000000 --libraries nlsq_large jaxopt optimistix --repeats 2
   R cpu-100M-scipy-lmfit --sizes 100000000 --libraries scipy lmfit --repeats 1

``benchmarks/README.md`` describes the harness. The pytest-based microbenchmarks
(``benchmarks/microbench/``, ``benchmarks/ci/``) and the earlier A/B baselines for individual changes are in
``benchmarks/`` and ``benchmarks/baselines/``.

See also
--------

- :doc:`../howto/optimize_performance` - tuning guide
- :doc:`../howto/handle_large_data` - large-dataset workflows
- :doc:`../explanation/gpu_architecture` - how NLSQ uses the GPU
