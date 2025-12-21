"""
Converted from nlsq_quickstart.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # NLSQ Quickstart
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ%20Quickstart.ipynb)
# ======================================================================
# ======================================================================
# ## Requirements
#
# **Important:** NLSQ requires Python 3.12 or higher. Please ensure your runtime environment meets this requirement before proceeding.
#
# ## Installing and Importing
#
# Make sure your runtime type is set to GPU rather than CPU for optimal performance. Then install NLSQ with pip:
# ======================================================================
# !pip install nlsq  # Uncomment to install in notebook environment
# ======================================================================
# Import NLSQ before importing JAX since we need NLSQ to set all the JAX computation to use 64 rather than 32 bit arrays.
# ======================================================================
import os
import sys

# Check Python version
from pathlib import Path

# Import NLSQ before importing JAX since NLSQ configures JAX to use 64-bit precision
import jax.numpy as jnp

from nlsq import CurveFit, __version__

print(f"NLSQ version: {__version__}")


# Let's also import some of the new advanced features
from nlsq import (
    MemoryConfig,
    estimate_memory_requirements,
    get_memory_config,
    memory_context,
    set_memory_limits,
)

# ======================================================================
# Now let's define a linear function using jax.numpy. You can construct function just like numpy with a few small caveats (see [current gotchas](https://github.com/Dipolar-Quantum-Gases/nlsq#current-gotchas)).
# ======================================================================


def linear(x, m, b):
    return m * x + b


# ======================================================================
# Using the function we just created, we'll simulate some synthetic fit data and show what it looks like.
# ======================================================================


# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib


import matplotlib.pyplot as plt
import numpy as np

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


# make the synthetic data
length = cap_samples(1000)
x = np.linspace(0, 10, length)
params = (3, 5)
y = linear(x, *params)
# add a little noise to the data to make things interesting
y += np.random.normal(0, 0.2, size=length)

plt.figure()
plt.title("Some Noisy Data")
plt.plot(x, y)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_quickstart"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# Now let's use NLSQ to fit this data
# ======================================================================


# ======================================================================
# ## Memory Management and Configuration
#
# NLSQ now includes sophisticated memory management features that help you optimize performance and handle large datasets more efficiently.
# ======================================================================


# Check current memory configuration
current_config = get_memory_config()
print(f"Current memory limit: {current_config.memory_limit_gb} GB")
print(f"Mixed precision fallback: {current_config.enable_mixed_precision_fallback}")

# Estimate memory requirements for our dataset
n_points = len(x)
n_params = 2  # m and b for linear function
memory_stats = estimate_memory_requirements(n_points, n_params)
print(f"\nMemory estimate for {n_points} points, {n_params} parameters:")
print(f"  Total memory needed: {memory_stats.total_memory_estimate_gb:.4f} GB")
print(f"  Recommended chunk size: {memory_stats.recommended_chunk_size}")
print(f"  Number of chunks needed: {memory_stats.n_chunks}")


# You can temporarily change memory settings using context managers
print("Default memory limit:", get_memory_config().memory_limit_gb, "GB")

# Use a temporary memory configuration
temp_config = MemoryConfig(memory_limit_gb=4.0, enable_mixed_precision_fallback=True)
with memory_context(temp_config):
    print("Inside context memory limit:", get_memory_config().memory_limit_gb, "GB")

print("After context memory limit:", get_memory_config().memory_limit_gb, "GB")

# Or you can set global memory limits
set_memory_limits(memory_limit_gb=6.0, safety_factor=0.9)
print("New global memory limit:", get_memory_config().memory_limit_gb, "GB")


# Practical example: Automatically configure memory for large datasets
print("=== Automatic Memory Management Example ===")

# Simulate a larger dataset scenario
large_n_points = cap_samples(50000)
large_n_params = 5

# Get memory requirements
large_stats = estimate_memory_requirements(large_n_points, large_n_params)
print(f"For {large_n_points} points with {large_n_params} parameters:")
print(f"  Estimated memory: {large_stats.total_memory_estimate_gb:.3f} GB")

# Automatically set appropriate memory limits based on the estimation
recommended_limit = max(4.0, large_stats.total_memory_estimate_gb * 1.5)
set_memory_limits(memory_limit_gb=recommended_limit)
print(f"  Set memory limit to: {get_memory_config().memory_limit_gb} GB")

# Now you can safely work with larger datasets
print("✓ Memory management configured for large dataset processing")


jcf = CurveFit()
popt, pcov = jcf.curve_fit(linear, x, y, p0=(1, 1))
y_fit = linear(x, *popt)

print("Actual Parameters", params)
print("Fit Parameters", popt)


# ======================================================================
# Now we'll take a look at NLSQ's speed. We do the same fit as above with $3\times 10^5$ data points for twenty different sets of data and plot the speed for each of these fits.
# ======================================================================


import time

from scipy.optimize import curve_fit


def get_random_parameters(mmin=1, mmax=10, bmin=0, bmax=10):
    deltam = mmax - mmin
    deltab = bmax - bmin
    m = mmin + deltam * np.random.random()
    b = bmin + deltab * np.random.random()
    return m, b


if QUICK:
    print("⏩ Quick mode: skipping extended speed benchmark.")
    sys.exit(0)

length = cap_samples(3 * 10**5)
x = np.linspace(0, 10, length)

jcf = CurveFit()
nlsq_fit_times = []
scipy_fit_times = []
nsamples = 3 if QUICK else 21
for i in range(nsamples):
    params = get_random_parameters()
    y = linear(x, *params) + np.random.normal(0, 0.2, size=length)

    # fit the data
    start_time = time.time()
    popt1, pcov1 = jcf.curve_fit(linear, x, y, p0=(1, 1))
    nlsq_fit_times.append(time.time() - start_time)

plt.figure()
plt.title("Fit Speeds")
plt.plot(nlsq_fit_times, label="NLSQ")
plt.xlabel("Fit Iteration")
plt.ylabel("Fit Time (seconds)")


# ======================================================================
# As you can see, the first fit is quite slow as JAX is tracing all the functions in the NLSQ CurveFit object behind the scenes. However, after it has traced them once then it runs extremely quickly.
# ======================================================================


# ======================================================================
# ## Varying Fit Data Array Size
#
# What happens if we change the size of the data for each of these random fits though. Here we increase the data size from $10^3$ to $10^6$ and look at the fit speed.
# ======================================================================


def get_coordinates(length, xmin=0, xmax=10):
    return np.linspace(xmin, xmax, length)


def get_random_data(length):
    xdata = get_coordinates(length)
    params = get_random_parameters()
    ydata = linear(xdata, *params) + np.random.normal(0, 0.2, size=length)
    return xdata, ydata


lmin = min(10**3, MAX_SAMPLES)
lmax = cap_samples(10**6)
nlengths = 20
lengths = np.linspace(lmin, lmax, nlengths, dtype=int)

jcf = CurveFit()
nlsq_fit_times = []
for length in lengths:
    xdata, ydata = get_random_data(length)

    start_time = time.time()
    popt1, pcov1 = jcf.curve_fit(linear, xdata, ydata, p0=(1, 1))
    nlsq_fit_times.append(time.time() - start_time)

print("Summed Fit Times", np.sum(nlsq_fit_times))

plt.figure()
plt.title("Fit Speeds")
plt.plot(lengths, nlsq_fit_times, label="NLSQ")
plt.xlabel("Data Length")
plt.ylabel("Fit Time (seconds)")


# ======================================================================
# The fit speed is slow for every fit. This is because JAX must retrace a function whenever the size of the input array changes. However, NLSQ has a clever way of getting around this. We set a fixed data size (which should be greater than or equal to the largest data we'll fit) and then we use dummy data behind the scenes to keep the array sizes fixed.
#
# We do the same fits as above, but this time we set a fixed array size length when we instantiate the CurveFit object.
# ======================================================================


fixed_length = cap_samples(int(np.amax(lengths)))
jcf = CurveFit(flength=fixed_length)

nlsq_fit_times = []
for length in lengths:
    xdata, ydata = get_random_data(length)

    start_time = time.time()
    popt1, pcov1 = jcf.curve_fit(linear, xdata, ydata, p0=(1, 1))
    nlsq_fit_times.append(time.time() - start_time)

print("Summed Fit Times", np.sum(nlsq_fit_times))

plt.figure()
plt.title("Fit Speeds")
plt.plot(lengths, nlsq_fit_times, label="NLSQ")
plt.xlabel("Data Length")
plt.ylabel("Fit Time (seconds)")


# ======================================================================
# Our fits now run extremely fast irrespective of the datasize. There is a slight caveat to this in that the speed of the fits is always that of the fixed data size even if our actual data is smaller.
#
# If you have two drastically different data sizes in your analysis however, you can instantiate two different CurveFit objects to get an overall fit speedup.
# ======================================================================


lmin = min(10**3, MAX_SAMPLES)
lmax = cap_samples(10**6)
nlengths = 20
lengths1 = np.linspace(
    min(10**3, MAX_SAMPLES), min(5 * 10**4, MAX_SAMPLES), nlengths, dtype=int
)
lengths2 = np.linspace(min(10**5, MAX_SAMPLES), cap_samples(10**6), nlengths, dtype=int)

fixed_length1 = np.amax(lengths1)
fixed_length2 = np.amax(lengths2)

jcf1 = CurveFit(flength=fixed_length1)
jcf2 = CurveFit(flength=fixed_length2)

nlsq_fit_times1 = []
nlsq_fit_times2 = []

for length1, length2 in zip(lengths1, lengths2, strict=False):
    xdata1, ydata1 = get_random_data(length1)
    xdata2, ydata2 = get_random_data(length2)

    start_time = time.time()
    popt1, pcov1 = jcf1.curve_fit(linear, xdata1, ydata1, p0=(1, 1))
    nlsq_fit_times1.append(time.time() - start_time)

    start_time = time.time()
    popt2, pcov2 = jcf2.curve_fit(linear, xdata2, ydata2, p0=(1, 1))
    nlsq_fit_times2.append(time.time() - start_time)

plt.figure()
plt.title("Fit Speeds")
plt.plot(nlsq_fit_times1, label="Small Data")
plt.plot(nlsq_fit_times2, label="Large Data")
plt.legend()

plt.xlabel("Fit Iteration")
plt.ylabel("Fit Time (seconds)")


# ======================================================================
# ## Fitting Multiple Functions
#
# It's important to instantiate a CurveFit object for each different function you're fitting as well to avoid JAX needing to retrace any underlying functions. First we show what happens if we use the same CurveFit object for two functions.
# ======================================================================


def quad_exp(x, a, b, c, d):
    return a * x**2 + b * x + c + jnp.exp(d)


length = cap_samples(3 * 10**5)
x = np.linspace(0, 10, length)

jcf = CurveFit()
nsamples = 3 if QUICK else 21

all_linear_params = np.random.random(size=(nsamples, 2))
all_quad_params = np.random.random(size=(nsamples, 4))

linear_fit_times = []
quad_fit_times = []
for i in range(nsamples):
    y_linear = linear(x, *all_linear_params[i]) + np.random.normal(0, 0.2, size=length)
    y_quad = quad_exp(x, *all_quad_params[i]) + np.random.normal(0, 0.2, size=length)

    # fit the data
    start_time = time.time()
    popt1, pcov1 = jcf.curve_fit(
        linear,
        x,
        y_linear,
        p0=(
            0.5,
            0.5,
        ),
    )
    linear_fit_times.append(time.time() - start_time)

    start_time = time.time()
    popt2, pcov2 = jcf.curve_fit(quad_exp, x, y_quad, p0=(0.5, 0.5, 0.5, 0.5))
    quad_fit_times.append(time.time() - start_time)

print("Summed Fit Times", np.sum(linear_fit_times + quad_fit_times))

plt.figure()
plt.plot(linear_fit_times, label="Linear Function")
plt.plot(quad_fit_times, label="Quad Function")
plt.xlabel("Fit Iteration")
plt.ylabel("Fit Time (seconds)")
plt.legend()
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_quickstart"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# And we see that by using the same fit object retracing is occurring for every fit. Now we instantiate two separate CurveFit objects for the two functions.
# ======================================================================


jcf_linear = CurveFit()
jcf_quad = CurveFit()


linear_fit_times = []
quad_fit_times = []
for i in range(nsamples):
    y_linear = linear(x, *all_linear_params[i]) + np.random.normal(0, 0.2, size=length)
    y_quad = quad_exp(x, *all_quad_params[i]) + np.random.normal(0, 0.2, size=length)

    # fit the data
    start_time = time.time()
    popt1, pcov1 = jcf_linear.curve_fit(
        linear,
        x,
        y_linear,
        p0=(
            0.5,
            0.5,
        ),
    )
    linear_fit_times.append(time.time() - start_time)

    start_time = time.time()
    popt2, pcov2 = jcf_quad.curve_fit(quad_exp, x, y_quad, p0=(0.5, 0.5, 0.5, 0.5))
    quad_fit_times.append(time.time() - start_time)

print("Summed Fit Times", np.sum(linear_fit_times + quad_fit_times))

plt.figure()
plt.plot(linear_fit_times, label="Linear Function")
plt.plot(quad_fit_times, label="Quad Function")
plt.xlabel("Fit Iteration")
plt.ylabel("Fit Time (seconds)")
plt.legend()
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_quickstart"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# And now retracing is only occurring for the first fit for each CurveFit object.
# ======================================================================


# ======================================================================
# ## NLSQ vs. SciPy Fit Speed
#
# Finally, let's compare the speed of NLSQ against SciPy.
# ======================================================================


def quad_exp_numpy(x, a, b, c, d):
    # Clip d to prevent overflow in exp function
    d_clipped = np.clip(
        d, -700, 700
    )  # exp(700) is near max float64, exp(-700) is near 0
    return a * x**2 + b * x + c + np.exp(d_clipped)


length = cap_samples(3 * 10**5)
x = np.linspace(0, 10, length)

jcf = CurveFit()
nlsq_fit_times = []
scipy_fit_times = []
nsamples = 3 if QUICK else 21

all_params = np.random.random(size=(nsamples, 4))

for i in range(nsamples):
    params = get_random_parameters()
    y = quad_exp(x, *all_params[i]) + np.random.normal(0, 0.2, size=length)

    # fit the data
    start_time = time.time()
    popt1, pcov1 = jcf.curve_fit(quad_exp, x, y, p0=(0.5, 0.5, 0.5, 0.5))
    nlsq_fit_times.append(time.time() - start_time)

    start_time = time.time()
    popt2, pcov2 = curve_fit(quad_exp_numpy, x, y, p0=(0.5, 0.5, 0.5, 0.5))
    scipy_fit_times.append(time.time() - start_time)

plt.figure()
plt.title("Fit Speeds")
plt.plot(nlsq_fit_times, label="NLSQ")
plt.plot(scipy_fit_times, label="SciPy")
plt.legend()
plt.xlabel("Fit Iteration")
plt.ylabel("Fit Time (seconds)")


# ======================================================================
# And we see it's so much faster minus the first fit in which tracing is occurring. Thus, by avoiding retracing and utilizing the GPU we get super fast fitting.
# ======================================================================
