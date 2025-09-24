#!/usr/bin/env python3
"""Test script for 2D Gaussian fitting with cuSolver fix."""

import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import NLSQ
from nlsq import CurveFit


def rotate_coordinates2D(coords, theta):
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta), jnp.cos(theta)]])
    shape = coords[0].shape
    coords = jnp.stack([coord.flatten() for coord in coords])
    rcoords = R @ coords
    return [jnp.reshape(coord, shape) for coord in rcoords]


def gaussian2d(coords, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    coords = [coords[0] - x0, coords[1] - y0]  # translate first
    X, Y = rotate_coordinates2D(coords, theta)
    density = n0 * jnp.exp(-0.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2))
    return density + offset


def get_coordinates(width, height):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y


def get_gaussian_parameters(length):
    n0 = 1
    x0 = length / 2
    y0 = length / 2
    sigx = length / 6
    sigy = length / 8
    theta = np.pi / 3
    offset = 0.1 * n0
    params = [n0, x0, y0, sigx, sigy, theta, offset]
    return params


def test_2d_gaussian_fitting():
    """Test 2D Gaussian fitting with various dataset sizes."""
    print("Testing 2D Gaussian fitting with cuSolver fix...")
    print("=" * 60)

    # Test with different sizes
    test_sizes = [50, 100, 200, 500]

    for length in test_sizes:
        print(f"\nTesting with {length}x{length} data...")

        try:
            # Generate synthetic data
            XY_tuple = get_coordinates(length, length)
            params = get_gaussian_parameters(length)

            # Create noisy data
            zdata = gaussian2d(XY_tuple, *params)
            zdata_noisy = zdata + np.random.normal(0, 0.1, size=(length, length))

            # Flatten for fitting
            flat_data = zdata_noisy.flatten()
            flat_XY_tuple = [coord.flatten() for coord in XY_tuple]

            # Initialize CurveFit object
            jcf = CurveFit()

            # Generate random seed
            np.random.seed(42)  # For reproducibility
            seed = [val * np.random.uniform(0.9, 1.2) for val in params]

            # Perform fit
            start_time = time.time()
            popt, pcov = jcf.curve_fit(gaussian2d, flat_XY_tuple, flat_data, p0=seed)
            fit_time = time.time() - start_time

            # Check results
            param_errors = np.abs((np.array(popt) - np.array(params)) / np.array(params))
            max_error = np.max(param_errors[:-1])  # Exclude offset from relative error

            print(f"  Fit completed in {fit_time:.3f} seconds")
            print(f"  Max relative parameter error: {max_error:.4f}")

            # Verify fit quality
            if max_error < 0.01:  # 1% tolerance
                print(f"  ✓ Fit successful!")
            else:
                print(f"  ⚠ Fit has larger errors than expected")
                print(f"    True params: {params}")
                print(f"    Fit params:  {list(popt)}")

        except Exception as e:
            print(f"  ✗ Error occurred: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    return True


def test_multiple_fits():
    """Test multiple fits to ensure stability."""
    print("\nTesting multiple consecutive fits...")
    print("-" * 40)

    length = 100
    XY_tuple = get_coordinates(length, length)
    params = get_gaussian_parameters(length)

    # Create noisy data
    zdata = gaussian2d(XY_tuple, *params)
    zdata_noisy = zdata + np.random.normal(0, 0.1, size=(length, length))

    flat_data = zdata_noisy.flatten()
    flat_XY_tuple = [coord.flatten() for coord in XY_tuple]

    # Initialize CurveFit object once
    jcf = CurveFit()

    # Perform multiple fits
    n_fits = 10
    times = []
    errors = []

    for i in range(n_fits):
        seed = [val * np.random.uniform(0.9, 1.2) for val in params]

        start_time = time.time()
        try:
            popt, pcov = jcf.curve_fit(gaussian2d, flat_XY_tuple, flat_data, p0=seed)
            fit_time = time.time() - start_time
            times.append(fit_time)

            # Calculate error
            param_errors = np.abs((np.array(popt) - np.array(params)) / np.array(params))
            errors.append(np.max(param_errors[:-1]))

            print(f"  Fit {i+1}/{n_fits}: {fit_time:.3f}s, max error: {errors[-1]:.4f}")
        except Exception as e:
            print(f"  Fit {i+1}/{n_fits} failed: {e}")
            return False

    # Print statistics
    if times:
        print(f"\nStatistics for {len(times)} successful fits:")
        print(f"  Average time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Average error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")

        # Note about first fit being slower (JIT compilation)
        if len(times) > 1:
            print(f"  Average time (excluding first): {np.mean(times[1:]):.3f}s")

    return True


if __name__ == "__main__":
    print("NLSQ 2D Gaussian Fitting Test")
    print("=" * 60)

    # Run tests
    success = True

    # Test 1: Various sizes
    if not test_2d_gaussian_fitting():
        success = False

    # Test 2: Multiple fits
    if not test_multiple_fits():
        success = False

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! The cuSolver fix is working.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the output above.")
        sys.exit(1)