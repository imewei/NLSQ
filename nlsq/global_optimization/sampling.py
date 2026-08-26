"""
Sampling Algorithms for Global Optimization
============================================

This module provides sampling algorithms for generating starting points in
multi-start global optimization. Includes Latin Hypercube Sampling (LHS),
Sobol sequences, and Halton sequences.

All samplers generate samples in the unit hypercube [0, 1]^n_dims, which
can then be transformed to parameter bounds using scale_samples_to_bounds().

Key Features
------------
- Latin Hypercube Sampling with stratification guarantees
- Sobol quasi-random sequences for low-discrepancy sampling
- Halton sequences using prime bases
- Bounds transformation utilities
- Centering around initial parameter estimates

Examples
--------
Generate LHS samples:

>>> from nlsq.global_optimization.sampling import latin_hypercube_sample
>>> import jax
>>> samples = latin_hypercube_sample(10, 3, rng_key=jax.random.PRNGKey(42))
>>> samples.shape
(10, 3)

Scale samples to parameter bounds:

>>> from nlsq.global_optimization.sampling import scale_samples_to_bounds
>>> import jax.numpy as jnp
>>> lb = jnp.array([0.0, -10.0])
>>> ub = jnp.array([100.0, 10.0])
>>> scaled = scale_samples_to_bounds(samples[:, :2], lb, ub)

Use the sampler factory:

>>> from nlsq.global_optimization.sampling import get_sampler
>>> sampler = get_sampler('lhs')
>>> samples = sampler(20, 4)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "center_samples_around_p0",
    "get_sampler",
    "halton_sample",
    "latin_hypercube_sample",
    "scale_samples_to_bounds",
    "sobol_sample",
]


# First 20 prime numbers for Halton sequence bases
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Sobol primitive-polynomial parameters for dimensions 2-21, from the real
# Joe & Kuo (2008) direction-number table
# (https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201).
# Dimension 1 needs no entry: it is the plain base-2 van der Corput sequence.
# Each tuple is (degree s, polynomial coefficient a, initial values m_1..m_s).
_SOBOL_POLY_PARAMS: list[tuple[int, int, list[int]]] = [
    (1, 0, [1]),
    (2, 1, [1, 3]),
    (3, 1, [1, 3, 1]),
    (3, 2, [1, 1, 1]),
    (4, 1, [1, 1, 3, 3]),
    (4, 4, [1, 3, 5, 13]),
    (5, 2, [1, 1, 5, 5, 17]),
    (5, 4, [1, 1, 5, 5, 5]),
    (5, 7, [1, 1, 7, 11, 19]),
    (5, 11, [1, 1, 5, 1, 1]),
    (5, 13, [1, 1, 1, 3, 11]),
    (5, 14, [1, 3, 5, 5, 31]),
    (6, 1, [1, 3, 3, 9, 7, 49]),
    (6, 13, [1, 1, 1, 15, 21, 21]),
    (6, 16, [1, 3, 1, 13, 27, 49]),
    (6, 19, [1, 1, 1, 15, 7, 5]),
    (6, 22, [1, 3, 1, 15, 13, 25]),
    (6, 25, [1, 1, 5, 5, 19, 61]),
    (7, 1, [1, 3, 7, 11, 23, 15, 103]),
    (7, 4, [1, 3, 7, 13, 13, 15, 69]),
]


def _sobol_direction_numbers(n_dims: int, max_bits: int) -> list[list[int]]:
    """Compute scaled Sobol direction numbers via the Joe & Kuo recurrence.

    For dimension d with primitive polynomial degree ``s``, coefficient
    ``a`` and initial values ``m_1..m_s``:

        V[i] = m[i] << (max_bits - i)                     for i in 1..s
        V[i] = V[i-s] ^ (V[i-s] >> s) ^ sum_k bit_k * V[i-k]   for i in s+1..max_bits

    where ``bit_k = (a >> (s-1-k)) & 1`` for k in 1..s-1. Dimension 1 uses
    the trivial degree-0 case (plain van der Corput): V[i] = 1 << (max_bits - i).
    """
    v: list[list[int]] = []
    mask = (1 << max_bits) - 1
    for dim in range(n_dims):
        v_dim = [0] * max_bits
        if dim == 0:
            for i in range(1, max_bits + 1):
                v_dim[i - 1] = (1 << (max_bits - i)) & mask
        else:
            s, a, m = _SOBOL_POLY_PARAMS[dim - 1]
            for i in range(1, s + 1):
                v_dim[i - 1] = (m[i - 1] << (max_bits - i)) & mask
            for i in range(s + 1, max_bits + 1):
                val = v_dim[i - 1 - s] ^ (v_dim[i - 1 - s] >> s)
                for k in range(1, s):
                    bit = (a >> (s - 1 - k)) & 1
                    if bit:
                        val ^= v_dim[i - 1 - k]
                v_dim[i - 1] = val & mask
        v.append(v_dim)
    return v


def latin_hypercube_sample(
    n_samples: int,
    n_dims: int,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """Generate Latin Hypercube samples in the unit hypercube.

    Latin Hypercube Sampling divides each dimension into n_samples equal strata
    and ensures exactly one sample falls in each stratum for each dimension.
    This provides better coverage than random sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_dims : int
        Number of dimensions.
    rng_key : jax.Array, optional
        JAX random key. If None, a default key is created.

    Returns
    -------
    jax.Array
        Array of shape (n_samples, n_dims) with values in [0, 1).

    Notes
    -----
    The stratification property ensures that when samples are sorted in any
    dimension, sample i falls in the stratum [i/n, (i+1)/n) for that dimension.

    Examples
    --------
    >>> import jax
    >>> samples = latin_hypercube_sample(10, 3, rng_key=jax.random.PRNGKey(42))
    >>> samples.shape
    (10, 3)

    >>> # Verify stratification
    >>> import numpy as np
    >>> sorted_dim0 = np.sort(np.asarray(samples[:, 0]))
    >>> for i, s in enumerate(sorted_dim0):
    ...     assert i / 10 <= s < (i + 1) / 10
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}")

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Split the key for different random operations
    keys = jax.random.split(rng_key, n_dims + 1)

    samples_list = []
    for dim in range(n_dims):
        # Generate random positions within each stratum
        key_uniform = keys[dim]
        key_perm = (
            keys[dim + 1] if dim + 1 < len(keys) else jax.random.fold_in(keys[0], dim)
        )

        # Random offset within each stratum
        uniform_samples = jax.random.uniform(key_perm, shape=(n_samples,))

        # Create stratum indices and shuffle them
        stratum_indices = jnp.arange(n_samples)
        shuffled_indices = jax.random.permutation(key_uniform, stratum_indices)

        # Place samples: stratum_start + offset_within_stratum
        # stratum i starts at i/n_samples, has width 1/n_samples
        dim_samples = (shuffled_indices + uniform_samples) / n_samples

        samples_list.append(dim_samples)

    # Stack dimensions
    samples = jnp.stack(samples_list, axis=1)
    return samples


def sobol_sample(
    n_samples: int,
    n_dims: int,
    skip: int = 0,
) -> jax.Array:
    """Generate Sobol quasi-random samples in the unit hypercube.

    Sobol sequences are deterministic low-discrepancy sequences that provide
    excellent coverage of the parameter space.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_dims : int
        Number of dimensions. Must be <= 21.
    skip : int, default=0
        Number of initial samples to skip (for different starting points).

    Returns
    -------
    jax.Array
        Array of shape (n_samples, n_dims) with values in [0, 1].

    Notes
    -----
    This implementation uses a Gray code approach for efficient generation.
    For high dimensions (>21), consider using scipy.stats.qmc.Sobol.

    Examples
    --------
    >>> samples = sobol_sample(16, 4)
    >>> samples.shape
    (16, 4)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}")
    if n_dims > 21:
        raise ValueError(f"n_dims must be <= 21 for Sobol sequence, got {n_dims}")

    # Use 32-bit precision for integer operations
    max_bits = 32

    v = _sobol_direction_numbers(n_dims, max_bits)

    samples = []

    # Gray code counter approach (Antonov-Saleev ordering): point i is
    # obtained from point i-1 by XORing direction number v[c], where c is
    # the position of the rightmost zero bit of (i-1) -- i.e. of the index
    # of the point *just emitted*, not of the point about to be produced.
    x = [0] * n_dims  # Current state

    for i in range(skip + n_samples):
        if i >= skip:
            # Convert current state to floating point in [0, 1]
            sample = [x[dim] / (2.0**max_bits) for dim in range(n_dims)]
            samples.append(sample)

        # Find position of rightmost zero bit in i (the just-emitted index)
        c = 0
        temp = i
        while temp & 1:
            c += 1
            temp >>= 1

        # XOR update for all dimensions
        for dim in range(n_dims):
            if c < max_bits:
                x[dim] ^= v[dim][c]

    return jnp.array(samples)


def halton_sample(
    n_samples: int,
    n_dims: int,
    skip: int = 0,
) -> jax.Array:
    """Generate Halton quasi-random samples in the unit hypercube.

    Halton sequences use different prime number bases for each dimension
    to create low-discrepancy samples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_dims : int
        Number of dimensions. Must be <= 20 (number of available prime bases).
    skip : int, default=0
        Number of initial samples to skip (for different starting points).

    Returns
    -------
    jax.Array
        Array of shape (n_samples, n_dims) with values in [0, 1).

    Notes
    -----
    Each dimension uses a different prime base: 2, 3, 5, 7, 11, ...
    The Halton sequence in base b for index n is computed by reflecting
    the base-b representation of n about the decimal point.

    Examples
    --------
    >>> samples = halton_sample(20, 5)
    >>> samples.shape
    (20, 5)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}")
    if n_dims > len(_PRIMES):
        raise ValueError(
            f"n_dims must be <= {len(_PRIMES)} for Halton sequence, got {n_dims}",
        )

    samples = []
    for i in range(skip, skip + n_samples):
        sample = []
        for dim in range(n_dims):
            base = _PRIMES[dim]
            value = _halton_element(i + 1, base)  # 1-indexed
            sample.append(value)
        samples.append(sample)

    return jnp.array(samples)


def _halton_element(index: int, base: int) -> float:
    """Compute a single Halton sequence element.

    Parameters
    ----------
    index : int
        Index in the sequence (1-indexed).
    base : int
        Prime base for the sequence.

    Returns
    -------
    float
        Halton sequence value in [0, 1).
    """
    result = 0.0
    f = 1.0 / base
    i = index

    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base

    return result


def scale_samples_to_bounds(
    samples: jax.Array,
    lb: jax.Array,
    ub: jax.Array,
) -> jax.Array:
    """Transform samples from [0, 1] to parameter bounds [lb, ub].

    Parameters
    ----------
    samples : jax.Array
        Array of shape (n_samples, n_dims) with values in [0, 1].
    lb : jax.Array
        Lower bounds for each dimension, shape (n_dims,).
    ub : jax.Array
        Upper bounds for each dimension, shape (n_dims,).

    Returns
    -------
    jax.Array
        Scaled samples with values in [lb, ub].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> samples = jnp.array([[0.0, 0.5], [1.0, 0.0]])
    >>> lb = jnp.array([0.0, -10.0])
    >>> ub = jnp.array([10.0, 10.0])
    >>> scaled = scale_samples_to_bounds(samples, lb, ub)
    >>> # First sample: [0, 0] in [0,1] -> [0, 0] in scaled space
    >>> # Second sample: [1, 0] in [0,1] -> [10, -10] in scaled space
    """
    lb = jnp.asarray(lb)
    ub = jnp.asarray(ub)

    # Linear interpolation: sample * (ub - lb) + lb
    return samples * (ub - lb) + lb


def center_samples_around_p0(
    samples: jax.Array,
    p0: jax.Array,
    scale_factor: float,
    lb: jax.Array,
    ub: jax.Array,
) -> jax.Array:
    """Center samples around p0 with specified scale factor.

    Instead of sampling uniformly in [lb, ub], this creates samples
    in a region centered at p0. The region width is scale_factor * (ub - lb).

    Parameters
    ----------
    samples : jax.Array
        Array of shape (n_samples, n_dims) with values in [0, 1].
    p0 : jax.Array
        Center point (initial parameter guess), shape (n_dims,).
    scale_factor : float
        Width of exploration region as fraction of (ub - lb).
        0.5 means explore within 50% of the range centered at p0.
    lb : jax.Array
        Lower bounds for each dimension, shape (n_dims,).
    ub : jax.Array
        Upper bounds for each dimension, shape (n_dims,).

    Returns
    -------
    jax.Array
        Centered samples, clipped to stay within [lb, ub].

    Notes
    -----
    The exploration region is:
    - Half-width = scale_factor * (ub - lb) / 2
    - Region = [p0 - half_width, p0 + half_width]
    - Clipped to [lb, ub] to ensure valid samples

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> samples = jnp.array([[0.0], [0.5], [1.0]])
    >>> p0 = jnp.array([5.0])
    >>> lb = jnp.array([0.0])
    >>> ub = jnp.array([10.0])
    >>> centered = center_samples_around_p0(samples, p0, 0.5, lb, ub)
    >>> # With scale_factor=0.5, half-width = 0.5 * 10 / 2 = 2.5
    >>> # Region = [2.5, 7.5]
    >>> # Sample at 0.5 maps to center (5.0)
    """
    p0 = jnp.asarray(p0)
    lb = jnp.asarray(lb)
    ub = jnp.asarray(ub)

    # Clamp p0 into [lb, ub] first -- an out-of-bounds p0 (e.g. a bad user
    # guess) would otherwise shift the centered region so far that
    # jnp.maximum(centered_lb, lb)/jnp.minimum(centered_ub, ub) below can
    # produce an inverted [centered_lb, centered_ub] interval, generating
    # samples outside the true [lb, ub] bounds.
    p0 = jnp.clip(p0, lb, ub)

    # Compute the exploration range
    full_range = ub - lb
    half_width = scale_factor * full_range / 2.0

    # Compute centered bounds
    centered_lb = p0 - half_width
    centered_ub = p0 + half_width

    # Clip to original bounds
    centered_lb = jnp.maximum(centered_lb, lb)
    centered_ub = jnp.minimum(centered_ub, ub)

    # Transform samples to centered bounds
    return scale_samples_to_bounds(samples, centered_lb, centered_ub)


def get_sampler(sampler_type: str) -> Callable[..., jax.Array]:
    """Get a sampler function by type name.

    Factory function that returns the appropriate sampling function
    based on the sampler type string.

    Parameters
    ----------
    sampler_type : str
        Type of sampler. One of: 'lhs', 'sobol', 'halton' (case-insensitive).

    Returns
    -------
    Callable
        Sampler function with signature ``(n_samples, n_dims, **kwargs) -> jax.Array``.

    Raises
    ------
    ValueError
        If sampler_type is not recognized.

    Examples
    --------
    >>> sampler = get_sampler('lhs')
    >>> samples = sampler(10, 3)
    >>> samples.shape
    (10, 3)

    >>> sampler = get_sampler('sobol')
    >>> samples = sampler(16, 4)
    """
    sampler_type_lower = sampler_type.lower()

    samplers: dict[str, Callable[..., jax.Array]] = {
        "lhs": latin_hypercube_sample,
        "sobol": sobol_sample,
        "halton": halton_sample,
    }

    if sampler_type_lower not in samplers:
        valid_types = list(samplers.keys())
        raise ValueError(
            f"Unknown sampler type '{sampler_type}'. Valid types: {valid_types}",
        )

    return samplers[sampler_type_lower]
