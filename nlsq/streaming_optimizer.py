"""Streaming optimizer for datasets that don't fit in memory.

This module provides true streaming optimization that can handle
datasets of unlimited size by processing them in an online fashion.
"""

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    HAS_H5PY = False

import jax.numpy as jnp
import numpy as np

from nlsq.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming optimization.

    Attributes
    ----------
    batch_size : int
        Size of batches to process at once
    learning_rate : float
        Initial learning rate for SGD-based methods
    momentum : float
        Momentum factor for parameter updates
    max_epochs : int
        Maximum number of passes through the data
    convergence_tol : float
        Convergence tolerance for parameter changes
    checkpoint_interval : int
        Save checkpoint every N batches
    use_adam : bool
        Use Adam optimizer instead of SGD
    adam_beta1 : float
        Adam beta1 parameter
    adam_beta2 : float
        Adam beta2 parameter
    adam_eps : float
        Adam epsilon for numerical stability
    """

    batch_size: int = 10000
    learning_rate: float = 0.01
    momentum: float = 0.9
    max_epochs: int = 10
    convergence_tol: float = 1e-6
    checkpoint_interval: int = 100
    use_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    gradient_clip: float = 10.0
    warmup_steps: int = 1000


class DataGenerator:
    """Generate data batches from various sources without loading all into memory."""

    def __init__(self, source, source_type: str = "auto"):
        """Initialize data generator.

        Parameters
        ----------
        source : various
            Data source (file path, array, generator, etc.)
        source_type : str
            Type of source: 'auto', 'hdf5', 'mmap', 'array', 'generator'
        """
        self.source = source
        self.source_type = self._detect_source_type(source, source_type)
        self._setup_source()

    def _detect_source_type(self, source, source_type):
        """Detect the type of data source."""
        if source_type != "auto":
            return source_type

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix in [".h5", ".hdf5"]:
                return "hdf5"
            elif path.suffix in [".npy", ".npz"]:
                return "mmap"
            else:
                return "file"
        elif hasattr(source, "__next__"):
            return "generator"
        elif hasattr(source, "__getitem__"):
            return "array"
        else:
            raise ValueError(f"Cannot detect source type for {type(source)}")

    def _setup_source(self):
        """Set up the data source for streaming."""
        if self.source_type == "hdf5":
            self.file = h5py.File(self.source, "r")
            self.x_data = self.file["x"]
            self.y_data = self.file["y"]
            self.n_samples = len(self.y_data)

        elif self.source_type == "mmap":
            # Memory-mapped numpy array
            data = np.load(self.source, mmap_mode="r")
            if isinstance(data, np.lib.npyio.NpzFile):
                self.x_data = data["x"]
                self.y_data = data["y"]
            else:
                # Assume it's a structured array
                self.x_data = data[:, :-1]
                self.y_data = data[:, -1]
            self.n_samples = len(self.y_data)

        elif self.source_type == "array":
            # In-memory array (but we'll still chunk it)
            self.x_data, self.y_data = self.source
            self.n_samples = len(self.y_data)

        elif self.source_type == "generator":
            # Generator doesn't have a fixed size
            self.n_samples = None

    def generate_batches(self, batch_size: int, shuffle: bool = True) -> Generator:
        """Generate batches of data.

        Parameters
        ----------
        batch_size : int
            Size of each batch
        shuffle : bool
            Whether to shuffle data (not applicable for generators)

        Yields
        ------
        x_batch, y_batch : tuple
            Batch of x and y data
        """
        if self.source_type == "generator":
            # Pass through generator
            yield from self.source

        else:
            # Generate from indexed source
            n_batches = (self.n_samples + batch_size - 1) // batch_size

            if shuffle:
                indices = np.random.permutation(self.n_samples)
            else:
                indices = np.arange(self.n_samples)

            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, self.n_samples)
                batch_indices = indices[start:end]

                # For HDF5, indices must be sorted in increasing order
                if self.source_type == "hdf5":
                    batch_indices = np.sort(batch_indices)

                x_batch = self.x_data[batch_indices]
                y_batch = self.y_data[batch_indices]

                # Convert to numpy arrays if needed
                if not isinstance(x_batch, np.ndarray):
                    x_batch = np.array(x_batch)
                if not isinstance(y_batch, np.ndarray):
                    y_batch = np.array(y_batch)

                yield x_batch, y_batch

    def close(self):
        """Clean up resources."""
        if self.source_type == "hdf5" and hasattr(self, "file"):
            self.file.close()


class StreamingOptimizer:
    """Optimizer that processes data in a streaming fashion.

    This optimizer never loads the full dataset into memory, enabling
    optimization on datasets of unlimited size.
    """

    def __init__(self, config: StreamingConfig | None = None):
        """Initialize streaming optimizer.

        Parameters
        ----------
        config : StreamingConfig, optional
            Configuration for streaming optimization

        Raises
        ------
        ImportError
            If h5py is not installed. Install with: pip install nlsq[streaming]
        """
        if not HAS_H5PY:
            raise ImportError(
                "StreamingOptimizer requires h5py for HDF5 file handling.\n"
                "Install with: pip install nlsq[streaming]\n"
                "Or: pip install h5py"
            )
        self.config = config or StreamingConfig()
        self.reset_state()

    def reset_state(self):
        """Reset optimizer state."""
        self.iteration = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_params = None

        # Optimizer state
        if self.config.use_adam:
            self.m = None  # First moment
            self.v = None  # Second moment
        else:
            self.velocity = None  # Momentum

    def fit_streaming(
        self,
        func: Callable,
        data_source,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        callback: Callable | None = None,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Fit model using streaming data.

        Parameters
        ----------
        func : Callable
            Model function to fit
        data_source : various
            Source of data (file, generator, etc.)
        p0 : np.ndarray
            Initial parameters
        bounds : tuple, optional
            Parameter bounds (lower, upper)
        callback : Callable, optional
            Callback function called after each batch
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed)

        Returns
        -------
        result : dict
            Optimization result
        """
        # Initialize
        params = p0.copy()
        n_params = len(params)
        self.reset_state()

        # Initialize optimizer state
        if self.config.use_adam:
            self.m = np.zeros(n_params)
            self.v = np.zeros(n_params)
        else:
            self.velocity = np.zeros(n_params)

        # Set up data generator
        generator = DataGenerator(data_source)

        # Training loop
        start_time = time.time()
        total_samples = 0
        losses = []

        if verbose >= 1:
            logger.info(
                f"Starting streaming optimization with batch_size={self.config.batch_size}"
            )
            logger.info(
                f"Using {'Adam' if self.config.use_adam else 'SGD with momentum'} optimizer"
            )

        try:
            for epoch in range(self.config.max_epochs):
                self.epoch = epoch
                epoch_loss = 0
                epoch_samples = 0

                if verbose >= 1:
                    logger.info(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")

                # Process batches
                for batch_idx, (x_batch, y_batch) in enumerate(
                    generator.generate_batches(self.config.batch_size)
                ):
                    self.iteration += 1

                    # Compute loss and gradient
                    loss, grad = self._compute_loss_and_gradient(
                        func, params, x_batch, y_batch
                    )

                    # Gradient clipping
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > self.config.gradient_clip:
                        grad = grad * self.config.gradient_clip / grad_norm

                    # Update parameters
                    params = self._update_parameters(params, grad, bounds)

                    # Track statistics
                    batch_size = len(y_batch)
                    epoch_loss += loss * batch_size
                    epoch_samples += batch_size
                    total_samples += batch_size
                    losses.append(loss)

                    # Verbose output
                    if verbose >= 2 and batch_idx % 10 == 0:
                        logger.debug(
                            f"  Batch {batch_idx}: loss={loss:.6f}, grad_norm={grad_norm:.6f}"
                        )

                    # Callback
                    if callback is not None:
                        callback(self.iteration, params, loss)

                    # Checkpointing
                    if self.iteration % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(params, losses)

                # Epoch statistics
                avg_epoch_loss = (
                    epoch_loss / epoch_samples if epoch_samples > 0 else float("inf")
                )

                if verbose >= 1:
                    elapsed = time.time() - start_time
                    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
                    logger.info(f"  Epoch loss: {avg_epoch_loss:.6f}")
                    logger.info(f"  Samples/sec: {samples_per_sec:.0f}")

                # Check for improvement
                if avg_epoch_loss < self.best_loss:
                    self.best_loss = avg_epoch_loss
                    self.best_params = params.copy()

                # Convergence check
                if epoch > 0 and len(losses) > 100:
                    recent_loss = np.mean(losses[-100:])
                    old_loss = np.mean(losses[-200:-100])
                    if abs(recent_loss - old_loss) < self.config.convergence_tol:
                        if verbose >= 1:
                            logger.info(f"Converged after {epoch + 1} epochs")
                        break

        finally:
            generator.close()

        # Prepare result
        elapsed_time = time.time() - start_time
        result = {
            "x": self.best_params,
            "fun": self.best_loss,
            "success": True,
            "message": "Streaming optimization completed",
            "nit": self.iteration,
            "n_epochs": self.epoch + 1,
            "total_samples": total_samples,
            "time": elapsed_time,
            "samples_per_sec": total_samples / elapsed_time
            if elapsed_time > 0
            else 0.0,
            "final_loss": losses[-1] if losses else float("inf"),
            "loss_history": np.array(losses),
        }

        return result

    def _compute_loss_and_gradient(
        self,
        func: Callable,
        params: np.ndarray,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute loss and gradient for a batch.

        Parameters
        ----------
        func : Callable
            Model function
        params : np.ndarray
            Current parameters
        x_batch : np.ndarray
            Batch of x data
        y_batch : np.ndarray
            Batch of y data

        Returns
        -------
        loss : float
            Batch loss
        grad : np.ndarray
            Parameter gradient
        """
        # Convert to JAX arrays for computation
        params_jax = jnp.array(params)

        # Compute predictions
        y_pred = func(x_batch, *params)

        # Compute loss (MSE)
        residuals = y_pred - y_batch
        loss = np.mean(residuals**2)

        # Compute gradient using finite differences (simplified)
        # In practice, would use JAX autodiff
        eps = 1e-8
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps

            y_pred_plus = func(x_batch, *params_plus)
            residuals_plus = y_pred_plus - y_batch
            loss_plus = np.mean(residuals_plus**2)

            grad[i] = (loss_plus - loss) / eps

        return loss, grad

    def _update_parameters(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Update parameters using gradient.

        Parameters
        ----------
        params : np.ndarray
            Current parameters
        grad : np.ndarray
            Gradient
        bounds : tuple, optional
            Parameter bounds

        Returns
        -------
        params_new : np.ndarray
            Updated parameters
        """
        # Learning rate schedule with warmup
        if self.iteration < self.config.warmup_steps:
            lr = self.config.learning_rate * self.iteration / self.config.warmup_steps
        else:
            # Cosine annealing
            progress = (self.iteration - self.config.warmup_steps) / max(
                1, self.config.max_epochs * 1000
            )
            lr = (
                self.config.learning_rate
                * 0.5
                * (1 + np.cos(np.pi * min(1.0, progress)))
            )

        if self.config.use_adam:
            # Adam optimizer
            self.m = (
                self.config.adam_beta1 * self.m + (1 - self.config.adam_beta1) * grad
            )
            self.v = (
                self.config.adam_beta2 * self.v + (1 - self.config.adam_beta2) * grad**2
            )

            # Bias correction
            m_hat = self.m / (1 - self.config.adam_beta1**self.iteration)
            v_hat = self.v / (1 - self.config.adam_beta2**self.iteration)

            # Update
            params_new = params - lr * m_hat / (np.sqrt(v_hat) + self.config.adam_eps)

        else:
            # SGD with momentum
            self.velocity = self.config.momentum * self.velocity - lr * grad
            params_new = params + self.velocity

        # Apply bounds if specified
        if bounds is not None:
            lower, upper = bounds
            params_new = np.clip(params_new, lower, upper)

        return params_new

    def _save_checkpoint(self, params: np.ndarray, losses: list):
        """Save checkpoint to disk.

        Parameters
        ----------
        params : np.ndarray
            Current parameters
        losses : list
            Loss history
        """
        checkpoint = {
            "params": params,
            "iteration": self.iteration,
            "epoch": self.epoch,
            "losses": losses[-1000:],  # Keep last 1000 losses
            "best_params": self.best_params,
            "best_loss": self.best_loss,
        }

        # Save to file
        checkpoint_path = f"checkpoint_iter_{self.iteration}.npz"
        np.savez_compressed(checkpoint_path, **checkpoint)


def create_hdf5_dataset(
    filename: str,
    func: Callable,
    params: np.ndarray,
    n_samples: int,
    chunk_size: int = 10000,
    noise_level: float = 0.01,
):
    """Create an HDF5 dataset for testing streaming optimization.

    Parameters
    ----------
    filename : str
        Output HDF5 filename
    func : Callable
        Function to generate data
    params : np.ndarray
        True parameters
    n_samples : int
        Number of samples to generate
    chunk_size : int
        Chunk size for HDF5 storage
    noise_level : float
        Noise level to add to y data
    """
    with h5py.File(filename, "w") as f:
        # Create datasets
        x_dataset = f.create_dataset(
            "x", (n_samples,), dtype="f8", chunks=(chunk_size,)
        )
        y_dataset = f.create_dataset(
            "y", (n_samples,), dtype="f8", chunks=(chunk_size,)
        )

        # Generate data in chunks
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            size = end - i

            # Generate x data
            x_chunk = np.random.randn(size)
            x_dataset[i:end] = x_chunk

            # Generate y data
            y_true = func(x_chunk, *params)
            y_noisy = y_true + noise_level * np.random.randn(size)
            y_dataset[i:end] = y_noisy

        # Store metadata
        f.attrs["n_samples"] = n_samples
        f.attrs["true_params"] = params
        f.attrs["noise_level"] = noise_level

    logger.info(f"Created HDF5 dataset with {n_samples} samples in {filename}")


def fit_unlimited_data(
    func: Callable,
    data_source,
    p0: np.ndarray,
    config: StreamingConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Fit model to unlimited data using streaming optimization.

    This is the main entry point for fitting models to datasets that
    don't fit in memory.

    Parameters
    ----------
    func : Callable
        Model function to fit
    data_source : various
        Data source (file path, generator, etc.)
    p0 : np.ndarray
        Initial parameters
    config : StreamingConfig, optional
        Streaming configuration
    **kwargs
        Additional arguments passed to optimizer

    Returns
    -------
    result : dict
        Optimization result

    Examples
    --------
    >>> # Fit to 100M points stored in HDF5
    >>> result = fit_unlimited_data(
    ...     lambda x, a, b: a * np.exp(-b * x),
    ...     'huge_dataset.h5',
    ...     p0=[1.0, 0.5]
    ... )

    >>> # Fit to streaming data generator
    >>> def data_generator():
    ...     while True:
    ...         x = np.random.randn(1000)
    ...         y = 2 * x + 1 + np.random.randn(1000) * 0.1
    ...         yield x, y
    >>>
    >>> result = fit_unlimited_data(
    ...     lambda x, a, b: a * x + b,
    ...     data_generator(),
    ...     p0=[1.0, 0.0],
    ...     config=StreamingConfig(max_epochs=5)
    ... )
    """
    optimizer = StreamingOptimizer(config)
    return optimizer.fit_streaming(func, data_source, p0, **kwargs)
