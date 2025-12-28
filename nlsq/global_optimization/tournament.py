"""
Tournament Selector for Global Optimization
============================================

This module provides the TournamentSelector class for progressive elimination
in multi-start optimization on streaming/large datasets. Tournament selection
is memory-efficient as it evaluates candidates on data batches without loading
the full dataset.

Key Features
------------
- Progressive elimination: N -> N/2 -> N/4 -> ... -> top M candidates
- Memory-efficient evaluation on streaming data batches
- Configurable elimination fraction and batches per round
- Checkpoint/resume support for fault tolerance
- Graceful handling of numerical failures

Examples
--------
Basic tournament selection:

>>> from nlsq.global_optimization import TournamentSelector, GlobalOptimizationConfig
>>> import numpy as np
>>>
>>> # Generate candidates
>>> candidates = np.random.randn(20, 3)  # 20 candidates, 3 parameters
>>>
>>> config = GlobalOptimizationConfig(
...     n_starts=20,
...     elimination_rounds=3,
...     elimination_fraction=0.5,
...     batches_per_round=50,
... )
>>>
>>> selector = TournamentSelector(candidates, config)
>>>
>>> def model(x, a, b, c):
...     return a * x**2 + b * x + c
>>>
>>> def data_batch_generator():
...     for _ in range(200):
...         x_batch = np.random.randn(100)
...         y_batch = 1.0 * x_batch**2 + 2.0 * x_batch + 3.0
...         yield x_batch, y_batch
>>>
>>> best_candidates = selector.run_tournament(
...     data_batch_iterator=data_batch_generator(),
...     model=model,
...     top_m=1,
... )

See Also
--------
GlobalOptimizationConfig : Configuration for multi-start optimization
MultiStartOrchestrator : Orchestrates multi-start for standard datasets
AdaptiveHybridStreamingOptimizer : Streaming optimizer with tournament integration
"""

from collections.abc import Callable, Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.utils.logging import get_logger

__all__ = ["TournamentSelector"]


class TournamentSelector:
    """Tournament selector for progressive elimination in multi-start optimization.

    This class implements tournament-style progressive elimination for selecting
    the best starting points when optimizing on large/streaming datasets. Instead
    of evaluating all candidates on the full dataset, candidates are evaluated
    on streaming batches and the worst performers are eliminated each round.

    The tournament proceeds as:
    - Round 1: N candidates -> Keep top (1 - elimination_fraction) * N
    - Round 2: Survivors -> Keep top (1 - elimination_fraction) * survivors
    - ...
    - Final: Return top M candidates

    Parameters
    ----------
    candidates : np.ndarray
        Array of shape (n_candidates, n_params) containing candidate starting points.
    config : GlobalOptimizationConfig
        Configuration controlling tournament parameters:
        - elimination_rounds: Number of elimination rounds
        - elimination_fraction: Fraction to eliminate each round (default 0.5)
        - batches_per_round: Number of batches to evaluate per round

    Attributes
    ----------
    n_candidates : int
        Total number of candidates.
    n_params : int
        Number of parameters per candidate.
    survival_mask : np.ndarray
        Boolean mask indicating which candidates are still alive.
    cumulative_losses : np.ndarray
        Accumulated loss for each candidate (inf for eliminated).
    current_round : int
        Current tournament round (0-indexed).
    round_history : list
        History of each round with statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from nlsq.global_optimization import TournamentSelector, GlobalOptimizationConfig
    >>>
    >>> candidates = np.random.randn(16, 3)
    >>> config = GlobalOptimizationConfig(
    ...     n_starts=16,
    ...     elimination_rounds=3,
    ...     elimination_fraction=0.5,
    ...     batches_per_round=10,
    ... )
    >>> selector = TournamentSelector(candidates, config)
    >>> print(f"Starting with {selector.n_candidates} candidates")
    Starting with 16 candidates

    Notes
    -----
    Tournament selection is particularly effective for:
    - Large datasets where full evaluation is expensive
    - Streaming datasets that don't fit in memory
    - High-dimensional parameter spaces with many local minima

    The elimination_fraction parameter controls the aggressiveness of pruning:
    - 0.5 (default): Eliminate half each round (log2(N) rounds to get to 1)
    - 0.25: Eliminate 25% each round (slower, more conservative)
    - 0.75: Eliminate 75% each round (faster, more aggressive)
    """

    def __init__(
        self,
        candidates: np.ndarray,
        config: GlobalOptimizationConfig,
    ):
        """Initialize tournament selector.

        Parameters
        ----------
        candidates : np.ndarray
            Array of shape (n_candidates, n_params) with candidate starting points.
        config : GlobalOptimizationConfig
            Configuration for tournament parameters.
        """
        self.candidates = np.asarray(candidates)
        self.config = config
        self.logger = get_logger("tournament")

        # Validate candidates shape
        if self.candidates.ndim == 1:
            # Single candidate
            self.candidates = self.candidates.reshape(1, -1)

        self.n_candidates = self.candidates.shape[0]
        self.n_params = self.candidates.shape[1]

        # Tournament state
        self.survival_mask = np.ones(self.n_candidates, dtype=bool)
        self.cumulative_losses = np.zeros(self.n_candidates)
        self.evaluation_counts = np.zeros(self.n_candidates, dtype=int)
        self.current_round = 0
        self.round_history: list[dict[str, Any]] = []

        # Tracking for diagnostics
        self.total_batches_evaluated = 0
        self.numerical_failures = 0

    @property
    def n_survivors(self) -> int:
        """Number of currently surviving candidates."""
        return int(np.sum(self.survival_mask))

    def run_tournament(
        self,
        data_batch_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        model: Callable,
        top_m: int = 1,
    ) -> list[np.ndarray]:
        """Run full tournament selection.

        Executes all elimination rounds and returns the top M surviving candidates.

        Parameters
        ----------
        data_batch_iterator : Iterator
            Iterator yielding (x_batch, y_batch) tuples of data.
        model : Callable
            Model function with signature ``model(x, *params) -> predictions``.
        top_m : int, default=1
            Number of top candidates to return.

        Returns
        -------
        list[np.ndarray]
            List of top M candidate parameter arrays, sorted by loss (best first).

        Notes
        -----
        The iterator is consumed during tournament execution. Ensure it yields
        enough batches: elimination_rounds * batches_per_round.

        If the iterator runs out of batches before completing all rounds,
        the tournament will return the best candidates found so far.
        """
        self.logger.info(
            f"Starting tournament with {self.n_candidates} candidates, "
            f"{self.config.elimination_rounds} rounds, "
            f"elimination_fraction={self.config.elimination_fraction}"
        )

        # Handle edge case: no elimination rounds
        if self.config.elimination_rounds == 0:
            self.logger.debug("No elimination rounds configured, evaluating once")
            self._evaluate_initial_round(data_batch_iterator, model)
            return self.get_top_candidates(top_m)

        # Run elimination rounds
        for round_num in range(self.config.elimination_rounds):
            if self.n_survivors <= top_m:
                self.logger.info(
                    f"Round {round_num}: Only {self.n_survivors} survivors, "
                    f"stopping early (need {top_m})"
                )
                break

            try:
                self._run_single_round(data_batch_iterator, model, round_num)
            except StopIteration:
                self.logger.warning(
                    f"Data exhausted during round {round_num}, returning best so far"
                )
                break

            self.current_round = round_num + 1

        self.logger.info(
            f"Tournament complete: {self.n_survivors} survivors from "
            f"{self.n_candidates} candidates"
        )

        return self.get_top_candidates(top_m)

    def _run_single_round(
        self,
        data_batch_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        model: Callable,
        round_number: int,
    ) -> None:
        """Run a single elimination round.

        Parameters
        ----------
        data_batch_iterator : Iterator
            Data batch iterator.
        model : Callable
            Model function.
        round_number : int
            Current round number (0-indexed).
        """
        n_survivors_before = self.n_survivors

        self.logger.debug(
            f"Round {round_number}: Evaluating {n_survivors_before} survivors "
            f"on {self.config.batches_per_round} batches"
        )

        # Reset losses for this round
        round_losses = np.zeros(self.n_candidates)
        round_loss_counts = np.zeros(self.n_candidates, dtype=int)

        # Evaluate on batches
        for batch_idx in range(self.config.batches_per_round):
            try:
                x_batch, y_batch = next(data_batch_iterator)
            except StopIteration:
                if batch_idx == 0:
                    raise  # No batches at all
                break  # Use partial evaluation

            batch_losses = self._evaluate_candidates_on_batch(x_batch, y_batch, model)

            # Accumulate losses for survivors
            for i in range(self.n_candidates):
                if self.survival_mask[i] and np.isfinite(batch_losses[i]):
                    round_losses[i] += batch_losses[i]
                    round_loss_counts[i] += 1

            self.total_batches_evaluated += 1

        # Compute average loss for this round (avoid division by zero)
        # Use safe division with masked array to prevent warning
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_round_losses = np.where(
                round_loss_counts > 0,
                round_losses / np.maximum(round_loss_counts, 1),
                np.inf,
            )

        # Update cumulative losses
        self.cumulative_losses += avg_round_losses
        self.evaluation_counts += round_loss_counts

        # Perform elimination
        n_to_eliminate = int(n_survivors_before * self.config.elimination_fraction)
        n_to_eliminate = max(0, min(n_to_eliminate, n_survivors_before - 1))

        if n_to_eliminate > 0:
            self._eliminate_worst(n_to_eliminate, avg_round_losses)

        # Record round history
        self.round_history.append(
            {
                "round": round_number,
                "n_survivors_before": n_survivors_before,
                "n_survivors_after": self.n_survivors,
                "n_eliminated": n_survivors_before - self.n_survivors,
                "batches_evaluated": min(batch_idx + 1, self.config.batches_per_round),
                "mean_loss": float(np.mean(avg_round_losses[self.survival_mask]))
                if self.n_survivors > 0
                else np.inf,
            }
        )

        self.logger.debug(
            f"Round {round_number} complete: {n_survivors_before} -> {self.n_survivors} survivors"
        )

    def _evaluate_initial_round(
        self,
        data_batch_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
        model: Callable,
    ) -> None:
        """Evaluate all candidates on initial batches (no elimination rounds case)."""
        for batch_idx in range(self.config.batches_per_round):
            try:
                x_batch, y_batch = next(data_batch_iterator)
            except StopIteration:
                break

            batch_losses = self._evaluate_candidates_on_batch(x_batch, y_batch, model)

            for i in range(self.n_candidates):
                if np.isfinite(batch_losses[i]):
                    self.cumulative_losses[i] += batch_losses[i]
                    self.evaluation_counts[i] += 1

            self.total_batches_evaluated += 1

    def _evaluate_candidates_on_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        model: Callable,
    ) -> np.ndarray:
        """Evaluate all surviving candidates on a single batch.

        Parameters
        ----------
        x_batch : np.ndarray
            Independent variable batch.
        y_batch : np.ndarray
            Dependent variable batch.
        model : Callable
            Model function.

        Returns
        -------
        np.ndarray
            Loss values for each candidate (inf for eliminated or failed).
        """
        x_jax = jnp.asarray(x_batch)
        y_jax = jnp.asarray(y_batch)

        losses = np.full(self.n_candidates, np.inf)

        for i in range(self.n_candidates):
            if not self.survival_mask[i]:
                continue  # Skip eliminated candidates

            try:
                params = self.candidates[i]
                predictions = model(x_jax, *params)
                residuals = y_jax - predictions

                # Compute mean squared error
                loss = float(jnp.mean(residuals**2))

                if np.isfinite(loss):
                    losses[i] = loss
                else:
                    self.numerical_failures += 1
                    self.logger.debug(f"Candidate {i}: Non-finite loss {loss}")

            except Exception as e:
                self.numerical_failures += 1
                self.logger.debug(f"Candidate {i} failed evaluation: {str(e)[:50]}")
                # Keep loss as inf

        return losses

    def _eliminate_worst(
        self,
        n_to_eliminate: int,
        round_losses: np.ndarray,
    ) -> None:
        """Eliminate the worst-performing candidates.

        Parameters
        ----------
        n_to_eliminate : int
            Number of candidates to eliminate.
        round_losses : np.ndarray
            Loss values for this round.
        """
        # Get indices of current survivors
        survivor_indices = np.where(self.survival_mask)[0]

        # Sort survivors by their round loss (worst first)
        survivor_losses = round_losses[survivor_indices]
        sorted_order = np.argsort(-survivor_losses)  # Descending (worst first)

        # Eliminate the worst n_to_eliminate
        for i in range(min(n_to_eliminate, len(sorted_order))):
            idx_to_eliminate = survivor_indices[sorted_order[i]]
            self.survival_mask[idx_to_eliminate] = False
            # Mark as eliminated in cumulative losses
            self.cumulative_losses[idx_to_eliminate] = np.inf

    def get_top_candidates(self, top_m: int = 1) -> list[np.ndarray]:
        """Get the top M candidates by cumulative loss.

        Parameters
        ----------
        top_m : int, default=1
            Number of top candidates to return.

        Returns
        -------
        list[np.ndarray]
            List of top candidate parameter arrays, sorted by loss (best first).
        """
        # Get survivors with finite cumulative loss
        valid_indices = np.where(
            self.survival_mask & np.isfinite(self.cumulative_losses)
        )[0]

        if len(valid_indices) == 0:
            # Fall back to any surviving candidate
            valid_indices = np.where(self.survival_mask)[0]
            if len(valid_indices) == 0:
                # Fall back to first candidate
                self.logger.warning("No valid survivors, returning first candidate")
                return [self.candidates[0].copy()]

        # Sort by cumulative loss
        losses = self.cumulative_losses[valid_indices]
        sorted_order = np.argsort(losses)

        # Return top M
        top_indices = valid_indices[sorted_order[:top_m]]
        return [self.candidates[i].copy() for i in top_indices]

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize tournament state to a checkpoint dictionary.

        Returns
        -------
        dict
            Checkpoint state that can be pickled and saved.

        Examples
        --------
        >>> import pickle
        >>> checkpoint = selector.to_checkpoint()
        >>> with open('tournament_checkpoint.pkl', 'wb') as f:
        ...     pickle.dump(checkpoint, f)
        """
        return {
            "candidates": self.candidates.copy(),
            "survival_mask": self.survival_mask.copy(),
            "cumulative_losses": self.cumulative_losses.copy(),
            "evaluation_counts": self.evaluation_counts.copy(),
            "current_round": self.current_round,
            "round_history": self.round_history.copy(),
            "total_batches_evaluated": self.total_batches_evaluated,
            "numerical_failures": self.numerical_failures,
            "n_candidates": self.n_candidates,
            "n_params": self.n_params,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        config: GlobalOptimizationConfig,
    ) -> "TournamentSelector":
        """Restore tournament selector from checkpoint.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint state from to_checkpoint().
        config : GlobalOptimizationConfig
            Configuration (must match original).

        Returns
        -------
        TournamentSelector
            Restored tournament selector.

        Examples
        --------
        >>> import pickle
        >>> with open('tournament_checkpoint.pkl', 'rb') as f:
        ...     checkpoint = pickle.load(f)
        >>> selector = TournamentSelector.from_checkpoint(checkpoint, config)
        """
        candidates = checkpoint["candidates"]
        selector = cls(candidates=candidates, config=config)

        # Restore state
        selector.survival_mask = checkpoint["survival_mask"].copy()
        selector.cumulative_losses = checkpoint["cumulative_losses"].copy()
        selector.evaluation_counts = checkpoint["evaluation_counts"].copy()
        selector.current_round = checkpoint["current_round"]
        selector.round_history = checkpoint["round_history"].copy()
        selector.total_batches_evaluated = checkpoint["total_batches_evaluated"]
        selector.numerical_failures = checkpoint["numerical_failures"]

        return selector

    def get_diagnostics(self) -> dict[str, Any]:
        """Get tournament diagnostics.

        Returns
        -------
        dict
            Dictionary with tournament statistics and history.
        """
        avg_loss = np.mean(self.cumulative_losses[self.survival_mask])

        return {
            "n_candidates_initial": self.n_candidates,
            "n_survivors": self.n_survivors,
            "elimination_rate": 1.0 - (self.n_survivors / self.n_candidates),
            "rounds_completed": self.current_round,
            "total_batches_evaluated": self.total_batches_evaluated,
            "numerical_failures": self.numerical_failures,
            "mean_survivor_loss": float(avg_loss) if np.isfinite(avg_loss) else None,
            "round_history": self.round_history,
        }
