"""Base class for notebook transformations using Strategy pattern."""

from abc import ABC, abstractmethod

from ..types import NotebookCell


class NotebookTransformer(ABC):
    """Base class for notebook transformations.

    Implements the Strategy pattern to allow different transformation algorithms
    to be composed into a processing pipeline.

    Each transformer should be:
    - Stateless: Can be reused across multiple notebooks
    - Idempotent: Running twice produces same result as running once
    - Pure: Only modifies notebook cells, no side effects

    Subclasses must implement:
    - transform(): Core transformation logic
    - name(): Unique identifier for this transformation
    - description(): Human-readable description
    """

    @abstractmethod
    def transform(
        self, cells: list[NotebookCell]
    ) -> tuple[list[NotebookCell], dict[str, int]]:
        """Transform notebook cells.

        Args:
            cells: List of notebook cells to transform

        Returns:
            Tuple of (modified_cells, stats_dict)
            - modified_cells: New list with transformations applied
            - stats_dict: Statistics about what was changed

        Note:
            Should return NEW list, not mutate input cells
        """

    @abstractmethod
    def name(self) -> str:
        """Return unique transformation name.

        Returns:
            Transformation identifier (e.g., "matplotlib_inline")
        """

    @abstractmethod
    def description(self) -> str:
        """Return human-readable transformation description.

        Returns:
            Description of what this transformation does
        """

    def should_apply(self, cells: list[NotebookCell]) -> bool:
        """Check if transformation should be applied.

        Default implementation always returns True.
        Override to skip transformation when not needed.

        Args:
            cells: Notebook cells to check

        Returns:
            True if transformation should run
        """
        return True

    def validate_result(
        self, original: list[NotebookCell], transformed: list[NotebookCell]
    ) -> bool:
        """Validate transformation result.

        Default implementation checks that result is a list.
        Override for custom validation logic.

        Args:
            original: Original cells before transformation
            transformed: Cells after transformation

        Returns:
            True if transformation result is valid

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(transformed, list):
            raise ValueError(f"{self.name()}: Result must be a list")

        return True
