"""Transformation pipeline for composing notebook transformations."""

import logging
from pathlib import Path

from .core import read_notebook, write_notebook
from .transformations.base import NotebookTransformer

logger = logging.getLogger(__name__)


class TransformationPipeline:
    """Composes multiple transformations with rollback support.

    Implements the Chain of Responsibility pattern to apply a sequence
    of transformations to notebook cells. Provides atomic commit semantics
    with automatic rollback on errors.
    """

    def __init__(self, transformers: list[NotebookTransformer]):
        """Initialize pipeline with transformers.

        Args:
            transformers: List of transformers to apply in order
        """
        self.transformers = transformers

    def run(
        self, notebook_path: Path, backup: bool = False, dry_run: bool = False
    ) -> dict[str, dict]:
        """Run all transformations with atomic commit.

        Args:
            notebook_path: Path to notebook file
            backup: Create .bak file before writing
            dry_run: Don't write changes, just return stats

        Returns:
            Dictionary mapping transformer names to their stats

        Raises:
            Exception: If any transformation fails
        """
        # Load notebook
        notebook = read_notebook(notebook_path)
        if notebook is None:
            logger.warning(f"Skipping {notebook_path} due to read error")
            return {}

        cells = notebook.get("cells", [])
        if not cells:
            return {}

        # Apply transformations with rollback support
        original = cells.copy()
        all_stats = {}

        try:
            for transformer in self.transformers:
                # Skip if transformer says not needed
                if not transformer.should_apply(cells):
                    logger.debug(
                        f"Skipping {transformer.name()} - should_apply returned False"
                    )
                    all_stats[transformer.name()] = {}
                    continue

                # Apply transformation
                logger.debug(
                    f"Applying {transformer.name()}: {transformer.description()}"
                )
                cells, stats = transformer.transform(cells)

                # Validate result
                transformer.validate_result(original, cells)

                # Track stats
                all_stats[transformer.name()] = stats
                logger.debug(f"{transformer.name()} stats: {stats}")

        except Exception as e:
            # Rollback on error
            logger.error(f"Transformation failed: {e}")
            logger.info("Rolling back to original cells")
            cells = original
            raise

        # Update notebook
        notebook["cells"] = cells

        # Save if not dry run
        if not dry_run:
            success = write_notebook(notebook_path, notebook, backup=backup)
            if not success:
                logger.warning(f"Failed to save {notebook_path}")
                raise RuntimeError(f"Failed to write notebook: {notebook_path}")

        return all_stats

    def get_transformers(self) -> list[NotebookTransformer]:
        """Get list of transformers in pipeline.

        Returns:
            List of transformer instances
        """
        return self.transformers.copy()

    def add_transformer(self, transformer: NotebookTransformer) -> None:
        """Add a transformer to the end of the pipeline.

        Args:
            transformer: Transformer instance to add
        """
        self.transformers.append(transformer)

    def describe(self) -> list[dict[str, str]]:
        """Get description of all transformers in pipeline.

        Returns:
            List of dicts with 'name' and 'description' keys
        """
        return [
            {"name": t.name(), "description": t.description()}
            for t in self.transformers
        ]
