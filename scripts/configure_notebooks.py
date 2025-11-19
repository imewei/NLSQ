#!/usr/bin/env python3
"""
Unified CLI for notebook transformations.

This script provides a modern command-line interface for configuring
Jupyter notebooks with various transformations. It uses the transformation
pipeline architecture for composable, testable modifications.

Usage:
    configure_notebooks.py [OPTIONS]

Examples:
    # Apply all transformations
    configure_notebooks.py

    # Apply specific transformations only
    configure_notebooks.py --transform matplotlib --transform imports

    # Dry run to see what would change
    configure_notebooks.py --dry-run

    # Process specific directory
    configure_notebooks.py --dir examples/notebooks/04_gallery

    # Enable parallel processing
    configure_notebooks.py --parallel --workers 4
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from notebook_utils.pipeline import TransformationPipeline
from notebook_utils.tracking import ProcessingTracker
from notebook_utils.transformations import (
    IPythonDisplayImportTransformer,
    MatplotlibInlineTransformer,
    PltShowReplacementTransformer,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_pipeline(transformations: list[str]) -> TransformationPipeline:
    """Build transformation pipeline from list of transformation names.

    Args:
        transformations: List of transformation names or 'all'

    Returns:
        TransformationPipeline with requested transformers
    """
    available_transformers = {
        "matplotlib": MatplotlibInlineTransformer(),
        "imports": IPythonDisplayImportTransformer(),
        "plt-show": PltShowReplacementTransformer(),
    }

    if "all" in transformations:
        transformers = list(available_transformers.values())
    else:
        transformers = [available_transformers[name] for name in transformations]

    return TransformationPipeline(transformers)


def process_single_notebook(
    notebook_path: Path,
    pipeline: TransformationPipeline,
    dry_run: bool,
    backup: bool,
    notebooks_dir: Path,
) -> tuple[Path, dict[str, dict], bool]:
    """Process a single notebook (for parallel execution).

    Args:
        notebook_path: Path to notebook
        pipeline: Transformation pipeline
        dry_run: Don't write changes
        backup: Create backup files
        notebooks_dir: Base directory for relative paths

    Returns:
        Tuple of (path, stats, success)
    """
    try:
        stats = pipeline.run(notebook_path, backup=backup, dry_run=dry_run)
        return notebook_path, stats, True
    except Exception as e:
        logger.exception(f"Error processing {notebook_path}: {e}")
        return notebook_path, {}, False


@click.command()
@click.option(
    "--dir",
    default="examples/notebooks",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing notebooks to process",
)
@click.option(
    "--transform",
    "-t",
    "transforms",
    multiple=True,
    type=click.Choice(
        ["matplotlib", "imports", "plt-show", "all"], case_sensitive=False
    ),
    default=["all"],
    help="Transformations to apply (can specify multiple)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
@click.option(
    "--backup/--no-backup",
    default=False,
    help="Create .bak files before modifying",
)
@click.option(
    "--parallel/--sequential",
    default=False,
    help="Process notebooks in parallel",
)
@click.option(
    "--workers",
    default=4,
    type=int,
    help="Number of parallel workers (only with --parallel)",
)
@click.option(
    "--pattern",
    default="*.ipynb",
    help="Glob pattern for notebook files",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--incremental/--full",
    default=False,
    help="Only process notebooks that have changed (uses checksum tracking)",
)
def main(
    dir: Path,
    transforms: tuple[str],
    dry_run: bool,
    backup: bool,
    parallel: bool,
    workers: int,
    pattern: str,
    verbose: bool,
    incremental: bool,
):
    """Configure Jupyter notebooks with specified transformations.

    This unified CLI provides access to all notebook configuration operations:
    - Adding matplotlib inline configuration
    - Injecting IPython.display imports
    - Replacing plt.show() with display pattern

    By default, all transformations are applied. Use --transform to select
    specific ones.
    """
    # Configure logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("notebook_utils").setLevel(logging.DEBUG)

    # Resolve directory
    if not dir.is_absolute():
        repo_root = Path(__file__).parent.parent
        dir = repo_root / dir

    if not dir.exists():
        click.echo(f"❌ Directory not found: {dir}", err=True)
        sys.exit(1)

    # Find notebooks
    notebooks = sorted(dir.rglob(pattern))

    if not notebooks:
        click.echo(f"❌ No notebooks found matching '{pattern}' in {dir}", err=True)
        sys.exit(1)

    # Build pipeline
    pipeline = build_pipeline(list(transforms))

    # Initialize tracking for incremental mode
    tracker = None
    if incremental:
        tracker = ProcessingTracker()
        transform_names = [t.name() for t in pipeline.get_transformers()]

        # Filter to only notebooks that need processing
        original_count = len(notebooks)
        notebooks = [
            nb for nb in notebooks if tracker.needs_processing(nb, transform_names)
        ]

        if len(notebooks) < original_count:
            click.echo(
                f"ℹ️  Incremental mode: Skipping {original_count - len(notebooks)} "
                f"unchanged notebook(s)"
            )

        if not notebooks:
            click.echo("✅ All notebooks already up-to-date!")
            return

    # Display configuration
    click.echo("🔍 Configuration:")
    click.echo(f"   Directory: {dir}")
    click.echo(f"   Notebooks: {len(notebooks)}")
    click.echo("   Transformations:")
    for t in pipeline.describe():
        click.echo(f"     - {t['name']}: {t['description']}")
    if dry_run:
        click.echo("   Mode: DRY RUN (no changes will be made)")
    if backup:
        click.echo("   Backup: Enabled (.bak files will be created)")
    if parallel:
        click.echo(f"   Processing: Parallel ({workers} workers)")
    click.echo()

    # Process notebooks
    total_modified = 0
    total_processed = 0
    errors = []

    if parallel:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_single_notebook,
                    nb_path,
                    pipeline,
                    dry_run,
                    backup,
                    dir,
                ): nb_path
                for nb_path in notebooks
            }

            # Process results with progress
            with click.progressbar(
                as_completed(futures),
                length=len(notebooks),
                label="Processing notebooks",
            ) as bar:
                for future in bar:
                    nb_path, stats, success = future.result()
                    total_processed += 1

                    if not success:
                        errors.append(nb_path)
                        continue

                    # Check if any modifications were made
                    has_changes = any(
                        sum(s.values()) > 0
                        for s in stats.values()
                        if isinstance(s, dict)
                    )
                    if has_changes:
                        total_modified += 1

                    # Mark as processed in incremental mode
                    if tracker and not dry_run:
                        transform_names = [
                            t.name() for t in pipeline.get_transformers()
                        ]
                        tracker.mark_processed(nb_path, transform_names, stats)

    else:
        # Sequential processing
        with click.progressbar(notebooks, label="Processing notebooks") as bar:
            for nb_path in bar:
                total_processed += 1

                try:
                    stats = pipeline.run(nb_path, backup=backup, dry_run=dry_run)

                    # Check if any modifications were made
                    has_changes = any(
                        sum(s.values()) > 0
                        for s in stats.values()
                        if isinstance(s, dict)
                    )
                    if has_changes:
                        total_modified += 1

                    # Mark as processed in incremental mode
                    if tracker and not dry_run:
                        transform_names = [
                            t.name() for t in pipeline.get_transformers()
                        ]
                        tracker.mark_processed(nb_path, transform_names, stats)

                except Exception as e:
                    logger.exception(f"Error processing {nb_path}: {e}")
                    errors.append(nb_path)

    # Print summary
    click.echo()
    click.echo("=" * 60)
    click.echo("📊 Summary:")
    click.echo("=" * 60)
    click.echo(f"Notebooks processed:  {total_processed}")
    click.echo(f"Notebooks modified:   {total_modified}")
    click.echo(f"Errors:               {len(errors)}")
    click.echo("=" * 60)

    if errors:
        click.echo()
        click.echo("❌ Errors occurred in the following notebooks:", err=True)
        for err_path in errors:
            click.echo(f"   - {err_path.relative_to(dir)}", err=True)
        sys.exit(1)
    elif total_modified > 0:
        if dry_run:
            click.echo()
            click.echo(
                f"✅ Dry run complete! {total_modified} notebook(s) would be modified."
            )
            click.echo("   Run without --dry-run to apply changes.")
        else:
            click.echo()
            click.echo(f"✅ Successfully configured {total_modified} notebook(s)!")
    else:
        click.echo()
        click.echo("✅ All notebooks already properly configured!")


if __name__ == "__main__":
    main()
