"""CLI command for copying configuration templates to user's project.

This module provides the `nlsq config` command for copying workflow and
custom model templates to the current directory.

Example Usage
-------------
From command line:
    $ nlsq config                    # Copy both templates
    $ nlsq config --workflow         # Copy only workflow template
    $ nlsq config --model            # Copy only custom model template
    $ nlsq config -o my_workflow.yaml --workflow  # Custom output name
"""

import shutil
from pathlib import Path

from nlsq.cli.templates import get_custom_model_template, get_workflow_template


def run_config(
    *,
    workflow: bool = False,
    model: bool = False,
    output: str | None = None,
    force: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Copy configuration templates to the current directory.

    Parameters
    ----------
    workflow : bool
        Copy only the workflow configuration template.
    model : bool
        Copy only the custom model template.
    output : str, optional
        Custom output filename (only valid when copying a single template).
    force : bool
        Overwrite existing files without prompting.
    verbose : bool
        Enable verbose output.

    Returns
    -------
    list[Path]
        List of paths to the copied template files.

    Raises
    ------
    FileExistsError
        If a target file exists and force is False.
    ValueError
        If output is specified but multiple templates are being copied.
    """
    # If neither specified, copy both
    copy_workflow = workflow or (not workflow and not model)
    copy_model = model or (not workflow and not model)

    # Validate output option
    if output and copy_workflow and copy_model:
        msg = "Cannot use --output when copying multiple templates. Use --workflow or --model to select one."
        raise ValueError(msg)

    copied_files: list[Path] = []
    cwd = Path.cwd()

    # Copy workflow template
    if copy_workflow:
        src = get_workflow_template()
        dest_name = output if output else "workflow_config.yaml"
        dest = cwd / dest_name

        if dest.exists() and not force:
            msg = f"File '{dest}' already exists. Use --force to overwrite."
            raise FileExistsError(msg)

        shutil.copy(src, dest)
        copied_files.append(dest)

        if verbose:
            print(f"Copied workflow template to: {dest}")

    # Copy custom model template
    if copy_model:
        src = get_custom_model_template()
        dest_name = output if output else "custom_model.py"
        dest = cwd / dest_name

        if dest.exists() and not force:
            msg = f"File '{dest}' already exists. Use --force to overwrite."
            raise FileExistsError(msg)

        shutil.copy(src, dest)
        copied_files.append(dest)

        if verbose:
            print(f"Copied custom model template to: {dest}")

    # Print summary
    if copied_files:
        print(f"Created {len(copied_files)} template file(s):")
        for f in copied_files:
            print(f"  - {f.name}")

        print("\nNext steps:")
        if copy_workflow:
            print(
                "  1. Edit workflow_config.yaml with your data path and model settings"
            )
            print("  2. Run: nlsq fit workflow_config.yaml")
        if copy_model:
            print("  - Edit custom_model.py to define your model function")
            print(
                "  - Reference it in your workflow: model.type: custom, model.path: custom_model.py"
            )

    return copied_files
