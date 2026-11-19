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

import contextlib
import os
import secrets
import shutil
import tempfile
from pathlib import Path

from nlsq.cli.model_validation import validate_path
from nlsq.cli.templates import get_custom_model_template, get_workflow_template


def _prepare_dest(cwd: Path, dest_name: str, force: bool) -> Path:
    """Validate the target path and create its parent directories."""
    dest = cwd / dest_name
    if not validate_path(dest, base_dir=cwd):
        msg = f"Output path '{dest_name}' escapes the current directory."
        raise ValueError(msg)
    if dest.is_dir():
        msg = f"Output path '{dest_name}' is a directory; give a file name."
        raise ValueError(msg)
    if dest.exists() and not force:
        msg = f"File '{dest}' already exists. Use --force to overwrite."
        raise FileExistsError(msg)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Cannot create directory for '{dest_name}': {e}"
        raise ValueError(msg) from e
    return dest


def _publish_all(pairs: list[tuple[Path, Path]], force: bool) -> None:
    """Move each staged file onto its destination, all or nothing.

    Without ``force`` a destination is claimed with ``os.link`` (fails if anything
    appeared there since the preflight, never clobbers). With ``force`` the old
    file is first moved aside. Any failure undoes the installs already made.
    """
    done: list[tuple[Path, Path | None, bool]] = []  # (dest, backup, installed)
    try:
        for tmp, dest in pairs:
            backup: Path | None = None
            if force:
                if dest.is_dir() and not dest.is_symlink():
                    raise IsADirectoryError(f"'{dest}' is a directory")
                if os.path.lexists(dest):
                    backup = dest.with_name(f".nlsq-{secrets.token_hex(6)}.bak")
                    os.replace(dest, backup)
                done.append((dest, backup, False))
                os.replace(tmp, dest)
                done[-1] = (dest, backup, True)
            else:
                try:
                    os.link(tmp, dest)
                except FileExistsError:
                    raise
                except OSError:  # no hard links here: fall back to a checked rename
                    if os.path.lexists(dest):
                        raise FileExistsError(
                            f"File '{dest}' already exists."
                        ) from None
                    os.replace(tmp, dest)
                done.append((dest, None, True))
    except BaseException:
        for dest, backup, installed in reversed(done):
            with contextlib.suppress(OSError):
                if installed:
                    dest.unlink()
                if backup is not None:
                    os.replace(backup, dest)
        raise
    for _, backup, _ in done:
        if backup is not None:
            backup.unlink(missing_ok=True)


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

    # Validate every destination before writing any, so a failure on the second
    # one cannot leave the first template behind.
    jobs: list[tuple[Path, Path, str]] = []
    if copy_workflow:
        dest = _prepare_dest(cwd, output or "workflow_config.yaml", force)
        jobs.append((get_workflow_template(), dest, "workflow"))
    if copy_model:
        dest = _prepare_dest(cwd, output or "custom_model.py", force)
        jobs.append((get_custom_model_template(), dest, "custom model"))

    # Stage every copy beside its destination, then publish all of them or none.
    staged: list[tuple[Path, Path, str]] = []
    try:
        for src, dest, label in jobs:
            fd, tmp_name = tempfile.mkstemp(
                dir=dest.parent, prefix=".nlsq-", suffix=".tmp"
            )
            os.close(fd)  # short fixed-length name: never derived from the destination
            staged.append((Path(tmp_name), dest, label))
            shutil.copy(src, tmp_name)
        _publish_all([(tmp, dest) for tmp, dest, _ in staged], force)
        for _, dest, label in staged:
            copied_files.append(dest)

            if verbose:
                print(f"Copied {label} template to: {dest}")
    finally:
        for tmp, _, _ in staged:
            tmp.unlink(missing_ok=True)

    # Print summary
    if copied_files:
        print(f"Created {len(copied_files)} template file(s):")
        for f in copied_files:
            print(f"  - {f.name}")

        # Name the files that were actually written (--output renames them).
        def shown(dest: Path) -> str:
            try:
                return dest.relative_to(cwd).as_posix()
            except ValueError:  # reached through a symlinked prefix
                return str(dest)

        names = {label: shown(dest) for _, dest, label in staged}
        print("\nNext steps:")
        if copy_workflow:
            print(
                f"  1. Edit {names['workflow']} with your data path and model settings",
            )
            print(f"  2. Run: nlsq fit {names['workflow']}")
        if copy_model:
            print(f"  - Edit {names['custom model']} to define your model function")
            print(
                "  - Reference it in your workflow: model.type: custom, "
                f"model.path: {names['custom model']}",
            )

    return copied_files
