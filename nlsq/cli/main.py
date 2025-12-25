r"""CLI entry point for NLSQ workflow commands.

This module provides the main command-line interface for NLSQ, supporting:
- `nlsq gui` - Launch the interactive Streamlit GUI
- `nlsq fit workflow.yaml` - Execute single curve fit from YAML config
- `nlsq batch w1.yaml w2.yaml ...` - Execute parallel batch fitting
- `nlsq info` - Display system and environment information
- `nlsq config` - Copy configuration templates to current directory

Example Usage
-------------
From command line::

    $ nlsq gui
    $ nlsq fit workflow.yaml
    $ nlsq fit workflow.yaml --output results.json
    $ nlsq fit workflow.yaml --stdout
    $ nlsq batch configs/\*.yaml --workers 4
    $ nlsq info
    $ nlsq config
    $ nlsq config --workflow
    $ nlsq config --model -o my_model.py
    $ nlsq --version
    $ nlsq --verbose fit workflow.yaml
"""

import argparse
import sys
from typing import Any

import nlsq
from nlsq.cli.errors import CLIError, ConfigError, DataLoadError, FitError, ModelError


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with fit, batch, gui, and info subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="nlsq",
        description="NLSQ: GPU/TPU-accelerated nonlinear least squares curve fitting",
        epilog="For more information, visit https://github.com/imewei/NLSQ",
    )

    # Top-level arguments
    parser.add_argument(
        "--version",
        action="version",
        version=f"nlsq {nlsq.__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        help="Use 'nlsq <command> --help' for more information",
    )

    # --- nlsq gui ---
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the interactive Streamlit GUI",
        description="Start the NLSQ graphical user interface for interactive curve fitting",
    )
    gui_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit server on (default: 8501)",
    )
    gui_parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: true)",
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically",
    )

    # --- nlsq fit ---
    fit_parser = subparsers.add_parser(
        "fit",
        help="Execute single curve fit from YAML workflow configuration",
        description="Load workflow configuration from YAML and execute curve fitting",
    )
    fit_parser.add_argument(
        "workflow",
        type=str,
        help="Path to workflow YAML configuration file",
    )
    fit_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Override export.results_file path",
    )
    fit_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output results as JSON to stdout (for piping)",
    )

    # --- nlsq batch ---
    batch_parser = subparsers.add_parser(
        "batch",
        help="Execute parallel batch fitting from multiple YAML files",
        description="Process multiple workflow configurations in parallel",
    )
    batch_parser.add_argument(
        "workflows",
        type=str,
        nargs="+",
        help="Paths to workflow YAML configuration files",
    )
    batch_parser.add_argument(
        "-s",
        "--summary",
        type=str,
        default=None,
        help="Path for aggregate summary file",
    )
    batch_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto-detect)",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing on individual workflow failures (default: true)",
    )

    # --- nlsq info ---
    subparsers.add_parser(
        "info",
        help="Display system and environment information",
        description="Show NLSQ version, Python, JAX backend, GPU info, and builtin models",
    )

    # --- nlsq config ---
    config_parser = subparsers.add_parser(
        "config",
        help="Copy configuration templates to current directory",
        description="Copy workflow and/or custom model templates to start a new project",
    )
    config_parser.add_argument(
        "--workflow",
        action="store_true",
        help="Copy only the workflow configuration template (workflow_config.yaml)",
    )
    config_parser.add_argument(
        "--model",
        action="store_true",
        help="Copy only the custom model template (custom_model.py)",
    )
    config_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Custom output filename (only valid with --workflow or --model)",
    )
    config_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting",
    )

    return parser


def handle_gui(args: argparse.Namespace) -> int:
    """Handle the 'gui' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    try:
        import subprocess
        from pathlib import Path

        # Find the app.py location
        import nlsq.gui

        gui_module_path = Path(nlsq.gui.__file__).parent
        app_path = gui_module_path / "app.py"

        if not app_path.exists():
            print(
                "Error: GUI app.py not found. Please reinstall NLSQ.", file=sys.stderr
            )
            return 1

        # Determine whether to open browser
        open_browser = args.browser and not args.no_browser

        # Build streamlit command
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(args.port),
        ]

        if not open_browser:
            cmd.extend(["--server.headless", "true"])

        # Print launch message
        print(f"Launching NLSQ GUI on port {args.port}...")
        if open_browser:
            print(f"Opening browser at http://localhost:{args.port}")
        else:
            print(f"Access the GUI at http://localhost:{args.port}")

        # Run streamlit
        result = subprocess.run(cmd, check=False)
        return result.returncode

    except ImportError as e:
        print(
            "\nError: GUI dependencies not installed.\n"
            "Install with: pip install nlsq[gui]\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError:
        print(
            "\nError: Streamlit not found.\nInstall with: pip install streamlit",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nGUI server stopped.")
        return 0


def handle_fit(args: argparse.Namespace) -> int:
    """Handle the 'fit' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    from nlsq.cli.commands import fit

    try:
        result = fit.run_fit(
            workflow_path=args.workflow,
            output_override=args.output,
            stdout=args.stdout,
            verbose=args.verbose if hasattr(args, "verbose") else False,
        )

        if result is None:
            return 1

        return 0

    except ConfigError as e:
        _print_error("Configuration Error", e)
        return 1
    except DataLoadError as e:
        _print_error("Data Loading Error", e)
        return 1
    except ModelError as e:
        _print_error("Model Error", e)
        return 1
    except FitError as e:
        _print_error("Fitting Error", e)
        return 1
    except CLIError as e:
        _print_error("CLI Error", e)
        return 1


def handle_batch(args: argparse.Namespace) -> int:
    """Handle the 'batch' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    from nlsq.cli.commands import batch

    try:
        results = batch.run_batch(
            workflow_paths=args.workflows,
            summary_file=args.summary,
            max_workers=args.workers,
            continue_on_error=args.continue_on_error,
            verbose=args.verbose if hasattr(args, "verbose") else False,
        )

        # Return non-zero if any workflows failed
        failures = [r for r in results if r.get("status") == "failed"]
        if failures:
            print(f"\nBatch completed with {len(failures)} failure(s)")
            return 1

        return 0

    except CLIError as e:
        _print_error("CLI Error", e)
        return 1


def handle_info(args: argparse.Namespace) -> int:
    """Handle the 'info' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (always 0 for info command).
    """
    from nlsq.cli.commands import info

    info.run_info(verbose=args.verbose if hasattr(args, "verbose") else False)
    return 0


def handle_config(args: argparse.Namespace) -> int:
    """Handle the 'config' subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    from nlsq.cli.commands import config

    try:
        config.run_config(
            workflow=args.workflow,
            model=args.model,
            output=args.output,
            force=args.force,
            verbose=args.verbose if hasattr(args, "verbose") else False,
        )
        return 0

    except FileExistsError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def _print_error(error_type: str, error: CLIError) -> None:
    """Print a formatted error message to stderr.

    Parameters
    ----------
    error_type : str
        Type of error for the header.
    error : CLIError
        The error object with message and context.
    """
    print(f"\nError: {error_type}", file=sys.stderr)
    print(f"  {error.message}", file=sys.stderr)

    if error.context:
        for key, value in error.context.items():
            print(f"  {key}: {value}", file=sys.stderr)

    if error.suggestion:
        print(f"\nSuggestion: {error.suggestion}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the NLSQ CLI.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    Examples
    --------
    >>> main(["gui"])
    0
    >>> main(["fit", "workflow.yaml"])
    0
    >>> main(["info"])
    0
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # No command specified - print help
    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to appropriate handler
    handlers: dict[str, Any] = {
        "gui": handle_gui,
        "fit": handle_fit,
        "batch": handle_batch,
        "info": handle_info,
        "config": handle_config,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
