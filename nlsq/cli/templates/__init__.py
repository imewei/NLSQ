"""Templates for NLSQ CLI.

This package contains template files for users to create custom models
and workflow configurations.

Templates:
    - custom_model_template.py: Template for creating custom model functions
    - workflow_config_template.yaml: Comprehensive workflow configuration template

Usage:
    from nlsq.cli.templates import get_template_path, get_custom_model_template, get_workflow_template

    # Get path to a template file
    model_path = get_template_path("custom_model_template.py")
    workflow_path = get_template_path("workflow_config_template.yaml")

    # Copy templates to your project
    import shutil
    shutil.copy(get_custom_model_template(), "my_model.py")
    shutil.copy(get_workflow_template(), "my_workflow.yaml")
"""

from pathlib import Path

__all__ = [
    "TEMPLATES_DIR",
    "get_custom_model_template",
    "get_template_path",
    "get_workflow_template",
]

TEMPLATES_DIR = Path(__file__).parent


def get_template_path(template_name: str) -> Path:
    """Get the absolute path to a template file.

    Parameters
    ----------
    template_name : str
        Name of the template file (e.g., "custom_model_template.py")

    Returns
    -------
    Path
        Absolute path to the template file

    Raises
    ------
    FileNotFoundError
        If the template file does not exist
    """
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        available = [
            f.name
            for f in TEMPLATES_DIR.iterdir()
            if f.is_file() and not f.name.startswith("_")
        ]
        msg = f"Template '{template_name}' not found. Available templates: {available}"
        raise FileNotFoundError(msg)
    return template_path


def get_custom_model_template() -> Path:
    """Get the path to the custom model template.

    Returns
    -------
    Path
        Absolute path to custom_model_template.py
    """
    return get_template_path("custom_model_template.py")


def get_workflow_template() -> Path:
    """Get the path to the workflow configuration template.

    Returns
    -------
    Path
        Absolute path to workflow_config_template.yaml
    """
    return get_template_path("workflow_config_template.yaml")
