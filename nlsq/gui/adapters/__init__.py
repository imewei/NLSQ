"""NLSQ GUI Adapters - Wrappers for NLSQ API integration."""

from nlsq.gui.adapters.config_adapter import (
    export_yaml_config,
    import_cli_workflow,
    load_yaml_config,
    merge_configs,
    validate_yaml_config,
)
from nlsq.gui.adapters.data_adapter import (
    ValidationResult,
    compute_statistics,
    detect_columns,
    detect_delimiter,
    is_2d_mode,
    load_from_clipboard,
    load_from_file,
    validate_data,
)
from nlsq.gui.adapters.export_adapter import (
    create_session_bundle,
    export_csv,
    export_json,
    export_plotly_html,
)
from nlsq.gui.adapters.fit_adapter import (
    FitConfig,
    ProgressCallback,
    create_fit_config_from_state,
    execute_fit,
    extract_confidence_intervals,
    extract_convergence_info,
    extract_fit_statistics,
    extract_parameter_uncertainties,
    get_recommended_chunk_size,
    is_large_dataset,
    validate_fit_inputs,
)
from nlsq.gui.adapters.model_adapter import (
    get_latex_equation,
    get_model,
    get_model_info,
    get_polynomial_latex,
    list_builtin_models,
    list_functions_in_module,
    load_custom_model_file,
    parse_custom_model_string,
    validate_jit_compatibility,
)

__all__ = [
    # Fit adapter
    "FitConfig",
    "ProgressCallback",
    # Data adapter
    "ValidationResult",
    "compute_statistics",
    "create_fit_config_from_state",
    # Export adapter
    "create_session_bundle",
    "detect_columns",
    "detect_delimiter",
    "execute_fit",
    "export_csv",
    "export_json",
    "export_plotly_html",
    # Config adapter
    "export_yaml_config",
    "extract_confidence_intervals",
    "extract_convergence_info",
    "extract_fit_statistics",
    "extract_parameter_uncertainties",
    # Model adapter
    "get_latex_equation",
    "get_model",
    "get_model_info",
    "get_polynomial_latex",
    "get_recommended_chunk_size",
    "import_cli_workflow",
    "is_2d_mode",
    "is_large_dataset",
    "list_builtin_models",
    "list_functions_in_module",
    "load_custom_model_file",
    "load_from_clipboard",
    "load_from_file",
    "load_yaml_config",
    "merge_configs",
    "parse_custom_model_string",
    "validate_data",
    "validate_fit_inputs",
    "validate_jit_compatibility",
    "validate_yaml_config",
]
