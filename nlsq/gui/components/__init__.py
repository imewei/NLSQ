"""NLSQ GUI Components - Reusable Streamlit widgets.

This package provides reusable UI components for the NLSQ Streamlit GUI.

Components:
- column_selector: Column role assignment for data loading
- code_editor: Custom model code editor with syntax validation
- model_preview: Model equation and parameter display
- param_config: Parameter configuration for fitting
- advanced_options: Advanced fitting options with tabs
- live_cost_plot: Real-time cost function plotting during optimization
- iteration_table: Parameter values at each iteration
- param_results: Parameter table with uncertainties and confidence intervals
- fit_statistics: R^2, RMSE, MAE, AIC, BIC display
- plotly_fit_plot: Data and fitted curve visualization
- plotly_residuals: Residual analysis plots
- plotly_histogram: Residual distribution histogram
"""

from nlsq.gui.components.advanced_options import (
    FITTING_METHODS,
    LOSS_FUNCTIONS,
    SAMPLERS,
    SUMMARY_FORMATS,
    get_batch_tab_config,
    get_fitting_tab_config,
    get_hpc_tab_config,
    get_multistart_tab_config,
    get_streaming_tab_config,
    render_advanced_options,
    render_batch_tab,
    render_fitting_tab,
    render_hpc_tab,
    render_multistart_tab,
    render_streaming_tab,
    validate_chunk_size,
    validate_max_iterations,
    validate_n_starts,
)
from nlsq.gui.components.code_editor import (
    get_code_validation_status,
    get_default_model_template,
    get_uploaded_file_content,
    render_code_editor,
    render_file_upload,
    validate_code_syntax,
)
from nlsq.gui.components.column_selector import (
    ROLE_COLORS,
    ROLE_DISPLAY_NAMES,
    compute_data_preview_stats,
    create_column_assignments,
    get_available_roles,
    get_column_color,
    get_required_roles,
    get_role_display_name,
    render_column_selector,
    validate_column_selections,
)
from nlsq.gui.components.fit_statistics import (
    format_convergence_info,
    format_statistics,
    get_fit_quality_label,
    render_fit_statistics,
    render_statistics_summary,
)
from nlsq.gui.components.iteration_table import (
    clear_iteration_history,
    create_iteration_history,
    format_iteration_table,
    format_parameter_change,
    get_table_display_config,
    limit_history_size,
    render_convergence_summary,
    render_iteration_table,
    update_iteration_history,
)
from nlsq.gui.components.live_cost_plot import (
    clear_cost_history,
    create_cost_history,
    create_cost_plot_figure,
    create_empty_cost_plot,
    get_cost_plot_summary,
    get_plot_config,
    render_live_cost_plot,
    update_cost_history,
)
from nlsq.gui.components.model_preview import (
    format_parameter_list,
    get_equation_for_model,
    get_model_summary,
    render_equation_display,
    render_model_capabilities,
    render_model_preview,
    render_parameter_table,
)
from nlsq.gui.components.param_config import (
    TRANSFORM_OPTIONS,
    create_param_config_dict,
    estimate_p0_for_model,
    format_p0_display,
    get_default_p0_value,
    get_param_names_from_model,
    render_param_config,
    render_single_param_row,
    validate_bounds,
    validate_p0_input,
)
from nlsq.gui.components.param_results import (
    compute_confidence_intervals,
    format_confidence_intervals,
    format_parameter_table,
    render_parameter_results,
)
from nlsq.gui.components.plotly_fit_plot import (
    create_fit_plot,
    create_fit_plot_from_result,
    render_fit_plot,
)
from nlsq.gui.components.plotly_histogram import (
    compute_normality_tests,
    create_histogram_from_result,
    create_residuals_histogram,
    render_residuals_histogram,
)
from nlsq.gui.components.plotly_residuals import (
    create_residuals_plot,
    create_residuals_plot_from_result,
    render_residuals_plot,
)

__all__ = [
    # Advanced options exports
    "FITTING_METHODS",
    "LOSS_FUNCTIONS",
    # Column selector exports
    "ROLE_COLORS",
    "ROLE_DISPLAY_NAMES",
    "SAMPLERS",
    "SUMMARY_FORMATS",
    # Parameter config exports
    "TRANSFORM_OPTIONS",
    # Live cost plot exports
    "clear_cost_history",
    # Iteration table exports
    "clear_iteration_history",
    # Parameter results exports
    "compute_confidence_intervals",
    "compute_data_preview_stats",
    # Plotly histogram exports
    "compute_normality_tests",
    "create_column_assignments",
    "create_cost_history",
    "create_cost_plot_figure",
    "create_empty_cost_plot",
    # Plotly fit plot exports
    "create_fit_plot",
    "create_fit_plot_from_result",
    "create_histogram_from_result",
    "create_iteration_history",
    "create_param_config_dict",
    "create_residuals_histogram",
    # Plotly residuals exports
    "create_residuals_plot",
    "create_residuals_plot_from_result",
    "estimate_p0_for_model",
    "format_confidence_intervals",
    # Fit statistics exports
    "format_convergence_info",
    "format_iteration_table",
    "format_p0_display",
    "format_parameter_change",
    # Model preview exports
    "format_parameter_list",
    "format_parameter_table",
    "format_statistics",
    "get_available_roles",
    "get_batch_tab_config",
    # Code editor exports
    "get_code_validation_status",
    "get_column_color",
    "get_cost_plot_summary",
    "get_default_model_template",
    "get_default_p0_value",
    "get_equation_for_model",
    "get_fit_quality_label",
    "get_fitting_tab_config",
    "get_hpc_tab_config",
    "get_model_summary",
    "get_multistart_tab_config",
    "get_param_names_from_model",
    "get_plot_config",
    "get_required_roles",
    "get_role_display_name",
    "get_streaming_tab_config",
    "get_table_display_config",
    "get_uploaded_file_content",
    "limit_history_size",
    "render_advanced_options",
    "render_batch_tab",
    "render_code_editor",
    "render_column_selector",
    "render_convergence_summary",
    "render_equation_display",
    "render_file_upload",
    "render_fit_plot",
    "render_fit_statistics",
    "render_fitting_tab",
    "render_hpc_tab",
    "render_iteration_table",
    "render_live_cost_plot",
    "render_model_capabilities",
    "render_model_preview",
    "render_multistart_tab",
    "render_param_config",
    "render_parameter_results",
    "render_parameter_table",
    "render_residuals_histogram",
    "render_residuals_plot",
    "render_single_param_row",
    "render_statistics_summary",
    "render_streaming_tab",
    "update_cost_history",
    "update_iteration_history",
    "validate_bounds",
    "validate_chunk_size",
    "validate_code_syntax",
    "validate_column_selections",
    "validate_max_iterations",
    "validate_n_starts",
    "validate_p0_input",
]
