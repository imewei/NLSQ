"""Workflow runner module for NLSQ CLI.

This module provides the WorkflowRunner class that orchestrates the complete
curve fitting workflow: data loading, model resolution, parameter extraction,
fitting execution, and result export.

Example Usage
-------------
>>> from nlsq.cli.workflow_runner import WorkflowRunner
>>>
>>> config = {
...     "data": {"input_file": "data.txt", "format": "ascii"},
...     "model": {"type": "builtin", "name": "exponential_decay"},
...     "fitting": {"p0": "auto", "method": "trf"},
...     "export": {"results_file": "results.json", "format": "json"},
... }
>>> runner = WorkflowRunner()
>>> result = runner.run(config)
>>> print(result["popt"])
"""

from typing import Any

import numpy as np

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import CLIError, DataLoadError, FitError, ModelError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.cli.result_exporter import ResultExporter


class WorkflowRunner:
    """Runner for curve fitting workflows.

    Orchestrates the complete workflow execution:
    1. Load data using DataLoader
    2. Resolve model using ModelRegistry
    3. Extract fitting parameters from config
    4. Execute curve fit using nlsq.curve_fit()
    5. Export results using ResultExporter

    Attributes
    ----------
    data_loader : DataLoader
        Instance of DataLoader for data file loading.
    model_registry : ModelRegistry
        Instance of ModelRegistry for model resolution.
    result_exporter : ResultExporter
        Instance of ResultExporter for result export.

    Methods
    -------
    run(config)
        Execute complete workflow and return result dict.

    Examples
    --------
    >>> runner = WorkflowRunner()
    >>> config = {
    ...     "data": {"input_file": "data.txt", "format": "ascii"},
    ...     "model": {"type": "builtin", "name": "linear"},
    ...     "fitting": {"p0": [1.0, 0.0]},
    ...     "export": {"results_file": "results.json"},
    ... }
    >>> result = runner.run(config)
    >>> print(f"Fitted parameters: {result['popt']}")
    """

    def __init__(self) -> None:
        """Initialize WorkflowRunner with component instances."""
        self.data_loader = DataLoader()
        self.model_registry = ModelRegistry()
        self.result_exporter = ResultExporter()

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute complete curve fitting workflow.

        Parameters
        ----------
        config : dict
            Workflow configuration dictionary containing:
            - data: Data loading configuration
            - model: Model configuration
            - fitting: Fitting parameters (p0, bounds, method, etc.)
            - export: Export configuration (optional)
            - metadata: Workflow metadata (optional)
            - validation: Data validation settings (optional)

        Returns
        -------
        dict
            Fit result dictionary containing:
            - popt: Fitted parameters (list)
            - pcov: Covariance matrix (list of lists)
            - success: bool indicating fit success
            - message: Convergence message
            - nfev: Number of function evaluations
            - Additional statistics and metadata

        Raises
        ------
        DataLoadError
            If data loading fails.
        ModelError
            If model resolution fails.
        FitError
            If curve fitting fails.
        CLIError
            If any other workflow error occurs.
        """
        # Step 1: Load data
        xdata, ydata, sigma = self._load_data(config)

        # Step 2: Resolve model
        model = self._resolve_model(config)

        # Step 3: Extract fitting parameters
        p0, bounds, fit_kwargs = self._extract_fit_params(config, xdata, ydata, model)

        # Step 4: Execute curve fit
        result = self._execute_fit(
            model=model,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            **fit_kwargs,
        )

        # Step 5: Export results if configured
        if "export" in config:
            self.result_exporter.export(result, config)

        return result

    def _load_data(
        self, config: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Load data from configured source.

        Parameters
        ----------
        config : dict
            Workflow configuration with data section.

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Tuple of (xdata, ydata, sigma).

        Raises
        ------
        DataLoadError
            If data loading fails.
        CLIError
            If data configuration is missing.
        """
        data_config = config.get("data", {})
        input_file = data_config.get("input_file")

        if not input_file:
            raise CLIError(
                "No input file specified",
                suggestion="Set data.input_file in workflow configuration",
            )

        # Extract validation config and merge into data_config
        validation_config = config.get("validation", {})
        data_config_with_validation = {**data_config, "validation": validation_config}

        try:
            xdata, ydata, sigma = self.data_loader.load(
                input_file, data_config_with_validation
            )
            return xdata, ydata, sigma
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(
                f"Unexpected error loading data: {e}",
                file_path=input_file,
                suggestion="Check data file format and configuration",
            ) from e

    def _resolve_model(self, config: dict[str, Any]) -> Any:
        """Resolve model function from configuration.

        Parameters
        ----------
        config : dict
            Workflow configuration with model section.

        Returns
        -------
        callable
            Model function f(x, *params).

        Raises
        ------
        ModelError
            If model resolution fails.
        CLIError
            If model configuration is missing.
        """
        model_config = config.get("model", {})

        if not model_config:
            raise CLIError(
                "No model specified",
                suggestion="Add model section to workflow configuration",
            )

        model_type = model_config.get("type", "builtin")
        model_name = model_config.get("name", model_config.get("path", ""))

        try:
            return self.model_registry.get_model(model_name, model_config)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                f"Unexpected error resolving model: {e}",
                model_name=model_name,
                model_type=model_type,
                suggestion="Check model configuration",
            ) from e

    def _extract_fit_params(
        self,
        config: dict[str, Any],
        xdata: np.ndarray,
        ydata: np.ndarray,
        model: Any,
    ) -> tuple[np.ndarray | None, tuple | None, dict[str, Any]]:
        """Extract fitting parameters from configuration.

        Parameters
        ----------
        config : dict
            Workflow configuration with fitting section.
        xdata : ndarray
            X data for auto-estimation.
        ydata : ndarray
            Y data for auto-estimation.
        model : callable
            Model function (may have estimate_p0 method).

        Returns
        -------
        tuple
            (p0, bounds, fit_kwargs) where:
            - p0: Initial parameter array or None
            - bounds: Bounds tuple or None
            - fit_kwargs: Additional kwargs for curve_fit
        """
        fitting_config = config.get("fitting", {})

        # Extract p0
        p0 = fitting_config.get("p0")
        if p0 == "auto" or p0 is None:
            # Try to estimate from model's estimate_p0 method
            if hasattr(model, "estimate_p0"):
                try:
                    p0 = model.estimate_p0(xdata, ydata)
                    p0 = np.asarray(p0)
                except Exception:
                    p0 = None
        elif isinstance(p0, (list, tuple)):
            p0 = np.asarray(p0)

        # Extract bounds
        bounds_config = fitting_config.get("bounds")
        bounds: tuple | None = None

        if bounds_config is not None:
            if isinstance(bounds_config, dict):
                lower = bounds_config.get("lower", -np.inf)
                upper = bounds_config.get("upper", np.inf)
                bounds = (np.asarray(lower), np.asarray(upper))
            elif isinstance(bounds_config, (list, tuple)) and len(bounds_config) == 2:
                bounds = (np.asarray(bounds_config[0]), np.asarray(bounds_config[1]))
        elif hasattr(model, "bounds"):
            # Use model's default bounds if no bounds specified
            try:
                model_bounds = model.bounds()
                if model_bounds:
                    bounds = (np.asarray(model_bounds[0]), np.asarray(model_bounds[1]))
            except Exception:
                pass

        # Extract additional fit kwargs
        fit_kwargs: dict[str, Any] = {}

        # Method
        method = fitting_config.get("method")
        if method:
            fit_kwargs["method"] = method

        # Tolerances (convert strings to float to handle scientific notation)
        for tol_key in ["ftol", "xtol", "gtol"]:
            tol_val = fitting_config.get(tol_key)
            if tol_val is not None:
                # YAML may parse scientific notation as strings (e.g., '1e-10')
                fit_kwargs[tol_key] = float(tol_val)

        # Max iterations
        max_nfev = fitting_config.get("max_nfev")
        if max_nfev is not None:
            fit_kwargs["max_nfev"] = max_nfev

        # Loss function
        loss = fitting_config.get("loss")
        if loss is not None:
            fit_kwargs["loss"] = loss

        # Sigma handling
        absolute_sigma = fitting_config.get("absolute_sigma", False)
        fit_kwargs["absolute_sigma"] = absolute_sigma

        # Check finite
        check_finite = fitting_config.get("check_finite", True)
        fit_kwargs["check_finite"] = check_finite

        return p0, bounds, fit_kwargs

    def _execute_fit(
        self,
        model: Any,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None,
        sigma: np.ndarray | None,
        bounds: tuple | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute curve fit and return result dict.

        Parameters
        ----------
        model : callable
            Model function. For 1D: f(x, *params). For 2D: f(xy, *params).
        xdata : ndarray
            Independent variable data. For 1D: shape (n,). For 2D: shape (2, n).
        ydata : ndarray
            Dependent variable data. Shape (n,) for both 1D and 2D.
        p0 : ndarray or None
            Initial parameter guess.
        sigma : ndarray or None
            Uncertainties in ydata.
        bounds : tuple or None
            Parameter bounds as (lower, upper).
        **kwargs
            Additional keyword arguments for curve_fit.

        Returns
        -------
        dict
            Result dictionary with popt, pcov, success, message, etc.

        Raises
        ------
        FitError
            If curve fitting fails.
        """
        import nlsq

        # Determine number of data points (accounting for 2D xdata)
        if xdata.ndim == 2:
            n_points = xdata.shape[1]
        else:
            n_points = len(xdata)

        try:
            # Build curve_fit arguments
            fit_args: dict[str, Any] = {
                "f": model,
                "xdata": xdata,
                "ydata": ydata,
            }

            if p0 is not None:
                fit_args["p0"] = p0

            if sigma is not None:
                fit_args["sigma"] = sigma

            if bounds is not None:
                fit_args["bounds"] = bounds

            # Add remaining kwargs
            fit_args.update(kwargs)

            # Execute fit
            result = nlsq.curve_fit(**fit_args)

            # Handle tuple return (popt, pcov) vs CurveFitResult
            if isinstance(result, tuple):
                popt, pcov = result
                result_dict = {
                    "popt": np.asarray(popt).tolist(),
                    "pcov": np.asarray(pcov).tolist(),
                    "success": True,
                    "message": "Optimization converged",
                    "nfev": 0,  # Not available in tuple return
                }
            else:
                # CurveFitResult object
                result_dict = {
                    "popt": np.asarray(result.popt).tolist(),
                    "pcov": np.asarray(result.pcov).tolist()
                    if result.pcov is not None
                    else [],
                    "success": getattr(result, "success", True),
                    "message": getattr(result, "message", "Optimization converged"),
                    "nfev": getattr(result, "nfev", 0),
                    "njev": getattr(result, "njev", 0),
                    "cost": getattr(result, "cost", None),
                }

                # Add residuals if available
                if hasattr(result, "fun") and result.fun is not None:
                    result_dict["fun"] = np.asarray(result.fun).tolist()

                # Store ydata for statistics calculation
                result_dict["ydata"] = ydata.tolist()

            return result_dict

        except ValueError as e:
            error_msg = str(e)

            # Check for underdetermined system
            if (
                "underdetermined" in error_msg.lower()
                or "fewer data points" in error_msg.lower()
            ):
                raise FitError(
                    "Curve fitting failed: insufficient data points for number of parameters",
                    context={"n_points": n_points, "error": error_msg},
                    suggestion="Provide more data points or use a simpler model with fewer parameters",
                ) from e

            # Check for convergence failure
            if "covariance" in error_msg.lower() or "singular" in error_msg.lower():
                raise FitError(
                    "Curve fitting failed: could not estimate covariance",
                    context={"error": error_msg},
                    suggestion="Try different initial parameters or check that the model is appropriate for the data",
                ) from e

            raise FitError(
                f"Curve fitting failed: {e}",
                suggestion="Check initial parameters and bounds",
            ) from e

        except RuntimeError as e:
            error_msg = str(e)

            if "maxfev" in error_msg.lower() or "max" in error_msg.lower():
                raise FitError(
                    "Curve fitting failed: maximum function evaluations exceeded",
                    context={"error": error_msg},
                    suggestion="Increase max_nfev or improve initial parameter guess",
                ) from e

            raise FitError(
                f"Curve fitting failed: {e}",
                suggestion="Check model function and input data",
            ) from e

        except Exception as e:
            raise FitError(
                f"Unexpected error during curve fitting: {e}",
                suggestion="Check model function, data, and configuration",
            ) from e
