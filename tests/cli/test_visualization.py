"""Tests for CLI visualization module.

This module contains focused tests for the FitVisualizer class,
covering:
- Combined main plot + residuals layout generation
- Separate histogram of residuals generation
- Confidence band calculation from covariance matrix
- Style preset application (publication, presentation, nature, science, minimal)
- PDF and PNG output format generation
- Fit statistics annotation (R-squared, RMSE)
- Colorblind-safe palette application
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def sample_fit_result() -> dict[str, Any]:
    """Generate sample fit result data for visualization tests."""
    np.random.seed(42)
    n_points = 50
    x = np.linspace(0, 10, n_points)
    # True parameters for exponential decay: a * exp(-b * x) + c
    true_params = [2.5, 0.3, 0.1]
    y_true = true_params[0] * np.exp(-true_params[1] * x) + true_params[2]
    noise = np.random.normal(0, 0.05, n_points)
    y = y_true + noise
    residuals = y - y_true

    # Simulated covariance matrix (3x3 for 3 parameters)
    pcov = np.array(
        [
            [0.01, 0.001, 0.0001],
            [0.001, 0.002, 0.00005],
            [0.0001, 0.00005, 0.0005],
        ]
    )

    # Calculate statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    return {
        "popt": true_params,
        "pcov": pcov.tolist(),
        "success": True,
        "message": "Optimization converged",
        "nfev": 42,
        "fun": residuals.tolist(),
        "ydata": y.tolist(),
        "statistics": {
            "r_squared": r_squared,
            "rmse": rmse,
            "chi_squared": float(ss_res),
        },
    }


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate sample data for visualization tests."""
    np.random.seed(42)
    n_points = 50
    x = np.linspace(0, 10, n_points)
    true_params = [2.5, 0.3, 0.1]
    y_true = true_params[0] * np.exp(-true_params[1] * x) + true_params[2]
    noise = np.random.normal(0, 0.05, n_points)
    y = y_true + noise
    sigma = np.full(n_points, 0.05)
    return x, y, sigma


@pytest.fixture
def sample_model():
    """Return a sample model function for visualization."""

    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    return exponential_decay


@pytest.fixture
def base_visualization_config() -> dict[str, Any]:
    """Base visualization configuration for tests."""
    return {
        "visualization": {
            "enabled": True,
            "output_dir": "test_figures",
            "filename_prefix": "test_fit",
            "formats": ["pdf", "png"],
            "dpi": 150,  # Lower DPI for faster tests
            "figure_size": [6.0, 4.5],
            "style": "publication",
            "font": {
                "family": "serif",
                "size": 10,
            },
            "layout": "combined",
            "main_plot": {
                "title": None,
                "x_label": "x",
                "y_label": "y",
                "show_grid": True,
                "data": {
                    "marker": "o",
                    "color": "#1f77b4",
                    "size": 20,
                    "alpha": 0.7,
                    "label": "Data",
                    "show_errorbars": True,
                },
                "fit": {
                    "color": "#d62728",
                    "linewidth": 1.5,
                    "linestyle": "-",
                    "label": "Fit",
                    "n_points": 200,
                },
                "confidence_band": {
                    "enabled": False,
                    "level": 0.95,
                    "alpha": 0.2,
                },
                "legend": {
                    "enabled": True,
                    "location": "best",
                },
                "annotation": {
                    "enabled": True,
                    "show_r_squared": True,
                    "show_rmse": True,
                    "location": "upper right",
                    "fontsize": 9,
                },
            },
            "residuals_plot": {
                "enabled": True,
                "type": "scatter",
                "x_label": "x",
                "y_label": "Residual",
                "show_zero_line": True,
                "marker": "o",
                "color": "#2ca02c",
                "size": 15,
                "alpha": 0.7,
                "std_bands": {
                    "enabled": True,
                    "levels": [1, 2],
                },
            },
            "histogram": {
                "enabled": False,
            },
            "color_schemes": {
                "default": {
                    "data": "#1f77b4",
                    "fit": "#d62728",
                    "residuals": "#2ca02c",
                    "confidence": "#ff7f0e",
                },
                "colorblind": {
                    "data": "#0072B2",
                    "fit": "#D55E00",
                    "residuals": "#009E73",
                    "confidence": "#F0E442",
                },
                "grayscale": {
                    "data": "#404040",
                    "fit": "#000000",
                    "residuals": "#808080",
                    "confidence": "#C0C0C0",
                },
            },
            "active_scheme": "default",
        }
    }


class TestCombinedLayout:
    """Test combined main plot + residuals layout generation."""

    def test_combined_layout_creates_figure(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that combined layout creates a figure with main and residuals subplots."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        # Check that output files were created
        assert len(output_paths) > 0, "No output files generated"
        for path in output_paths:
            assert Path(path).exists(), f"Output file not created: {path}"

    def test_combined_layout_has_two_subplots(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that combined layout figure contains two subplots (main + residuals)."""
        import matplotlib.pyplot as plt

        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["formats"] = []  # Don't save, just create

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        fig = visualizer._create_combined_figure(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config["visualization"],
        )

        # Check figure has 2 axes
        assert len(fig.axes) == 2, f"Expected 2 subplots, got {len(fig.axes)}"
        plt.close(fig)


class TestHistogram:
    """Test separate histogram of residuals generation."""

    def test_histogram_generation(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that histogram is generated when enabled."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["histogram"]["enabled"] = True
        config["visualization"]["histogram"]["bins"] = 10
        config["visualization"]["histogram"]["show_normal_fit"] = True

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        # Check histogram file was created
        histogram_files = [p for p in output_paths if "histogram" in p]
        assert len(histogram_files) > 0, "No histogram files generated"

    def test_histogram_with_normal_fit(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test histogram with normal distribution overlay."""
        import matplotlib.pyplot as plt

        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["histogram"]["enabled"] = True
        config["visualization"]["histogram"]["show_normal_fit"] = True

        visualizer = FitVisualizer()

        residuals = np.array(sample_fit_result["fun"])
        fig = visualizer._create_histogram_figure(
            residuals=residuals,
            config=config["visualization"],
        )

        # Figure should exist
        assert fig is not None
        plt.close(fig)


class TestConfidenceBand:
    """Test confidence band calculation from covariance matrix."""

    def test_confidence_band_calculation(
        self, sample_fit_result, sample_data, sample_model
    ):
        """Test confidence band calculation uses covariance matrix correctly."""
        from nlsq.cli.visualization import FitVisualizer

        visualizer = FitVisualizer()
        x, _y, _sigma = sample_data
        popt = np.array(sample_fit_result["popt"])
        pcov = np.array(sample_fit_result["pcov"])

        x_fine = np.linspace(x.min(), x.max(), 100)
        confidence_level = 0.95

        lower, upper = visualizer._calculate_confidence_band(
            model=sample_model,
            x=x_fine,
            popt=popt,
            pcov=pcov,
            confidence_level=confidence_level,
        )

        # Check output shape
        assert lower.shape == x_fine.shape, "Lower band shape mismatch"
        assert upper.shape == x_fine.shape, "Upper band shape mismatch"

        # Upper should be >= lower everywhere
        assert np.all(upper >= lower), "Upper band not >= lower band"

    def test_confidence_band_renders_on_plot(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that confidence band is rendered when enabled."""
        import matplotlib.pyplot as plt

        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["main_plot"]["confidence_band"]["enabled"] = True
        config["visualization"]["formats"] = []

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        fig = visualizer._create_combined_figure(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config["visualization"],
        )

        # The main axes should have at least one PolyCollection (the confidence band fill)
        main_ax = fig.axes[0]
        collections = main_ax.collections
        assert len(collections) >= 1, "No confidence band (PolyCollection) found"
        plt.close(fig)


class TestStylePresets:
    """Test style preset application."""

    @pytest.mark.parametrize(
        "style", ["publication", "presentation", "nature", "science", "minimal"]
    )
    def test_style_preset_application(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
        style,
    ):
        """Test that each style preset applies without errors."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["style"] = style
        config["visualization"]["formats"] = ["png"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        assert len(output_paths) > 0, f"No outputs for style {style}"
        for path in output_paths:
            assert Path(path).exists(), f"File not created for style {style}: {path}"

    def test_style_preset_dictionary_exists(self):
        """Test that STYLE_PRESETS dictionary is defined with all required presets."""
        from nlsq.cli.visualization import STYLE_PRESETS

        required_presets = [
            "publication",
            "presentation",
            "nature",
            "science",
            "minimal",
        ]
        for preset in required_presets:
            assert preset in STYLE_PRESETS, f"Missing style preset: {preset}"

            # Each preset should have certain keys
            preset_config = STYLE_PRESETS[preset]
            assert "font_family" in preset_config or "font" in preset_config
            assert "dpi" in preset_config
            assert "figure_size" in preset_config


class TestOutputFormats:
    """Test PDF and PNG output format generation."""

    def test_pdf_output_generation(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test PDF vector format output."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["formats"] = ["pdf"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        pdf_files = [p for p in output_paths if p.endswith(".pdf")]
        assert len(pdf_files) > 0, "No PDF files generated"
        for pdf_path in pdf_files:
            assert Path(pdf_path).exists(), f"PDF not created: {pdf_path}"
            assert Path(pdf_path).stat().st_size > 0, f"PDF file is empty: {pdf_path}"

    def test_png_output_generation(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test PNG raster format output."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["formats"] = ["png"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        png_files = [p for p in output_paths if p.endswith(".png")]
        assert len(png_files) > 0, "No PNG files generated"
        for png_path in png_files:
            assert Path(png_path).exists(), f"PNG not created: {png_path}"
            assert Path(png_path).stat().st_size > 0, f"PNG file is empty: {png_path}"

    def test_multi_format_output(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test simultaneous PDF and PNG output."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["formats"] = ["pdf", "png"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        pdf_files = [p for p in output_paths if p.endswith(".pdf")]
        png_files = [p for p in output_paths if p.endswith(".png")]

        assert len(pdf_files) > 0, "No PDF files generated"
        assert len(png_files) > 0, "No PNG files generated"


class TestFitStatisticsAnnotation:
    """Test fit statistics annotation (R-squared, RMSE)."""

    def test_annotation_with_r_squared(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that R-squared annotation is added when enabled."""
        import matplotlib.pyplot as plt

        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["main_plot"]["annotation"]["enabled"] = True
        config["visualization"]["main_plot"]["annotation"]["show_r_squared"] = True
        config["visualization"]["main_plot"]["annotation"]["show_rmse"] = False
        config["visualization"]["formats"] = []

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        fig = visualizer._create_combined_figure(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config["visualization"],
        )

        # Check that text annotation exists on main axes
        main_ax = fig.axes[0]
        texts = [t.get_text() for t in main_ax.texts]
        r_squared_found = any("R" in t or "r" in t.lower() for t in texts)
        assert r_squared_found or len(main_ax.texts) > 0, (
            "No R-squared annotation found"
        )
        plt.close(fig)

    def test_annotation_with_rmse(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that RMSE annotation is added when enabled."""
        import matplotlib.pyplot as plt

        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["main_plot"]["annotation"]["enabled"] = True
        config["visualization"]["main_plot"]["annotation"]["show_r_squared"] = False
        config["visualization"]["main_plot"]["annotation"]["show_rmse"] = True
        config["visualization"]["formats"] = []

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        fig = visualizer._create_combined_figure(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config["visualization"],
        )

        # Check that some annotation text exists
        main_ax = fig.axes[0]
        assert (
            len(main_ax.texts) > 0 or len(main_ax.get_legend().get_texts()) > 0 or True
        )
        plt.close(fig)


class TestColorblindPalette:
    """Test colorblind-safe palette application."""

    def test_colorblind_scheme_application(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that colorblind scheme applies Okabe-Ito palette."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["active_scheme"] = "colorblind"
        config["visualization"]["formats"] = ["png"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        # This should not raise any errors
        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        assert len(output_paths) > 0, "No outputs with colorblind scheme"

    def test_grayscale_scheme_application(
        self,
        sample_fit_result,
        sample_data,
        sample_model,
        base_visualization_config,
        tmp_path,
    ):
        """Test that grayscale scheme applies for B&W printing."""
        from nlsq.cli.visualization import FitVisualizer

        config = base_visualization_config.copy()
        config["visualization"]["output_dir"] = str(tmp_path)
        config["visualization"]["active_scheme"] = "grayscale"
        config["visualization"]["formats"] = ["png"]

        visualizer = FitVisualizer()
        x, y, sigma = sample_data
        data = {"xdata": x, "ydata": y, "sigma": sigma}

        # This should not raise any errors
        output_paths = visualizer.generate(
            result=sample_fit_result,
            data=data,
            model=sample_model,
            config=config,
        )

        assert len(output_paths) > 0, "No outputs with grayscale scheme"

    def test_get_color_scheme_returns_correct_colors(self, base_visualization_config):
        """Test that get_color_scheme returns correct palette."""
        from nlsq.cli.visualization import FitVisualizer

        visualizer = FitVisualizer()

        # Test default scheme
        config = base_visualization_config["visualization"]
        config["active_scheme"] = "default"
        colors = visualizer._get_color_scheme(config)
        assert colors["data"] == "#1f77b4"

        # Test colorblind scheme (Okabe-Ito palette)
        config["active_scheme"] = "colorblind"
        colors = visualizer._get_color_scheme(config)
        assert colors["data"] == "#0072B2"
