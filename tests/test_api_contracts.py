"""API contract tests to prevent signature drift.

These tests validate that method signatures match specifications
and prevent accidental API breakage.
"""

import inspect

import numpy as np

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestStreamingOptimizerAPI:
    """Test StreamingOptimizer API contracts."""

    def test_fit_method_signature(self):
        """Validate fit() method signature."""
        sig = inspect.signature(StreamingOptimizer.fit)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "data_source",
            "func",
            "p0",
            "bounds",
            "callback",
            "verbose",
        ]

        assert params == expected, (
            f"StreamingOptimizer.fit() signature changed!\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {params}\n"
            f"  This breaks backward compatibility."
        )

    def test_fit_return_type(self):
        """Validate fit() returns correct dictionary structure."""
        config = StreamingConfig(batch_size=10, max_epochs=1)
        optimizer = StreamingOptimizer(config)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = 2 * x[:, 0] + 1
        model = lambda x, a, b: a * x + b

        result = optimizer.fit((x, y), model, p0=[1.0, 0.0], verbose=0)

        # Required keys
        required_keys = [
            "x",
            "success",
            "message",
            "fun",
            "best_loss",
            "final_epoch",
            "streaming_diagnostics",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: '{key}'"

        # Validate types
        assert isinstance(result["x"], np.ndarray)
        assert isinstance(result["success"], bool)
        assert isinstance(result["message"], str)
        assert isinstance(result["fun"], float)
        assert isinstance(result["best_loss"], float)
        assert isinstance(result["final_epoch"], int)
        assert isinstance(result["streaming_diagnostics"], dict)

    def test_streaming_diagnostics_structure(self):
        """Validate streaming_diagnostics has expected structure."""
        config = StreamingConfig(batch_size=10, max_epochs=1)
        optimizer = StreamingOptimizer(config)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = 2 * x[:, 0] + 1
        model = lambda x, a, b: a * x + b

        result = optimizer.fit((x, y), model, p0=[1.0, 0.0], verbose=0)

        # Validate diagnostics structure
        diag = result["streaming_diagnostics"]
        diag_required = [
            "failed_batches",
            "retry_counts",
            "error_types",
            "batch_success_rate",
            "total_batches_attempted",
            "total_retries",
            "convergence_achieved",
            "final_epoch",
            "elapsed_time",
            "checkpoint_info",
            "recent_batch_stats",
            "aggregate_stats",
        ]

        for key in diag_required:
            assert key in diag, f"Missing diagnostics key: '{key}'"

    def test_aggregate_stats_structure(self):
        """Validate aggregate_stats contains expected fields."""
        config = StreamingConfig(batch_size=10, max_epochs=1)
        optimizer = StreamingOptimizer(config)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = 2 * x[:, 0] + 1
        model = lambda x, a, b: a * x + b

        result = optimizer.fit((x, y), model, p0=[1.0, 0.0], verbose=0)

        aggregate = result["streaming_diagnostics"]["aggregate_stats"]

        required_fields = [
            "mean_loss",
            "std_loss",
            "min_loss",
            "max_loss",
            "mean_grad_norm",
            "std_grad_norm",
            "mean_batch_time",
            "std_batch_time",
        ]

        for field in required_fields:
            assert field in aggregate, f"Missing aggregate_stats field: '{field}'"
            assert isinstance(aggregate[field], float), (
                f"aggregate_stats['{field}'] should be float, got {type(aggregate[field])}"
            )

    def test_batch_stats_structure(self):
        """Validate batch statistics dictionary structure."""
        config = StreamingConfig(batch_size=10, max_epochs=1)
        optimizer = StreamingOptimizer(config)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = 2 * x[:, 0] + 1
        model = lambda x, a, b: a * x + b

        result = optimizer.fit((x, y), model, p0=[1.0, 0.0], verbose=0)

        # Get batch stats from recent_batch_stats
        batch_stats = result["streaming_diagnostics"]["recent_batch_stats"]

        if len(batch_stats) > 0:
            batch_stat = batch_stats[0]

            required_fields = [
                "batch_idx",
                "loss",
                "grad_norm",
                "batch_time",
                "success",
                "retry_count",
            ]

            for field in required_fields:
                assert field in batch_stat, f"Missing batch_stat field: '{field}'"

    def test_streaming_config_fields(self):
        """Validate StreamingConfig has expected fields."""
        config = StreamingConfig()

        # Critical fields
        assert hasattr(config, "batch_size")
        assert hasattr(config, "max_epochs")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "enable_fault_tolerance")
        assert hasattr(config, "validate_numerics")
        assert hasattr(config, "min_success_rate")
        assert hasattr(config, "max_retries_per_batch")

        # Validate defaults match documentation
        assert config.batch_size == 32
        assert config.max_epochs == 10
        assert config.enable_fault_tolerance is True
        assert config.min_success_rate == 0.5
        assert config.max_retries_per_batch == 2

    def test_data_source_types_accepted(self):
        """Validate that fit() accepts expected data_source types."""
        config = StreamingConfig(batch_size=10, max_epochs=1)
        optimizer = StreamingOptimizer(config)

        x = np.linspace(0, 1, 50).reshape(-1, 1)
        y = 2 * x[:, 0] + 1
        model = lambda x, a, b: a * x + b
        p0 = [1.0, 0.0]

        # Test 1: Tuple of arrays (should work)
        result = optimizer.fit((x, y), model, p0=p0, verbose=0)
        assert result["success"]

        # Test 2: Generator (should work)
        def data_generator():
            for i in range(0, len(x), 10):
                yield x[i : i + 10], y[i : i + 10]

        # Reset optimizer
        optimizer = StreamingOptimizer(config)
        result = optimizer.fit(data_generator(), model, p0=p0, verbose=0)
        assert result["success"]
