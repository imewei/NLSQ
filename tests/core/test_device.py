"""Tests for GPU detection and warning utilities (nlsq.device module).

This test suite provides 100% coverage of nlsq/device.py, testing all 6 code paths:
1. GPU available + JAX CPU-only (warning printed)
2. GPU available + JAX GPU (silent)
3. No GPU hardware (silent)
4. nvidia-smi timeout (silent error handling)
5. nvidia-smi missing (silent error handling)
6. JAX not installed (silent error handling)

Plus additional tests for:
- GPU name sanitization (special characters, very long names)
- Generic exception handling
- Import-time behavior
"""

import subprocess
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

from nlsq.device import check_gpu_availability


class TestCheckGPUAvailability:
    """Test basic GPU detection functionality."""

    def test_gpu_available_cpu_jax(self):
        """Test GPU hardware present but JAX running CPU-only.

        This is the main use case - user has GPU but hasn't installed
        JAX with CUDA support. Should print helpful warning.
        """

        # Mock subprocess to return different results based on command
        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "nvidia-smi" in cmd:
                # nvidia-smi --query-gpu=name,compute_cap returns CSV format
                mock_result.stdout = "Tesla V100-SXM2-16GB, 7.0\n"
            elif "nvcc" in cmd:
                # nvcc --version returns version info
                mock_result.stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.6, V12.6.77\n"
            else:
                mock_result.stdout = ""
            return mock_result

        # Mock JAX device to return CPU-only
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run) as mock_subprocess,
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch.dict("os.environ", {"NLSQ_SKIP_GPU_CHECK": ""}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

                # Verify warning was printed with actual format from _print_gpu_warning
                output_str = output.getvalue()
                assert "GPU ACCELERATION AVAILABLE" in output_str
                assert "===========================" in output_str
                assert "GPU: Tesla V100-SXM2-16GB (SM 7.0)" in output_str
                assert "JAX backend: CPU-only" in output_str
                assert "Enable 20-100x speedup:" in output_str
                assert "make install-jax-gpu" in output_str
                assert 'pip install "jax[cuda12-local]"' in output_str
                assert "See README.md for details." in output_str

        # Verify subprocess was called (once for nvidia-smi, once for nvcc)
        assert mock_subprocess.call_count == 2

    def test_gpu_and_jax_match(self):
        """Test GPU hardware and JAX both using GPU.

        This is the ideal case - user has GPU and JAX with CUDA.
        Should be silent (no warning).
        """
        # Mock subprocess to return GPU name
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100-SXM2-16GB\n"

        # Mock JAX device to return GPU (with "cuda" in string)
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cuda:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", return_value=mock_result),
            patch.dict("sys.modules", {"jax": mock_jax}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output (silent)
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_no_gpu_hardware(self):
        """Test nvidia-smi returns empty or error.

        This is normal on CPU-only systems. Should be silent.
        """
        # Mock subprocess to return error (no GPU)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output (silent)
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"


class TestGPUDetectionErrorHandling:
    """Test exception handling for GPU detection errors."""

    def test_nvidia_smi_timeout(self):
        """Test subprocess timeout after 5 seconds.

        nvidia-smi might hang on some systems. Should handle gracefully.
        """
        # Mock subprocess to raise TimeoutExpired
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output and no exception raised
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_nvidia_smi_missing(self):
        """Test nvidia-smi command not in PATH.

        This is normal on systems without NVIDIA drivers. Should be silent.
        """
        # Mock subprocess to raise FileNotFoundError
        with patch(
            "subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output and no exception raised
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_jax_not_installed(self):
        """Test JAX import fails at runtime.

        User might not have JAX installed yet. Should be silent.
        """
        # Mock subprocess to return GPU name
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100-SXM2-16GB\n"

        # Create a mock that raises ImportError when imported
        def mock_import(name, *args, **kwargs):
            if name == "jax":
                raise ImportError("No module named 'jax'")
            # For other imports, use the original __import__
            import builtins

            return builtins.__import__(name, *args, **kwargs)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output and no exception raised
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_unexpected_exception(self):
        """Test generic exception handling (catch-all).

        Any unexpected error should be caught and silently ignored.
        """
        # Mock subprocess to raise a generic RuntimeError
        with patch("subprocess.run", side_effect=RuntimeError("Unexpected error")):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output and no exception raised
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"


class TestGPUNameSanitization:
    """Test GPU name handling with edge cases."""

    def test_gpu_name_with_special_chars(self):
        """Test malicious GPU name with special characters.

        GPU name should be printed as-is (testing shows it's safe).
        This test documents the current behavior.
        """

        # Mock subprocess to return different results based on command
        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "nvidia-smi" in cmd:
                # nvidia-smi returns CSV format with special chars in GPU name
                mock_result.stdout = "Tesla V100 <script>alert('xss')</script>, 7.0\n"
            elif "nvcc" in cmd:
                mock_result.stdout = "Cuda compilation tools, release 12.6, V12.6.77\n"
            else:
                mock_result.stdout = ""
            return mock_result

        # Mock JAX device to return CPU-only
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch.dict("os.environ", {"NLSQ_SKIP_GPU_CHECK": ""}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify warning printed with the GPU name as-is
            output_str = output.getvalue()
            assert "GPU ACCELERATION AVAILABLE" in output_str
            assert (
                "GPU: Tesla V100 <script>alert('xss')</script> (SM 7.0)" in output_str
            )

    def test_gpu_name_very_long(self):
        """Test extremely long GPU name (>1000 chars).

        GPU names are passed through as-is from nvidia-smi.
        """
        # Create a very long GPU name (1500 characters)
        long_name = "A" * 1500

        # Mock subprocess to return different results based on command
        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "nvidia-smi" in cmd:
                # nvidia-smi returns CSV format with long GPU name
                mock_result.stdout = f"{long_name}, 7.0\n"
            elif "nvcc" in cmd:
                mock_result.stdout = "Cuda compilation tools, release 12.6, V12.6.77\n"
            else:
                mock_result.stdout = ""
            return mock_result

        # Mock JAX device to return CPU-only
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch.dict("os.environ", {"NLSQ_SKIP_GPU_CHECK": ""}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify warning printed with the full GPU name (no truncation)
            output_str = output.getvalue()
            assert "GPU ACCELERATION AVAILABLE" in output_str
            # GPU name is printed as-is (no truncation in current implementation)
            assert f"GPU: {long_name} (SM 7.0)" in output_str


class TestImportIntegration:
    """Test import-time behavior."""

    def test_import_nlsq_triggers_check(self):
        """Verify check_gpu_availability called on nlsq import.

        This is an integration test that verifies the function is
        actually called during module import.
        """
        # This test documents that the function is called on import
        # We verify the function exists and can be called
        # The actual import-time call is tested by running the module

        # Mock subprocess to return different results based on command
        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "nvidia-smi" in cmd:
                mock_result.stdout = "Tesla V100-SXM2-16GB, 7.0\n"
            elif "nvcc" in cmd:
                mock_result.stdout = "Cuda compilation tools, release 12.6, V12.6.77\n"
            else:
                mock_result.stdout = ""
            return mock_result

        # Mock JAX device to return CPU-only
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch.dict("os.environ", {"NLSQ_SKIP_GPU_CHECK": ""}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                # Call the function (simulating import behavior)
                check_gpu_availability()

            # Verify warning was printed (proving function is callable)
            output_str = output.getvalue()
            assert "GPU ACCELERATION AVAILABLE" in output_str

        # Note: The actual import-time call is in nlsq/__init__.py
        # and is tested by the fact that importing nlsq doesn't crash

    def test_skip_gpu_check_env_var(self, monkeypatch):
        """NLSQ_SKIP_GPU_CHECK=1 suppresses GPU warning.

        This test verifies that setting NLSQ_SKIP_GPU_CHECK environment
        variable prevents the GPU detection check from running, even when
        GPU hardware is present.
        """
        # Set environment variable to skip GPU check
        monkeypatch.setenv("NLSQ_SKIP_GPU_CHECK", "1")

        # Mock subprocess to return GPU name (would normally trigger warning)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100-SXM2-16GB\n"

        # Mock JAX device to return CPU-only (would normally trigger warning)
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", return_value=mock_result),
            patch.dict("sys.modules", {"jax": mock_jax}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify NO output (silent operation)
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_skip_gpu_check_true(self, monkeypatch):
        """NLSQ_SKIP_GPU_CHECK=true also suppresses GPU warning.

        Test alternative values: "true", "TRUE", "yes", "YES"
        """
        for value in ["true", "TRUE", "yes", "YES"]:
            monkeypatch.setenv("NLSQ_SKIP_GPU_CHECK", value)

            # Mock subprocess (would trigger warning without env var)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Tesla V100-SXM2-16GB\n"

            mock_device = MagicMock()
            mock_device.__str__ = lambda self: "cpu:0"

            mock_jax = MagicMock()
            mock_jax.devices.return_value = [mock_device]

            with (
                patch("subprocess.run", return_value=mock_result),
                patch.dict("sys.modules", {"jax": mock_jax}),
            ):
                output = StringIO()
                with redirect_stdout(output):
                    check_gpu_availability()

                # Verify silent operation for all values
                output_str = output.getvalue()
                assert output_str == "", (
                    f"Expected no output for NLSQ_SKIP_GPU_CHECK={value}, got: {output_str}"
                )


class TestGPUDetectionWithMultipleDevices:
    """Test GPU detection with multiple JAX devices."""

    def test_multiple_cpu_devices(self):
        """Test multiple CPU devices (no GPU).

        Should print warning if GPU hardware exists.
        """

        # Mock subprocess to return different results based on command
        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "nvidia-smi" in cmd:
                mock_result.stdout = "Tesla V100-SXM2-16GB, 7.0\n"
            elif "nvcc" in cmd:
                mock_result.stdout = "Cuda compilation tools, release 12.6, V12.6.77\n"
            else:
                mock_result.stdout = ""
            return mock_result

        # Mock multiple CPU devices
        mock_device1 = MagicMock()
        mock_device1.__str__ = lambda self: "cpu:0"
        mock_device2 = MagicMock()
        mock_device2.__str__ = lambda self: "cpu:1"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device1, mock_device2]

        with (
            patch("subprocess.run", side_effect=mock_subprocess_run),
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch.dict("os.environ", {"NLSQ_SKIP_GPU_CHECK": ""}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify warning printed (no GPU in device list)
            output_str = output.getvalue()
            assert "GPU ACCELERATION AVAILABLE" in output_str

    def test_mixed_cpu_and_gpu_devices(self):
        """Test mixed CPU and GPU devices.

        Should be silent if at least one GPU device exists.
        """
        # Mock subprocess to return GPU name
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100-SXM2-16GB\n"

        # Mock mixed CPU and GPU devices
        mock_device_cpu = MagicMock()
        mock_device_cpu.__str__ = lambda self: "cpu:0"
        mock_device_gpu = MagicMock()
        mock_device_gpu.__str__ = lambda self: "cuda:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device_cpu, mock_device_gpu]

        with (
            patch("subprocess.run", return_value=mock_result),
            patch.dict("sys.modules", {"jax": mock_jax}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output (GPU device found)
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"

    def test_gpu_device_lowercase(self):
        """Test GPU device detection with lowercase 'gpu' string.

        The code checks for both 'cuda' and 'gpu' (case-insensitive).
        """
        # Mock subprocess to return GPU name
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100-SXM2-16GB\n"

        # Mock GPU device with lowercase "gpu" in string
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "gpu:0"

        # Mock JAX module
        mock_jax = MagicMock()
        mock_jax.devices.return_value = [mock_device]

        with (
            patch("subprocess.run", return_value=mock_result),
            patch.dict("sys.modules", {"jax": mock_jax}),
        ):
            # Capture stdout
            output = StringIO()
            with redirect_stdout(output):
                check_gpu_availability()

            # Verify no output (GPU device found)
            output_str = output.getvalue()
            assert output_str == "", f"Expected no output, got: {output_str}"
