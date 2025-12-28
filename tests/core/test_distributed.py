"""Tests for distributed processing and cluster detection (Task Group 6).

This module tests:
- ClusterDetector PBS environment detection
- ClusterDetector non-cluster environment handling
- Multi-GPU sharding configuration
- pmap/pjit integration for data parallelism
- "hpc_distributed" preset configuration
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from nlsq.core.workflow import (
    WORKFLOW_PRESETS,
    ClusterDetector,
    ClusterInfo,
    MultiGPUConfig,
    WorkflowConfig,
    create_distributed_config,
    get_multi_gpu_config,
)


class TestClusterDetector:
    """Tests for ClusterDetector class."""

    def test_detect_pbs_environment_with_mock_nodefile(self, tmp_path: Path) -> None:
        """Test ClusterDetector detects PBS environment via PBS_NODEFILE."""
        # Create mock PBS_NODEFILE with 3 nodes (duplicated entries like real PBS)
        nodefile = tmp_path / "pbs_nodefile"
        nodefile.write_text("node01\nnode01\nnode02\nnode02\nnode03\nnode03\n")

        # Mock environment and JAX devices
        with patch.dict(
            os.environ,
            {
                "PBS_NODEFILE": str(nodefile),
                "PBS_JOBID": "12345.pbs_server",
            },
        ):
            detector = ClusterDetector(default_gpus_per_node=8)

            # Check PBS environment detection
            assert detector.is_pbs_environment() is True

            # Mock JAX to return 0 GPUs (will use default)
            with patch.object(detector, "_detect_gpus_per_node", return_value=0):
                cluster_info = detector.detect_pbs()

            assert cluster_info is not None
            assert cluster_info.node_count == 3
            assert cluster_info.gpus_per_node == 8  # default
            assert cluster_info.total_gpus == 24
            assert cluster_info.node_list == ["node01", "node02", "node03"]
            assert cluster_info.scheduler == "pbs"
            assert cluster_info.job_id == "12345.pbs_server"

    def test_detect_non_cluster_environment(self) -> None:
        """Test ClusterDetector returns None for non-cluster environment."""
        # Ensure PBS_NODEFILE is not set
        env = os.environ.copy()
        env.pop("PBS_NODEFILE", None)

        with patch.dict(os.environ, env, clear=True):
            detector = ClusterDetector()

            # Not in PBS environment
            assert detector.is_pbs_environment() is False

            # Mock JAX to return 0 GPUs (no GPUs available)
            with patch.object(detector, "_detect_gpus_per_node", return_value=0):
                cluster_info = detector.detect()

            # Should return None when not in cluster and no GPUs
            assert cluster_info is None

    def test_multi_gpu_sharding_config_generation(self) -> None:
        """Test multi-GPU sharding configuration generation."""
        # Create cluster info for single-node multi-GPU setup
        cluster_info = ClusterInfo(
            node_count=1,
            gpus_per_node=4,
            total_gpus=4,
            node_list=["localhost"],
            scheduler="local",
        )

        gpu_config = get_multi_gpu_config(cluster_info)

        assert gpu_config is not None
        assert gpu_config.n_devices == 4
        assert gpu_config.shard_axis == 0  # Batch dimension
        assert gpu_config.use_pmap is True  # Single-node uses pmap
        assert gpu_config.use_pjit is False  # Multi-node uses pjit
        assert gpu_config.per_device_batch_size == 10000
        assert gpu_config.total_batch_size == 40000

    def test_pmap_pjit_integration_for_data_parallelism(self) -> None:
        """Test pmap/pjit integration for data parallelism."""
        # Test single-node: should use pmap
        single_node_info = ClusterInfo(
            node_count=1,
            gpus_per_node=8,
            total_gpus=8,
            node_list=["node01"],
            scheduler="local",
        )
        single_config = get_multi_gpu_config(single_node_info)
        assert single_config is not None
        assert single_config.use_pmap is True
        assert single_config.use_pjit is False

        # Test multi-node: should use pjit
        multi_node_info = ClusterInfo(
            node_count=6,
            gpus_per_node=8,
            total_gpus=48,
            node_list=["node01", "node02", "node03", "node04", "node05", "node06"],
            scheduler="pbs",
        )
        multi_config = get_multi_gpu_config(multi_node_info)
        assert multi_config is not None
        assert multi_config.use_pmap is False  # Multi-node should NOT use pmap
        assert multi_config.use_pjit is True  # Multi-node SHOULD use pjit
        # Multi-node should have larger per-device batch size
        assert multi_config.per_device_batch_size == 50000

    def test_hpc_distributed_preset_configuration(self) -> None:
        """Test 'hpc_distributed' preset configuration."""
        # Check preset exists
        assert "hpc_distributed" in WORKFLOW_PRESETS

        preset = WORKFLOW_PRESETS["hpc_distributed"]

        # Verify preset properties
        assert preset["tier"] == "STREAMING_CHECKPOINT"
        assert preset["goal"] == "ROBUST"
        assert preset["enable_multistart"] is True
        assert preset["n_starts"] == 10
        assert preset["enable_checkpoints"] is True
        assert preset["distributed"] is True
        assert preset["chunk_size"] == 500_000  # Larger for distributed
        assert preset["auto_detect_cluster"] is True

        # Create WorkflowConfig from preset
        config = WorkflowConfig.from_preset("hpc_distributed")
        assert config.enable_checkpoints is True
        assert config.enable_multistart is True
        assert config.chunk_size == 500_000


class TestClusterInfoDataclass:
    """Tests for ClusterInfo dataclass."""

    def test_cluster_info_creation(self) -> None:
        """Test ClusterInfo can be created with all fields."""
        info = ClusterInfo(
            node_count=6,
            gpus_per_node=8,
            total_gpus=48,
            node_list=["node01", "node02", "node03", "node04", "node05", "node06"],
            scheduler="pbs",
            job_id="12345.pbs_server",
            interconnect="infiniband",
        )

        assert info.node_count == 6
        assert info.gpus_per_node == 8
        assert info.total_gpus == 48
        assert len(info.node_list) == 6
        assert info.scheduler == "pbs"
        assert info.job_id == "12345.pbs_server"
        assert info.interconnect == "infiniband"

    def test_cluster_info_serialization(self) -> None:
        """Test ClusterInfo to_dict and from_dict."""
        info = ClusterInfo(
            node_count=3,
            gpus_per_node=4,
            total_gpus=12,
            node_list=["n1", "n2", "n3"],
            scheduler="pbs",
            job_id="123",
        )

        d = info.to_dict()
        restored = ClusterInfo.from_dict(d)

        assert restored.node_count == info.node_count
        assert restored.gpus_per_node == info.gpus_per_node
        assert restored.total_gpus == info.total_gpus
        assert restored.node_list == info.node_list
        assert restored.scheduler == info.scheduler
        assert restored.job_id == info.job_id


class TestMultiGPUConfig:
    """Tests for MultiGPUConfig dataclass."""

    def test_multi_gpu_config_creation(self) -> None:
        """Test MultiGPUConfig creation."""
        config = MultiGPUConfig(
            n_devices=4,
            per_device_batch_size=5000,
        )

        assert config.n_devices == 4
        assert config.shard_axis == 0
        assert config.use_pmap is True
        assert config.use_pjit is False
        assert config.per_device_batch_size == 5000
        assert config.total_batch_size == 20000

    def test_multi_gpu_config_serialization(self) -> None:
        """Test MultiGPUConfig to_dict."""
        config = MultiGPUConfig(n_devices=8, per_device_batch_size=10000)
        d = config.to_dict()

        assert d["n_devices"] == 8
        assert d["per_device_batch_size"] == 10000
        assert "total_batch_size" not in d  # Property, not serialized


class TestCreateDistributedConfig:
    """Tests for create_distributed_config function."""

    def test_distributed_config_for_hpc_cluster(self) -> None:
        """Test distributed config generation for HPC cluster."""
        cluster_info = ClusterInfo(
            node_count=6,
            gpus_per_node=8,
            total_gpus=48,
            node_list=["node01", "node02", "node03", "node04", "node05", "node06"],
            scheduler="pbs",
            job_id="12345",
        )

        config = create_distributed_config(cluster_info)

        assert config["tier"] == "STREAMING_CHECKPOINT"
        assert config["distributed"] is True
        assert config["n_devices"] == 48
        assert config["nodes"] == 6
        assert config["gpus_per_node"] == 8
        assert config["enable_checkpoints"] is True  # Multi-node enables checkpoints
        assert config["enable_multistart"] is True
        assert config["n_starts"] == 20  # Scaled with GPUs (min of 48, 20)
        assert 100_000 <= config["chunk_size"] <= 1_000_000

    def test_distributed_config_for_small_cluster(self) -> None:
        """Test distributed config for small multi-GPU setup."""
        cluster_info = ClusterInfo(
            node_count=1,
            gpus_per_node=2,
            total_gpus=2,
            node_list=["localhost"],
            scheduler="local",
        )

        config = create_distributed_config(cluster_info)

        assert config["n_devices"] == 2
        assert config["nodes"] == 1
        # Small cluster with 2 GPUs should not enable checkpoints
        assert config["enable_checkpoints"] is False


class TestClusterDetectorEdgeCases:
    """Tests for ClusterDetector edge cases."""

    def test_empty_nodefile(self, tmp_path: Path) -> None:
        """Test handling of empty PBS_NODEFILE."""
        nodefile = tmp_path / "empty_nodefile"
        nodefile.write_text("")

        with patch.dict(os.environ, {"PBS_NODEFILE": str(nodefile)}):
            detector = ClusterDetector()
            result = detector.detect_pbs()
            assert result is None

    def test_nonexistent_nodefile(self) -> None:
        """Test handling of non-existent PBS_NODEFILE."""
        with patch.dict(os.environ, {"PBS_NODEFILE": "/nonexistent/path/nodefile"}):
            detector = ClusterDetector()
            result = detector.detect_pbs()
            assert result is None

    def test_detect_with_jax_gpus(self) -> None:
        """Test GPU detection via JAX devices."""
        detector = ClusterDetector()

        # Mock JAX devices to return GPUs
        mock_devices = [
            MagicMock(platform="gpu"),
            MagicMock(platform="gpu"),
            MagicMock(platform="gpu"),
            MagicMock(platform="gpu"),
        ]

        with patch("jax.devices", return_value=mock_devices):
            gpu_count = detector._detect_gpus_per_node()
            assert gpu_count == 4

    def test_detect_infiniband_interconnect(self, tmp_path: Path) -> None:
        """Test Infiniband interconnect detection."""
        # Create mock /sys/class/infiniband directory
        ib_path = tmp_path / "sys" / "class" / "infiniband"
        ib_path.mkdir(parents=True)

        with patch.object(Path, "exists", return_value=True):
            detector = ClusterDetector()
            # Directly test _detect_interconnect
            # Need to mock the Path check
            with patch("nlsq.core.workflow.Path") as mock_path:
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                mock_path.return_value = mock_path_instance
                interconnect = detector._detect_interconnect()
                # First check is /sys/class/infiniband
                assert interconnect == "infiniband"


class TestGetMultiGPUConfig:
    """Tests for get_multi_gpu_config function."""

    def test_returns_none_for_no_gpus(self) -> None:
        """Test that None is returned when no GPUs available."""
        cluster_info = ClusterInfo(
            node_count=1,
            gpus_per_node=0,
            total_gpus=0,
            node_list=["localhost"],
            scheduler="local",
        )

        config = get_multi_gpu_config(cluster_info)
        assert config is None

    def test_auto_detection_when_no_cluster_info(self) -> None:
        """Test auto-detection when cluster_info is None."""
        # When no cluster and no GPUs, should return None
        with patch.object(ClusterDetector, "detect", return_value=None):
            config = get_multi_gpu_config(None)
            assert config is None
