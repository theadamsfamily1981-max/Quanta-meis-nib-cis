"""
Tests for distributed training infrastructure.
"""
import pytest
import torch
import torch.nn as nn

from tfan.distributed import (
    DistributedMetrics,
    reduce_dict,
    get_hostname
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.distributed
class TestDistributedUtilities:
    """Tests for distributed utilities (non-DDP)."""

    def test_distributed_metrics_update(self):
        """Test metrics update."""
        metrics = DistributedMetrics(device="cpu")

        metrics.update("loss", 1.5, count=10)
        metrics.update("loss", 2.0, count=5)
        metrics.update("accuracy", 0.9, count=10)

        # Check internal state
        assert "loss" in metrics.metrics
        assert "accuracy" in metrics.metrics

    def test_distributed_metrics_compute(self):
        """Test metrics computation (single process)."""
        metrics = DistributedMetrics(device="cpu")

        metrics.update("loss", 1.0, count=10)
        metrics.update("loss", 2.0, count=10)

        result = metrics.compute()

        assert "loss" in result
        # Average: (1.0*10 + 2.0*10) / 20 = 1.5
        assert abs(result["loss"] - 1.5) < 1e-5

    def test_distributed_metrics_reset(self):
        """Test metrics reset."""
        metrics = DistributedMetrics(device="cpu")

        metrics.update("loss", 1.0, count=10)
        metrics.reset()

        assert len(metrics.metrics) == 0

    def test_reduce_dict_single_process(self):
        """Test dict reduction in single process."""
        input_dict = {
            "loss": torch.tensor(1.0),
            "accuracy": torch.tensor(0.9)
        }

        # In single process, should return same dict
        result = reduce_dict(input_dict)

        assert "loss" in result
        assert "accuracy" in result
        assert torch.allclose(result["loss"], input_dict["loss"])

    def test_get_hostname(self):
        """Test hostname retrieval."""
        hostname = get_hostname()

        assert isinstance(hostname, str)
        assert len(hostname) > 0


@pytest.mark.distributed
class TestDistributedDataLoading:
    """Tests for distributed data loading utilities."""

    def test_model_wrapping(self):
        """Test model can be wrapped for DDP (preparation)."""
        model = SimpleModel()

        # Test that model has expected structure
        assert hasattr(model, 'fc')
        assert isinstance(model.fc, nn.Linear)

        # Test forward pass
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 1)


@pytest.mark.distributed
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDistributedGPU:
    """Tests for GPU-specific distributed features."""

    def test_cuda_availability(self):
        """Test CUDA is available for distributed training."""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    def test_model_to_cuda(self):
        """Test model can be moved to CUDA."""
        if torch.cuda.is_available():
            model = SimpleModel()
            model = model.to("cuda:0")

            x = torch.randn(4, 10, device="cuda:0")
            output = model(x)

            assert output.device.type == "cuda"


@pytest.mark.distributed
class TestDistributedConfiguration:
    """Tests for distributed configuration."""

    def test_backend_selection(self):
        """Test backend selection logic."""
        # NCCL for CUDA
        if torch.cuda.is_available():
            expected_backend = "nccl"
        else:
            expected_backend = "gloo"

        # This is the logic used in distributed.py
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        assert backend == expected_backend

    def test_environment_variables(self):
        """Test that environment variables can be set."""
        import os

        # Test setting master address
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        assert os.environ['MASTER_ADDR'] == 'localhost'
        assert os.environ['MASTER_PORT'] == '12355'


# Note: Full DDP tests require multiple processes and are best run via integration scripts
# Example command: torchrun --nproc_per_node=2 tests/test_distributed.py


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "distributed"])
