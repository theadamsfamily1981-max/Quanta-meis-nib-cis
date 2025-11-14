"""
Integration tests for MAML meta-learning.
"""
import pytest
import torch
import torch.nn as nn

from tfan.meta_trainer import MAMLTrainer, compute_adaptation_metrics
from tfan.meta_datasets import (
    create_meta_dataloaders,
    SinusoidRegressionTask,
    MetaTaskSampler
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.meta
class TestMAMLTrainer:
    """Tests for MAML trainer."""

    def test_maml_initialization(self):
        """Test MAML trainer initialization."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.01,
            num_inner_steps=5,
            first_order=True
        )

        assert maml.inner_lr == 0.01
        assert maml.num_inner_steps == 5
        assert maml.first_order == True
        assert maml.meta_step == 0

    def test_inner_loop_adaptation(self):
        """Test inner loop adaptation on support set."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.01,
            num_inner_steps=5
        )

        # Create simple support data
        support_data = [
            (torch.randn(1, 1), torch.randn(1, 1))
            for _ in range(5)
        ]

        criterion = nn.MSELoss()

        # Adapt model
        import copy
        task_model = copy.deepcopy(model)
        adapted_model = maml.inner_loop(task_model, support_data, criterion)

        assert adapted_model is not None

    def test_meta_train_step(self):
        """Test single meta-training step."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.01,
            num_inner_steps=3,
            first_order=True  # Faster for testing
        )

        # Create task batch
        task_batch = []
        for _ in range(2):  # 2 tasks
            task = {
                'support': [(torch.randn(1, 1), torch.randn(1, 1)) for _ in range(3)],
                'query': [(torch.randn(1, 1), torch.randn(1, 1)) for _ in range(5)]
            }
            task_batch.append(task)

        criterion = nn.MSELoss()

        # Meta-training step
        stats = maml.meta_train_step(task_batch, criterion)

        assert 'meta_loss' in stats
        assert 'task_losses_mean' in stats
        assert maml.meta_step == 1

    def test_fast_adapt(self):
        """Test fast adaptation to new task."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.01,
            num_inner_steps=5
        )

        # Create support data
        support_data = [
            (torch.randn(1, 1), torch.randn(1, 1))
            for _ in range(5)
        ]

        criterion = nn.MSELoss()

        # Fast adapt
        adapted_model = maml.fast_adapt(support_data, criterion)

        assert adapted_model is not None
        adapted_model.eval()

        # Test inference
        test_input = torch.randn(1, 1)
        output = adapted_model(test_input)
        assert output.shape == (1, 1)


@pytest.mark.meta
class TestMetaDatasets:
    """Tests for meta-learning datasets."""

    def test_sinusoid_task(self):
        """Test sinusoid regression task generation."""
        task = SinusoidRegressionTask(num_samples=50)

        assert len(task) == 50
        x, y = task[0]
        assert x.shape == (1,)
        assert y.shape == (1,)

    def test_meta_task_sampler(self):
        """Test meta-task sampling."""
        base_dataset = SinusoidRegressionTask(num_samples=100)

        sampler = MetaTaskSampler(
            base_dataset,
            num_shots=5,
            num_queries=15
        )

        task = sampler.sample_task()

        assert 'support' in task
        assert 'query' in task
        assert len(task['support']) == 5
        assert len(task['query']) == 15

    def test_meta_dataloaders(self):
        """Test meta-dataloader creation."""
        meta_train_loader, meta_val_loader = create_meta_dataloaders(
            dataset_type="sinusoid",
            num_train_tasks=50,
            num_val_tasks=10,
            num_shots=5,
            num_queries=15,
            tasks_per_batch=4
        )

        # Check train loader
        task_batch = next(iter(meta_train_loader))
        assert len(task_batch) == 4

        # Check first task
        task = task_batch[0]
        assert len(task['support']) == 5
        assert len(task['query']) == 15


@pytest.mark.meta
@pytest.mark.slow
class TestMAMLIntegration:
    """Integration tests for complete MAML training."""

    def test_maml_training_loop(self):
        """Test complete MAML training loop."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.01,
            num_inner_steps=3,
            first_order=True,
            use_fdt_meta=False  # Disable for faster testing
        )

        # Create small meta-dataloaders
        meta_train_loader, meta_val_loader = create_meta_dataloaders(
            dataset_type="sinusoid",
            num_train_tasks=20,
            num_val_tasks=5,
            num_shots=5,
            num_queries=10,
            tasks_per_batch=2
        )

        criterion = nn.MSELoss()

        # Train for few epochs
        history = maml.meta_train(
            meta_train_loader,
            meta_val_loader,
            criterion,
            num_meta_epochs=2,
            tasks_per_batch=2,
            eval_every=5
        )

        # Check history
        assert 'meta_train_losses' in history
        assert 'meta_val_losses' in history
        assert len(history['meta_train_losses']) > 0

    def test_adaptation_improvement(self):
        """Test that adaptation improves task performance."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        maml = MAMLTrainer(
            model=model,
            meta_optimizer=optimizer,
            inner_lr=0.05,
            num_inner_steps=10,
            first_order=True
        )

        # Create task
        task_dataset = SinusoidRegressionTask(num_samples=50)
        support_data = [task_dataset[i] for i in range(5)]
        test_data = [task_dataset[i+40] for i in range(5)]

        criterion = nn.MSELoss()

        # Model before adaptation
        import copy
        model_before = copy.deepcopy(maml.model)
        model_before.eval()

        # Adapt model
        model_after = maml.fast_adapt(support_data, criterion, num_steps=10)

        # Compute metrics
        metrics = compute_adaptation_metrics(
            model_before,
            model_after,
            test_data,
            criterion
        )

        # After adaptation should be better
        assert metrics['loss_after_adaptation'] < metrics['loss_before_adaptation']
        assert metrics['improvement_percent'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "meta"])
