"""
Continuous Adaptation Framework for TFAN.
Enables online learning, continual learning, and self-improvement in production.

Key Features:
- Online meta-learning
- Catastrophic forgetting prevention
- Experience replay
- Elastic weight consolidation (EWC)
- Progressive neural networks
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
import copy
import numpy as np

from .meta_trainer import MAMLTrainer
from .trainer import FDTScheduler


class ExperienceReplay:
    """
    Experience replay buffer for continual learning.

    Stores past experiences to prevent catastrophic forgetting.
    """

    def __init__(self, capacity: int = 10000, prioritized: bool = False):
        """
        Args:
            capacity: Maximum buffer size
            prioritized: Use prioritized sampling
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer: Deque = deque(maxlen=capacity)
        self.priorities: Deque = deque(maxlen=capacity) if prioritized else None

    def add(self, experience: Tuple, priority: float = 1.0):
        """
        Add experience to buffer.

        Args:
            experience: (state, action, reward, next_state) tuple
            priority: Priority for prioritized replay
        """
        self.buffer.append(experience)

        if self.prioritized:
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> List:
        """
        Sample batch from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.prioritized:
            # Prioritized sampling
            priorities = np.array(list(self.priorities))
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

    Maintains importance of weights for previous tasks.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        """
        Args:
            model: Neural network model
            lambda_ewc: EWC regularization strength
        """
        self.model = model
        self.lambda_ewc = lambda_ewc

        # Storage for Fisher information and optimal parameters
        self.fisher_info = {}
        self.optimal_params = {}

    def compute_fisher_information(self, dataloader, num_samples: int = 1000):
        """
        Compute Fisher information matrix diagonal.

        Args:
            dataloader: DataLoader for current task
            num_samples: Number of samples to use
        """
        print(f"Computing Fisher information ({num_samples} samples)...")

        self.model.eval()

        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            self.fisher_info[name] = torch.zeros_like(param)

        # Accumulate gradients
        num_processed = 0

        for inputs, targets in dataloader:
            if num_processed >= num_samples:
                break

            self.model.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Use log probabilities for Fisher
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Sample from distribution
            sampled = torch.multinomial(torch.exp(log_probs), 1).squeeze()

            # Backward pass
            loss = -log_probs.gather(1, sampled.unsqueeze(1)).mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.pow(2) / num_samples

            num_processed += inputs.size(0)

        # Store optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        print("Fisher information computed")

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Returns:
            EWC loss term
        """
        loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                # Penalize changes to important weights
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]

                loss += (fisher * (param - optimal).pow(2)).sum()

        return self.lambda_ewc * loss


class OnlineMetaLearner:
    """
    Online meta-learning for continuous adaptation.

    Adapts to new data distributions without forgetting previous knowledge.
    """

    def __init__(self,
                 model: nn.Module,
                 meta_lr: float = 1e-3,
                 inner_lr: float = 0.01,
                 num_inner_steps: int = 5,
                 use_ewc: bool = True,
                 ewc_lambda: float = 1000.0,
                 replay_capacity: int = 10000):
        """
        Args:
            model: Model to adapt
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            num_inner_steps: Number of inner loop steps
            use_ewc: Use EWC for forgetting prevention
            ewc_lambda: EWC regularization strength
            replay_capacity: Experience replay buffer size
        """
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

        # EWC
        self.use_ewc = use_ewc
        if use_ewc:
            self.ewc = ElasticWeightConsolidation(model, lambda_ewc=ewc_lambda)
        else:
            self.ewc = None

        # Experience replay
        self.replay_buffer = ExperienceReplay(capacity=replay_capacity)

        # Statistics
        self.adaptation_count = 0
        self.task_boundaries = []

    def online_adapt(self,
                    new_data: List[Tuple],
                    task_boundary: bool = False) -> Dict:
        """
        Online adaptation to new data.

        Args:
            new_data: List of (input, target) tuples
            task_boundary: Whether this is a new task (compute Fisher)

        Returns:
            Adaptation statistics
        """
        # Add to replay buffer
        for data in new_data:
            self.replay_buffer.add(data)

        # If task boundary, compute Fisher information
        if task_boundary and self.ewc is not None:
            # Create temporary dataloader
            from torch.utils.data import DataLoader, TensorDataset

            if len(self.replay_buffer) > 0:
                samples = self.replay_buffer.sample(min(len(self.replay_buffer), 1000))
                inputs = torch.stack([s[0] for s in samples])
                targets = torch.stack([s[1] for s in samples])

                temp_dataset = TensorDataset(inputs, targets)
                temp_loader = DataLoader(temp_dataset, batch_size=32)

                self.ewc.compute_fisher_information(temp_loader)
                self.task_boundaries.append(self.adaptation_count)

        # Inner loop adaptation on new data
        self.model.train()

        for step in range(self.num_inner_steps):
            # Sample from new data
            batch_new = new_data[:min(len(new_data), 32)]

            # Sample from replay buffer
            batch_replay = self.replay_buffer.sample(min(len(self.replay_buffer), 32))

            # Combine batches
            all_data = batch_new + batch_replay

            # Forward pass
            total_loss = 0.0

            for inputs, targets in all_data:
                outputs = self.model(inputs.unsqueeze(0))
                loss = nn.functional.mse_loss(outputs, targets.unsqueeze(0))
                total_loss += loss

            # Add EWC regularization
            if self.ewc is not None and len(self.ewc.fisher_info) > 0:
                ewc_loss = self.ewc.ewc_loss()
                total_loss += ewc_loss

            # Backward pass
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            self.meta_optimizer.step()

        self.adaptation_count += 1

        stats = {
            "adaptation_count": self.adaptation_count,
            "replay_buffer_size": len(self.replay_buffer),
            "num_task_boundaries": len(self.task_boundaries)
        }

        return stats

    def evaluate_forgetting(self,
                          task_dataloaders: List) -> Dict[int, float]:
        """
        Evaluate catastrophic forgetting on previous tasks.

        Args:
            task_dataloaders: List of dataloaders for each task

        Returns:
            Dict mapping task_id to accuracy
        """
        self.model.eval()
        task_accuracies = {}

        with torch.no_grad():
            for task_id, dataloader in enumerate(task_dataloaders):
                correct = 0
                total = 0

                for inputs, targets in dataloader:
                    outputs = self.model(inputs)
                    predicted = outputs.argmax(dim=-1)

                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

                accuracy = correct / total if total > 0 else 0.0
                task_accuracies[task_id] = accuracy

        return task_accuracies


class ProgressiveNeuralNetwork:
    """
    Progressive Neural Networks for continual learning.

    Adds new capacity for each task while preserving previous knowledge.
    """

    def __init__(self, base_model_factory, max_tasks: int = 10):
        """
        Args:
            base_model_factory: Function that creates a new column
            max_tasks: Maximum number of tasks/columns
        """
        self.base_model_factory = base_model_factory
        self.max_tasks = max_tasks

        # List of columns (one per task)
        self.columns: List[nn.Module] = []

        # Lateral connections between columns
        self.lateral_connections: List[nn.Module] = []

    def add_task(self) -> int:
        """
        Add a new column for a new task.

        Returns:
            Task ID
        """
        if len(self.columns) >= self.max_tasks:
            raise ValueError(f"Maximum tasks ({self.max_tasks}) reached")

        # Create new column
        new_column = self.base_model_factory()
        self.columns.append(new_column)

        # Create lateral connections from previous columns
        if len(self.columns) > 1:
            # Simple linear projection from previous columns
            lateral = nn.Linear(
                in_features=(len(self.columns) - 1) * 64,  # Assume 64-dim hidden
                out_features=64
            )
            self.lateral_connections.append(lateral)

        task_id = len(self.columns) - 1
        print(f"Added task {task_id} | Total columns: {len(self.columns)}")

        return task_id

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass for specific task.

        Args:
            x: Input tensor
            task_id: Task to execute

        Returns:
            Output tensor
        """
        if task_id >= len(self.columns):
            raise ValueError(f"Task {task_id} not available")

        # Get features from current column
        current_features = self.columns[task_id](x)

        # If not first task, add lateral connections
        if task_id > 0:
            prev_features = []

            for prev_id in range(task_id):
                prev_feat = self.columns[prev_id](x)
                prev_features.append(prev_feat)

            # Concatenate previous features
            prev_features_cat = torch.cat(prev_features, dim=-1)

            # Apply lateral connection
            lateral_feat = self.lateral_connections[task_id - 1](prev_features_cat)

            # Combine with current features
            output = current_features + lateral_feat
        else:
            output = current_features

        return output


class ContinualLearningTrainer:
    """
    Complete continual learning training system.

    Combines multiple strategies for effective lifelong learning.
    """

    def __init__(self,
                 model: nn.Module,
                 strategy: str = "ewc",
                 meta_lr: float = 1e-3,
                 ewc_lambda: float = 1000.0):
        """
        Args:
            model: Model to train
            strategy: Continual learning strategy ('ewc', 'replay', 'online_maml', 'progressive')
            meta_lr: Learning rate
            ewc_lambda: EWC regularization strength
        """
        self.model = model
        self.strategy = strategy

        if strategy == "online_maml":
            self.learner = OnlineMetaLearner(
                model,
                meta_lr=meta_lr,
                use_ewc=True,
                ewc_lambda=ewc_lambda
            )
        elif strategy == "ewc":
            self.ewc = ElasticWeightConsolidation(model, lambda_ewc=ewc_lambda)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        elif strategy == "replay":
            self.replay_buffer = ExperienceReplay(capacity=10000)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

        self.task_id = 0

    def train_task(self,
                  dataloader,
                  num_epochs: int = 10,
                  task_boundary: bool = True) -> Dict:
        """
        Train on a new task.

        Args:
            dataloader: DataLoader for task
            num_epochs: Number of epochs
            task_boundary: Whether this is a new task

        Returns:
            Training statistics
        """
        print(f"\n=== Training Task {self.task_id} ===")

        if self.strategy == "online_maml":
            # Online meta-learning
            for epoch in range(num_epochs):
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    # Convert to list of tuples
                    new_data = [(inputs[i], targets[i]) for i in range(len(inputs))]

                    # Online adaptation
                    stats = self.learner.online_adapt(
                        new_data,
                        task_boundary=(epoch == 0 and batch_idx == 0 and task_boundary)
                    )

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Replay buffer: {stats['replay_buffer_size']}")

        elif self.strategy == "ewc":
            # EWC training
            if task_boundary and self.task_id > 0:
                self.ewc.compute_fisher_information(dataloader)

            for epoch in range(num_epochs):
                epoch_loss = 0.0

                for inputs, targets in dataloader:
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)

                    # Add EWC loss
                    if self.task_id > 0:
                        ewc_loss = self.ewc.ewc_loss()
                        loss += ewc_loss

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")

        self.task_id += 1

        return {"task_id": self.task_id - 1, "epochs": num_epochs}


if __name__ == "__main__":
    # Demo continuous adaptation
    print("=== Continuous Adaptation Demo ===\n")

    # Test experience replay
    print("1. Experience Replay")
    replay = ExperienceReplay(capacity=100)

    for i in range(50):
        replay.add((torch.randn(10), torch.randn(1)), priority=np.random.rand())

    batch = replay.sample(10)
    print(f"Buffer size: {len(replay)}")
    print(f"Sampled batch: {len(batch)} experiences")

    # Test EWC
    print("\n2. Elastic Weight Consolidation")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = TinyModel()
    ewc = ElasticWeightConsolidation(model, lambda_ewc=100.0)

    print("EWC initialized")
    print(f"Lambda: {ewc.lambda_ewc}")

    print("\nContinuous adaptation demo complete!")
