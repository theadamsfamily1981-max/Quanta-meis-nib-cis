"""
Meta-learning task sampling infrastructure for TFAN.
Creates episodic tasks from existing datasets for MAML training.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
import random

from .datasets import (
    LongSequenceDataset,
    KnowledgeGraphDataset,
    OODVisionDataset
)


class MetaTaskSampler:
    """
    Samples episodic tasks for meta-learning.

    Each task consists of:
    - Support set: K examples for adaptation
    - Query set: M examples for evaluation
    """

    def __init__(self,
                 base_dataset: Dataset,
                 num_shots: int = 5,
                 num_queries: int = 15,
                 task_batch_size: int = 1):
        """
        Args:
            base_dataset: Base dataset to sample from
            num_shots: Number of support examples per task (K-shot)
            num_queries: Number of query examples per task
            task_batch_size: Batch size for support/query sets
        """
        self.base_dataset = base_dataset
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.task_batch_size = task_batch_size

    def sample_task(self) -> Dict:
        """
        Sample a single task.

        Returns:
            Dict with 'support' and 'query' keys, each containing
            list of (input, target) tuples
        """
        # Sample indices
        total_samples = self.num_shots + self.num_queries
        indices = random.sample(range(len(self.base_dataset)), total_samples)

        # Split into support and query
        support_indices = indices[:self.num_shots]
        query_indices = indices[self.num_shots:]

        # Create support set
        support_data = []
        for idx in support_indices:
            sample = self.base_dataset[idx]
            if isinstance(sample, dict):
                # Handle dict-based datasets
                inputs = sample.get('input', sample.get('x', None))
                targets = sample.get('target', sample.get('y', None))
            else:
                # Handle tuple-based datasets
                inputs, targets = sample

            support_data.append((inputs, targets))

        # Create query set
        query_data = []
        for idx in query_indices:
            sample = self.base_dataset[idx]
            if isinstance(sample, dict):
                inputs = sample.get('input', sample.get('x', None))
                targets = sample.get('target', sample.get('y', None))
            else:
                inputs, targets = sample

            query_data.append((inputs, targets))

        return {
            'support': support_data,
            'query': query_data
        }

    def __iter__(self):
        """Iterate over tasks."""
        while True:
            yield self.sample_task()


class SinusoidRegressionTask(Dataset):
    """
    Sinusoidal regression task for meta-learning benchmarks.

    Each task is a different sinusoid: y = A * sin(x + phase)
    """

    def __init__(self,
                 amplitude_range: Tuple[float, float] = (0.1, 5.0),
                 phase_range: Tuple[float, float] = (0.0, np.pi),
                 x_range: Tuple[float, float] = (-5.0, 5.0),
                 num_samples: int = 100):
        """
        Args:
            amplitude_range: Range for amplitude A
            phase_range: Range for phase shift
            x_range: Range for input x
            num_samples: Number of samples per task
        """
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.x_range = x_range
        self.num_samples = num_samples

        # Sample task parameters
        self.amplitude = np.random.uniform(*amplitude_range)
        self.phase = np.random.uniform(*phase_range)

        # Generate data
        self.x = np.random.uniform(*x_range, size=num_samples)
        self.y = self.amplitude * np.sin(self.x + self.phase)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor([self.x[idx]], dtype=torch.float32)
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x, y


class MetaDatasetWrapper(Dataset):
    """
    Wraps a standard dataset to create meta-learning tasks.
    """

    def __init__(self,
                 dataset_type: str = "sinusoid",
                 num_tasks: int = 1000,
                 num_shots: int = 5,
                 num_queries: int = 15,
                 **dataset_kwargs):
        """
        Args:
            dataset_type: Type of base dataset
            num_tasks: Total number of tasks to generate
            num_shots: K-shot learning
            num_queries: Number of query samples
            dataset_kwargs: Additional kwargs for base dataset
        """
        self.dataset_type = dataset_type
        self.num_tasks = num_tasks
        self.num_shots = num_shots
        self.num_queries = num_queries

        # Create task generators
        self.task_generators = []
        for _ in range(num_tasks):
            if dataset_type == "sinusoid":
                task_dataset = SinusoidRegressionTask(**dataset_kwargs)
            elif dataset_type == "long_sequence":
                task_dataset = LongSequenceDataset(**dataset_kwargs)
            elif dataset_type == "knowledge_graph":
                task_dataset = KnowledgeGraphDataset(**dataset_kwargs)
            elif dataset_type == "ood_vision":
                task_dataset = OODVisionDataset(**dataset_kwargs)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            sampler = MetaTaskSampler(
                task_dataset,
                num_shots=num_shots,
                num_queries=num_queries
            )
            self.task_generators.append(sampler)

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, idx):
        """Sample a task from the idx-th task generator."""
        return self.task_generators[idx].sample_task()


def create_meta_dataloaders(
        dataset_type: str = "sinusoid",
        num_train_tasks: int = 10000,
        num_val_tasks: int = 1000,
        num_shots: int = 5,
        num_queries: int = 15,
        tasks_per_batch: int = 4,
        **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create meta-learning train and validation dataloaders.

    Args:
        dataset_type: Type of base dataset
        num_train_tasks: Number of training tasks
        num_val_tasks: Number of validation tasks
        num_shots: K-shot learning
        num_queries: Number of query samples per task
        tasks_per_batch: Number of tasks per meta-batch
        dataset_kwargs: Additional dataset arguments

    Returns:
        (meta_train_loader, meta_val_loader)
    """
    # Training tasks
    meta_train_dataset = MetaDatasetWrapper(
        dataset_type=dataset_type,
        num_tasks=num_train_tasks,
        num_shots=num_shots,
        num_queries=num_queries,
        **dataset_kwargs
    )

    meta_train_loader = DataLoader(
        meta_train_dataset,
        batch_size=tasks_per_batch,
        shuffle=True,
        num_workers=0,  # Tasks are already pre-generated
        collate_fn=lambda x: x  # Return list of tasks as-is
    )

    # Validation tasks
    meta_val_dataset = MetaDatasetWrapper(
        dataset_type=dataset_type,
        num_tasks=num_val_tasks,
        num_shots=num_shots,
        num_queries=num_queries,
        **dataset_kwargs
    )

    meta_val_loader = DataLoader(
        meta_val_dataset,
        batch_size=1,  # Evaluate one task at a time
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x
    )

    return meta_train_loader, meta_val_loader


class AdaptiveDifficultyTaskSampler:
    """
    Samples tasks with adaptive difficulty based on meta-learner performance.

    Harder tasks are sampled more frequently when the meta-learner is performing well.
    """

    def __init__(self,
                 task_generators: List[MetaTaskSampler],
                 initial_difficulties: Optional[List[float]] = None,
                 adaptation_rate: float = 0.1):
        """
        Args:
            task_generators: List of task generators
            initial_difficulties: Initial difficulty scores (0-1)
            adaptation_rate: Rate of difficulty adaptation
        """
        self.task_generators = task_generators
        self.num_tasks = len(task_generators)

        if initial_difficulties is None:
            # Start with uniform difficulties
            self.difficulties = [0.5] * self.num_tasks
        else:
            self.difficulties = initial_difficulties

        self.adaptation_rate = adaptation_rate
        self.task_performance = [[] for _ in range(self.num_tasks)]

    def sample_task(self, difficulty_target: float = 0.5) -> Tuple[int, Dict]:
        """
        Sample a task close to the target difficulty.

        Args:
            difficulty_target: Desired difficulty (0=easy, 1=hard)

        Returns:
            (task_idx, task_dict)
        """
        # Find task closest to target difficulty
        distances = [abs(d - difficulty_target) for d in self.difficulties]
        task_idx = np.argmin(distances)

        task = self.task_generators[task_idx].sample_task()

        return task_idx, task

    def update_difficulty(self, task_idx: int, performance: float):
        """
        Update task difficulty based on performance.

        Args:
            task_idx: Index of the task
            performance: Performance metric (lower is better, e.g., loss)
        """
        self.task_performance[task_idx].append(performance)

        # Compute average performance for this task
        avg_performance = np.mean(self.task_performance[task_idx][-10:])  # Last 10 samples

        # Update difficulty (higher performance → lower difficulty)
        # This is a simple heuristic
        if avg_performance < 0.5:  # Good performance
            self.difficulties[task_idx] = max(0.0, self.difficulties[task_idx] - self.adaptation_rate)
        else:  # Poor performance
            self.difficulties[task_idx] = min(1.0, self.difficulties[task_idx] + self.adaptation_rate)


class CurriculumMetaLearning:
    """
    Curriculum learning for meta-learning.

    Gradually increases task difficulty during meta-training.
    """

    def __init__(self,
                 easy_tasks: List[MetaTaskSampler],
                 medium_tasks: List[MetaTaskSampler],
                 hard_tasks: List[MetaTaskSampler],
                 curriculum_schedule: str = "linear"):
        """
        Args:
            easy_tasks: Easy task samplers
            medium_tasks: Medium difficulty task samplers
            hard_tasks: Hard task samplers
            curriculum_schedule: 'linear', 'exponential', or 'step'
        """
        self.easy_tasks = easy_tasks
        self.medium_tasks = medium_tasks
        self.hard_tasks = hard_tasks
        self.curriculum_schedule = curriculum_schedule
        self.meta_step = 0

    def get_task_distribution(self, progress: float) -> Dict[str, float]:
        """
        Get mixing ratios for task difficulties based on training progress.

        Args:
            progress: Training progress (0.0 to 1.0)

        Returns:
            Dict with 'easy', 'medium', 'hard' ratios
        """
        if self.curriculum_schedule == "linear":
            easy_ratio = max(0.0, 1.0 - progress)
            hard_ratio = min(1.0, progress)
            medium_ratio = 1.0 - easy_ratio - hard_ratio
        elif self.curriculum_schedule == "exponential":
            easy_ratio = np.exp(-3 * progress)
            hard_ratio = 1 - np.exp(-3 * progress)
            medium_ratio = 1.0 - easy_ratio - hard_ratio
        elif self.curriculum_schedule == "step":
            if progress < 0.33:
                easy_ratio, medium_ratio, hard_ratio = 0.7, 0.3, 0.0
            elif progress < 0.67:
                easy_ratio, medium_ratio, hard_ratio = 0.3, 0.5, 0.2
            else:
                easy_ratio, medium_ratio, hard_ratio = 0.1, 0.3, 0.6
        else:
            raise ValueError(f"Unknown schedule: {self.curriculum_schedule}")

        # Normalize
        total = easy_ratio + medium_ratio + hard_ratio
        return {
            'easy': easy_ratio / total,
            'medium': medium_ratio / total,
            'hard': hard_ratio / total
        }

    def sample_task(self, progress: float) -> Dict:
        """
        Sample a task according to curriculum.

        Args:
            progress: Training progress (0.0 to 1.0)

        Returns:
            Sampled task dict
        """
        distribution = self.get_task_distribution(progress)

        # Sample difficulty level
        rand = random.random()
        if rand < distribution['easy']:
            task_list = self.easy_tasks
        elif rand < distribution['easy'] + distribution['medium']:
            task_list = self.medium_tasks
        else:
            task_list = self.hard_tasks

        # Sample random task from selected difficulty
        task_sampler = random.choice(task_list)
        return task_sampler.sample_task()


if __name__ == "__main__":
    # Demo meta-task sampling
    print("=== Meta-Task Sampling Demo ===\n")

    # Create sinusoid regression tasks
    print("Creating 5-shot, 15-query sinusoid tasks...")
    meta_train_loader, meta_val_loader = create_meta_dataloaders(
        dataset_type="sinusoid",
        num_train_tasks=100,
        num_val_tasks=20,
        num_shots=5,
        num_queries=15,
        tasks_per_batch=4
    )

    # Sample a batch of tasks
    for batch_idx, task_batch in enumerate(meta_train_loader):
        print(f"\nTask batch {batch_idx+1}:")
        print(f"  Number of tasks: {len(task_batch)}")

        for task_idx, task in enumerate(task_batch):
            print(f"  Task {task_idx+1}:")
            print(f"    Support set size: {len(task['support'])}")
            print(f"    Query set size: {len(task['query'])}")

            # Show first support example
            x_sup, y_sup = task['support'][0]
            print(f"    Support example: x={x_sup.item():.3f}, y={y_sup.item():.3f}")

        if batch_idx >= 2:
            break

    print("\nMeta-task sampling demo complete!")
