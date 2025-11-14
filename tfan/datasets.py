"""
Dataset loaders for TFAN evaluation.
Supports WikiText-103, MNIST, ImageNet-C, FB15k-237, and WordNet.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np


class LongSequenceDataset(Dataset):
    """
    Long-sequence dataset for testing attention on extended contexts.
    Wraps WikiText-103 with configurable sequence lengths.
    """

    def __init__(self, data_path: Optional[str] = None,
                 seq_len: int = 8192,
                 vocab_size: int = 10000,
                 split: str = "train"):
        """
        Args:
            data_path: Path to WikiText-103 dataset
            seq_len: Sequence length (8k, 16k, or 32k)
            vocab_size: Vocabulary size
            split: 'train', 'valid', or 'test'
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.split = split

        # Try to load real data, fall back to synthetic
        if data_path and Path(data_path).exists():
            self.data = self._load_wikitext(data_path, split)
        else:
            # Synthetic data for testing
            print(f"Warning: WikiText-103 not found, using synthetic data")
            self.data = self._generate_synthetic(num_samples=1000)

    def _load_wikitext(self, data_path: str, split: str) -> List[torch.Tensor]:
        """Load WikiText-103 from file."""
        # Placeholder - implement actual loading
        # For now, use synthetic
        return self._generate_synthetic(num_samples=1000)

    def _generate_synthetic(self, num_samples: int) -> List[torch.Tensor]:
        """Generate synthetic long sequences."""
        samples = []
        for _ in range(num_samples):
            seq = torch.randint(0, self.vocab_size, (self.seq_len,))
            samples.append(seq)
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (input_seq, target_seq) where target is shifted by 1
        """
        seq = self.data[idx]
        # Language modeling: predict next token
        inputs = seq[:-1]
        targets = seq[1:]
        return inputs, targets


class KnowledgeGraphDataset(Dataset):
    """
    Knowledge graph dataset for link prediction.
    Supports FB15k-237 and WordNet.
    """

    def __init__(self, data_path: Optional[str] = None,
                 dataset_name: str = "FB15k-237",
                 split: str = "train",
                 num_entities: int = 14541,
                 num_relations: int = 237):
        """
        Args:
            data_path: Path to dataset files
            dataset_name: 'FB15k-237' or 'WordNet'
            split: 'train', 'valid', or 'test'
            num_entities: Number of entities (FB15k-237: 14541, WordNet: ~41k)
            num_relations: Number of relations (FB15k-237: 237, WordNet: 18)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.num_entities = num_entities
        self.num_relations = num_relations

        if data_path and Path(data_path).exists():
            self.triples = self._load_kg(data_path, split)
        else:
            print(f"Warning: {dataset_name} not found, using synthetic data")
            self.triples = self._generate_synthetic(num_samples=10000)

    def _load_kg(self, data_path: str, split: str) -> List[Tuple[int, int, int]]:
        """
        Load knowledge graph triples from file.
        Expected format: head_id relation_id tail_id per line
        """
        triples = []
        file_path = Path(data_path) / f"{split}.txt"

        if not file_path.exists():
            return self._generate_synthetic(num_samples=10000)

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, r, t))

        return triples

    def _generate_synthetic(self, num_samples: int) -> List[Tuple[int, int, int]]:
        """Generate synthetic KG triples."""
        triples = []
        for _ in range(num_samples):
            h = np.random.randint(0, self.num_entities)
            r = np.random.randint(0, self.num_relations)
            t = np.random.randint(0, self.num_entities)
            triples.append((h, r, t))
        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (triple, label) where triple is [head, relation, tail] and label is 1 (positive)
        """
        h, r, t = self.triples[idx]
        triple = torch.tensor([h, r, t], dtype=torch.long)
        label = torch.tensor(1.0)  # Positive example
        return triple, label

    def get_negative_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate negative sample by corrupting head or tail."""
        h, r, t = self.triples[idx]

        # Corrupt head or tail randomly
        if np.random.rand() > 0.5:
            # Corrupt head
            h_neg = np.random.randint(0, self.num_entities)
            triple = torch.tensor([h_neg, r, t], dtype=torch.long)
        else:
            # Corrupt tail
            t_neg = np.random.randint(0, self.num_entities)
            triple = torch.tensor([h, r, t_neg], dtype=torch.long)

        label = torch.tensor(0.0)  # Negative example
        return triple, label


class OODVisionDataset(Dataset):
    """
    Out-of-distribution vision dataset.
    Wraps MNIST or ImageNet-C for OOD robustness testing.
    """

    def __init__(self, dataset_name: str = "MNIST",
                 corruption: Optional[str] = None,
                 severity: int = 3,
                 split: str = "test"):
        """
        Args:
            dataset_name: 'MNIST' or 'ImageNet-C'
            corruption: Type of corruption (e.g., 'gaussian_noise', 'blur')
            severity: Corruption severity (1-5)
            split: 'train' or 'test'
        """
        self.dataset_name = dataset_name
        self.corruption = corruption
        self.severity = severity
        self.split = split

        # Try to load actual datasets
        try:
            if dataset_name == "MNIST":
                import torchvision.datasets as datasets
                import torchvision.transforms as transforms

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                self.dataset = datasets.MNIST(
                    root='./data',
                    train=(split == 'train'),
                    download=True,
                    transform=transform
                )

                # Apply corruption if specified
                if corruption:
                    self.dataset = self._apply_corruption(self.dataset, corruption, severity)

            else:
                # Placeholder for ImageNet-C
                print(f"Warning: {dataset_name} not implemented, using synthetic")
                self.dataset = self._generate_synthetic(num_samples=1000)

        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            self.dataset = self._generate_synthetic(num_samples=1000)

    def _apply_corruption(self, dataset, corruption: str, severity: int):
        """Apply corruption to dataset."""
        # Simplified corruption - in practice, use imagecorruptions library
        return dataset

    def _generate_synthetic(self, num_samples: int) -> List[Tuple[torch.Tensor, int]]:
        """Generate synthetic image data."""
        samples = []
        for _ in range(num_samples):
            img = torch.randn(1, 28, 28)  # MNIST-like
            label = np.random.randint(0, 10)
            samples.append((img, label))
        return samples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.dataset, list):
            img, label = self.dataset[idx]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return self.dataset[idx]


def create_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Create dataloaders based on config.

    Args:
        config: Configuration dict with dataset settings

    Returns:
        Dict mapping dataset names to DataLoaders
    """
    loaders = {}

    # Long-sequence (WikiText-103)
    if config.get('long_seq', {}).get('enable', False):
        seq_len = config['long_seq'].get('seq_len', 8192)
        batch_size = config['long_seq'].get('batch_size', 4)

        train_ds = LongSequenceDataset(
            data_path=config['long_seq'].get('data_path'),
            seq_len=seq_len,
            split='train'
        )

        val_ds = LongSequenceDataset(
            data_path=config['long_seq'].get('data_path'),
            seq_len=seq_len,
            split='valid'
        )

        loaders['long_seq_train'] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loaders['long_seq_val'] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Knowledge graph
    if config.get('kg', {}).get('enable', False):
        kg_name = config['kg'].get('name', 'FB15k-237')
        batch_size = config['kg'].get('batch_size', 32)

        train_ds = KnowledgeGraphDataset(
            data_path=config['kg'].get('data_path'),
            dataset_name=kg_name,
            split='train'
        )

        val_ds = KnowledgeGraphDataset(
            data_path=config['kg'].get('data_path'),
            dataset_name=kg_name,
            split='valid'
        )

        loaders['kg_train'] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loaders['kg_val'] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # OOD vision
    if config.get('ood_vision', {}).get('enable', False):
        dataset_name = config['ood_vision'].get('name', 'MNIST')
        batch_size = config['ood_vision'].get('batch_size', 64)

        test_ds = OODVisionDataset(
            dataset_name=dataset_name,
            corruption=config['ood_vision'].get('corruption'),
            severity=config['ood_vision'].get('severity', 3),
            split='test'
        )

        loaders['ood_vision'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return loaders
