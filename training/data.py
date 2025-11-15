"""
Data loading utilities for TF-A-N 7B training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Dict, Iterator
import numpy as np


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for testing/demo.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer (with encode method)
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate/pad
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            # Pad with pad_token_id (assume 0)
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # For causal LM, labels = input_ids shifted
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class TokenizedDataset(IterableDataset):
    """
    Streaming dataset for pre-tokenized data.

    Assumes data is stored as .bin files with token IDs.

    Args:
        data_path: Path to tokenized data file
        seq_length: Sequence length for chunks
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int = 2048,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.seed = seed

        # Load data
        try:
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
        except Exception as e:
            print(f"Warning: Could not load data from {data_path}: {e}")
            print("Creating dummy data for testing...")
            # Create dummy data
            self.tokens = np.random.randint(0, 32768, size=1000000, dtype=np.uint16)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataset, yielding sequences of length seq_length.
        """
        rng = np.random.RandomState(self.seed)

        while True:
            # Random starting position
            start_idx = rng.randint(0, len(self.tokens) - self.seq_length - 1)

            # Extract sequence
            chunk = self.tokens[start_idx : start_idx + self.seq_length + 1]

            # Input and labels (shifted by 1)
            input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
            labels = torch.from_numpy(chunk[1:].astype(np.int64))

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create DataLoader with standard settings.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        dataloader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class DummyDataset(IterableDataset):
    """
    Dummy dataset for testing without real data.

    Generates random token sequences.

    Args:
        vocab_size: Vocabulary size
        seq_length: Sequence length
        num_samples: Number of samples (None for infinite)
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        seq_length: int = 2048,
        num_samples: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0
        while self.num_samples is None or count < self.num_samples:
            # Generate random tokens
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

            count += 1


__all__ = [
    "SimpleTextDataset",
    "TokenizedDataset",
    "DummyDataset",
    "create_dataloader",
]
