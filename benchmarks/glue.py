"""
GLUE Benchmark Suite
General Language Understanding Evaluation

9 tasks for comprehensive NLP evaluation:
1. CoLA - Linguistic Acceptability
2. SST-2 - Sentiment Analysis
3. MRPC - Paraphrase Detection
4. STS-B - Semantic Textual Similarity
5. QQP - Quora Question Pairs
6. MNLI - Natural Language Inference
7. QNLI - Question NLI
8. RTE - Recognizing Textual Entailment
9. WNLI - Winograd NLI
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GLUEExample:
    """Single GLUE example."""
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass
class GLUEFeatures:
    """Features for a single example."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label: Optional[int] = None


class GLUEDataset(Dataset):
    """Base dataset class for GLUE tasks."""

    def __init__(
        self,
        examples: List[GLUEExample],
        tokenizer,
        max_length: int = 128,
        label_map: Optional[Dict] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize
        if example.text_b:
            encoding = self.tokenizer(
                example.text_a,
                example.text_b,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                example.text_a,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        # Prepare features
        features = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if 'token_type_ids' in encoding:
            features['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        # Add label
        if example.label is not None:
            if self.label_map:
                label = self.label_map.get(example.label, 0)
            else:
                label = float(example.label) if '.' in str(example.label) else int(example.label)
            features['label'] = torch.tensor(label)

        return features


# ============================================================================
# Task-Specific Processors
# ============================================================================

class GLUEProcessor:
    """Base processor for GLUE tasks."""

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        raise NotImplementedError

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        raise NotImplementedError

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        raise NotImplementedError

    def _read_tsv(self, input_file: str) -> List[List[str]]:
        """Read TSV file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            return [line.strip().split('\t') for line in f]


class ColaProcessor(GLUEProcessor):
    """
    Processor for CoLA (Corpus of Linguistic Acceptability).
    Binary classification: acceptable vs unacceptable.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[3] if len(line) > 3 else line[1]
            label = line[1] if set_type != "test" else "0"
            examples.append(GLUEExample(guid=guid, text_a=text_a, label=label))
        return examples


class Sst2Processor(GLUEProcessor):
    """
    Processor for SST-2 (Stanford Sentiment Treebank).
    Binary classification: positive vs negative sentiment.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1] if set_type != "test" else "0"
            examples.append(GLUEExample(guid=guid, text_a=text_a, label=label))
        return examples


class MrpcProcessor(GLUEProcessor):
    """
    Processor for MRPC (Microsoft Research Paraphrase Corpus).
    Binary classification: paraphrase vs not paraphrase.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = line[0] if set_type != "test" else "0"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StsbProcessor(GLUEProcessor):
    """
    Processor for STS-B (Semantic Textual Similarity Benchmark).
    Regression task: similarity score 0-5.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return [None]  # Regression

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{i}"
            text_a = line[7]
            text_b = line[8]
            label = line[9] if set_type != "test" else "0.0"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(GLUEProcessor):
    """
    Processor for QQP (Quora Question Pairs).
    Binary classification: duplicate vs not duplicate.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            try:
                guid = f"{set_type}-{i}"
                text_a = line[3]
                text_b = line[4]
                label = line[5] if set_type != "test" else "0"
                examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except IndexError:
                continue
        return examples


class MnliProcessor(GLUEProcessor):
    """
    Processor for MNLI (Multi-Genre Natural Language Inference).
    Three-way classification: entailment, contradiction, neutral.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched"
        )

    def get_labels(self) -> List[str]:
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{line[0]}"
            text_a = line[8]
            text_b = line[9]
            label = line[-1] if set_type != "test_matched" else "neutral"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(GLUEProcessor):
    """
    Processor for QNLI (Question Natural Language Inference).
    Binary classification: entailment vs not_entailment.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "not_entailment"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(GLUEProcessor):
    """
    Processor for RTE (Recognizing Textual Entailment).
    Binary classification: entailment vs not_entailment.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "not_entailment"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(GLUEProcessor):
    """
    Processor for WNLI (Winograd Natural Language Inference).
    Binary classification: entailment vs not_entailment.
    """

    def get_train_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[GLUEExample]:
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[GLUEExample]:
        examples = []
        for i, line in enumerate(lines[1:]):  # Skip header
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "0"
            examples.append(GLUEExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(task_name: str, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute task-specific metrics."""

    assert len(preds) == len(labels)

    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}

    elif task_name == "sst-2":
        return {"acc": (preds == labels).mean()}

    elif task_name == "mrpc":
        return {
            "acc": (preds == labels).mean(),
            "f1": f1_score(labels, preds)
        }

    elif task_name == "sts-b":
        return {
            "pearson": pearsonr(preds, labels)[0],
            "spearman": spearmanr(preds, labels)[0]
        }

    elif task_name == "qqp":
        return {
            "acc": (preds == labels).mean(),
            "f1": f1_score(labels, preds)
        }

    elif task_name == "mnli":
        return {"acc": (preds == labels).mean()}

    elif task_name == "qnli":
        return {"acc": (preds == labels).mean()}

    elif task_name == "rte":
        return {"acc": (preds == labels).mean()}

    elif task_name == "wnli":
        return {"acc": (preds == labels).mean()}

    else:
        raise KeyError(f"Unknown task: {task_name}")


# ============================================================================
# Task Registry
# ============================================================================

GLUE_TASKS = {
    "cola": ColaProcessor,
    "sst-2": Sst2Processor,
    "mrpc": MrpcProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "mnli": MnliProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor
}


def load_glue_task(
    task_name: str,
    data_dir: str,
    tokenizer,
    split: str = "train",
    max_length: int = 128
) -> GLUEDataset:
    """
    Load a GLUE task dataset.

    Args:
        task_name: Task name (e.g., "cola", "sst-2")
        data_dir: Data directory
        tokenizer: Tokenizer instance
        split: "train", "dev", or "test"
        max_length: Maximum sequence length

    Returns:
        GLUEDataset instance
    """
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(GLUE_TASKS.keys())}")

    processor = GLUE_TASKS[task_name]()

    # Load examples
    if split == "train":
        examples = processor.get_train_examples(data_dir)
    elif split == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif split == "test":
        examples = processor.get_test_examples(data_dir)
    else:
        raise ValueError(f"Invalid split: {split}")

    # Create label map
    labels = processor.get_labels()
    if labels[0] is not None:
        label_map = {label: i for i, label in enumerate(labels)}
    else:
        label_map = None

    logger.info(f"Loaded {len(examples)} examples for {task_name}/{split}")

    return GLUEDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        label_map=label_map
    )
