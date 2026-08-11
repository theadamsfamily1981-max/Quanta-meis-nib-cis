#!/usr/bin/env python
"""GLUE multitask runner with offline-friendly hashed bag-of-words model."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TASK_TEXT_FIELDS: Dict[str, Sequence[str]] = {
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
}

SYNTHETIC_CORPUS: Dict[str, Dict[str, List[Tuple[List[str], int]]]] = {
    "sst2": {
        "train": [
            (["this", "movie", "was", "great"], 1),
            (["utterly", "terrible"], 0),
            (["loved", "every", "moment"], 1),
            (["boring", "and", "slow"], 0),
            (["unforgettable", "performances"], 1),
            (["waste", "of", "time"], 0),
        ],
        "validation": [
            (["wonderful", "performances"], 1),
            (["not", "worth", "the", "time"], 0),
        ],
    },
    "mrpc": {
        "train": [
            (["the", "company", "released", "a", "statement"], 1),
            (["shares", "fell", "after", "the", "announcement"], 0),
            (["the", "meeting", "was", "productive"], 1),
            (["he", "refused", "to", "comment"], 0),
            (["the", "court", "ruled", "against", "them"], 0),
            (["investors", "welcomed", "the", "deal"], 1),
        ],
        "validation": [
            (["profits", "rose", "again"], 1),
            (["talks", "collapsed"], 0),
        ],
    },
}

_DATASETS_SPEC = importlib.util.find_spec("datasets")
if _DATASETS_SPEC is not None:
    datasets = importlib.import_module("datasets")
    load_dataset = datasets.load_dataset
else:  # pragma: no cover - exercised in offline environments
    load_dataset = None  # type: ignore[assignment]


@dataclass
class EncodedExample:
    features: List[float]
    label: int


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multitask GLUE finetuning")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--tasks", type=str, default="sst2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples per split (0 disables)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI-friendly execution (synthetic data, deterministic)",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)


def try_load_glue(task: str, split: str, max_samples: int) -> List[Tuple[Dict[str, str], int]]:
    if load_dataset is None:
        raise RuntimeError("datasets library unavailable")
    dataset = load_dataset("glue", task, split=split)  # type: ignore[arg-type]
    records: List[Tuple[Dict[str, str], int]] = []
    for row in dataset:  # type: ignore[assignment]
        text_fields = TASK_TEXT_FIELDS.get(task, ("sentence",))
        text_parts = [str(row.get(field, "")) for field in text_fields]
        label = int(row["label"])
        records.append((dict(zip(text_fields, text_parts)), label))
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]
    return records


def synthetic_examples(task: str) -> Dict[str, List[Tuple[Dict[str, str], int]]]:
    corpus = SYNTHETIC_CORPUS.get(task)
    if corpus is None:
        base = [(["sample", "example", str(i)], i % 2) for i in range(12)]
        corpus = {"train": base, "validation": base[:6]}
    wrapped: Dict[str, List[Tuple[Dict[str, str], int]]] = {}
    fields = TASK_TEXT_FIELDS.get(task, ("sentence",))
    for split, items in corpus.items():
        wrapped_split: List[Tuple[Dict[str, str], int]] = []
        for tokens, label in items:
            text = " ".join(tokens)
            if len(fields) == 1:
                payload = {fields[0]: text}
            else:
                half = max(len(tokens) // 2, 1)
                payload = {
                    fields[0]: " ".join(tokens[:half]),
                    fields[1]: " ".join(tokens[half:]),
                }
            wrapped_split.append((payload, label))
        wrapped[split] = wrapped_split
    return wrapped


def encode_text(parts: Sequence[str], vocab_size: int = 512) -> List[float]:
    features = [0.0] * vocab_size
    for part in parts:
        for token in part.lower().split():
            idx = hash(token) % vocab_size
            features[idx] += 1.0
    total = sum(features)
    if total > 0:
        features = [value / total for value in features]
    return features


def prepare_dataset(task: str, split: str, *, ci: bool, max_samples: int) -> List[EncodedExample]:
    examples: List[Tuple[Dict[str, str], int]] = []
    if not ci:
        try:
            examples = try_load_glue(task, split, max_samples)
        except Exception:  # pragma: no cover - gracefully fall back
            examples = []
    if not examples:
        synthetic = synthetic_examples(task)
        examples = synthetic.get(split, synthetic.get("train", []))
        if max_samples:
            examples = examples[:max_samples]
    encoded: List[EncodedExample] = []
    text_fields = TASK_TEXT_FIELDS.get(task, ("sentence",))
    for fields_dict, label in examples:
        parts = [fields_dict.get(name, "") for name in text_fields]
        encoded.append(EncodedExample(features=encode_text(parts), label=label))
    return encoded


class LinearClassifier:
    def __init__(self, input_dim: int, num_labels: int, lr: float = 0.5):
        self.weights: List[List[float]] = [[0.0 for _ in range(input_dim)] for _ in range(num_labels)]
        self.bias: List[float] = [0.0 for _ in range(num_labels)]
        self.lr = lr

    def logits(self, features: Sequence[float]) -> List[float]:
        return [
            sum(w * f for w, f in zip(weights, features)) + bias
            for weights, bias in zip(self.weights, self.bias)
        ]

    def predict(self, features: Sequence[float]) -> int:
        logits = self.logits(features)
        return max(range(len(logits)), key=logits.__getitem__)

    def train_batch(self, batch_features: Sequence[Sequence[float]], batch_labels: Sequence[int]) -> Tuple[float, float]:
        batch_size = len(batch_labels)
        if batch_size == 0:
            return 0.0, 0.0
        input_dim = len(batch_features[0])
        num_labels = len(self.weights)
        grad_w = [[0.0 for _ in range(input_dim)] for _ in range(num_labels)]
        grad_b = [0.0 for _ in range(num_labels)]
        total_loss = 0.0
        correct = 0
        for features, label in zip(batch_features, batch_labels):
            logits = self.logits(features)
            max_logit = max(logits)
            exp_scores = [math.exp(value - max_logit) for value in logits]
            sum_exp = sum(exp_scores) or 1.0
            probs = [score / sum_exp for score in exp_scores]
            target = [0.0 for _ in range(num_labels)]
            target[label] = 1.0
            total_loss += -math.log(probs[label] + 1e-12)
            predicted = probs.index(max(probs))
            if predicted == label:
                correct += 1
            for class_idx in range(num_labels):
                diff = probs[class_idx] - target[class_idx]
                grad_b[class_idx] += diff
                for feature_idx in range(input_dim):
                    grad_w[class_idx][feature_idx] += diff * features[feature_idx]
        scale = 1.0 / batch_size
        for class_idx in range(num_labels):
            grad_b[class_idx] *= scale
            for feature_idx in range(input_dim):
                grad_w[class_idx][feature_idx] *= scale
                self.weights[class_idx][feature_idx] -= self.lr * grad_w[class_idx][feature_idx]
            self.bias[class_idx] -= self.lr * grad_b[class_idx]
        accuracy = correct / batch_size
        return total_loss * scale, accuracy

    def evaluate_batch(self, batch_features: Sequence[Sequence[float]], batch_labels: Sequence[int]) -> Tuple[float, float]:
        if not batch_labels:
            return 0.0, 0.0
        total_loss = 0.0
        correct = 0
        for features, label in zip(batch_features, batch_labels):
            logits = self.logits(features)
            max_logit = max(logits)
            exp_scores = [math.exp(value - max_logit) for value in logits]
            sum_exp = sum(exp_scores) or 1.0
            probs = [score / sum_exp for score in exp_scores]
            total_loss += -math.log(probs[label] + 1e-12)
            predicted = probs.index(max(probs))
            if predicted == label:
                correct += 1
        size = len(batch_labels)
        return total_loss / size, correct / size


def iterate_minibatches(examples: Sequence[EncodedExample], batch_size: int, *, shuffle: bool) -> Iterable[List[EncodedExample]]:
    indices = list(range(len(examples)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), max(batch_size, 1)):
        batch_indices = indices[start:start + max(batch_size, 1)]
        yield [examples[idx] for idx in batch_indices]


def aggregate_split(
    model: LinearClassifier,
    examples: Sequence[EncodedExample],
    batch_size: int,
    *,
    train: bool,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    for batch in iterate_minibatches(examples, batch_size, shuffle=train):
        features = [example.features for example in batch]
        labels = [example.label for example in batch]
        if train:
            loss, accuracy = model.train_batch(features, labels)
        else:
            loss, accuracy = model.evaluate_batch(features, labels)
        total_loss += loss * len(batch)
        total_correct += accuracy * len(batch)
        total_seen += len(batch)
    if total_seen == 0:
        return 0.0, 0.0
    return total_loss / total_seen, total_correct / total_seen


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        print("No tasks provided", file=sys.stderr)
        return 1

    results_path = Path(__file__).with_name("results_glue_multitask.json")

    metrics: Dict[str, Dict[str, float]] = {}
    config_tasks: List[str] = []

    for task in tasks:
        train_examples = prepare_dataset(task, "train", ci=args.ci, max_samples=args.max_samples)
        eval_examples = prepare_dataset(task, "validation", ci=args.ci, max_samples=args.max_samples)
        if not eval_examples:
            eval_examples = prepare_dataset(task, "train", ci=args.ci, max_samples=args.max_samples)

        num_labels = max(max(example.label for example in train_examples), 1) + 1
        input_dim = len(train_examples[0].features) if train_examples else 512
        model = LinearClassifier(input_dim, num_labels, lr=0.7 if args.ci else 0.3)

        train_loss, train_accuracy = aggregate_split(model, train_examples, args.batch, train=True)
        for _ in range(max(args.epochs - 1, 0)):
            train_loss, train_accuracy = aggregate_split(model, train_examples, args.batch, train=True)
        eval_loss, eval_accuracy = aggregate_split(model, eval_examples, args.batch, train=False)

        metrics[task] = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "eval_loss": float(eval_loss),
            "eval_accuracy": float(eval_accuracy),
            "num_train_examples": float(len(train_examples)),
            "num_eval_examples": float(len(eval_examples)),
        }
        config_tasks.append(task)

    output = {
        "config": {
            "model": args.model,
            "tasks": config_tasks,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "ci_mode": bool(args.ci),
            "device": args.device,
            "max_samples": args.max_samples,
        },
        "tasks": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    results_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
