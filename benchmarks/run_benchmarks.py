#!/usr/bin/env python3
"""
GLUE & SuperGLUE Benchmark Runner
Complete evaluation suite for NLP models.

Usage:
    python benchmarks/run_benchmarks.py --model tfan --tasks cola,sst-2,mrpc
    python benchmarks/run_benchmarks.py --model tfan --suite glue --all
    python benchmarks/run_benchmarks.py --model tfan --suite superglue --all
"""

import argparse
import logging
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from glue import load_glue_task, compute_metrics as glue_metrics, GLUE_TASKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs GLUE/SuperGLUE benchmarks."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def evaluate_task(
        self,
        task_name: str,
        data_dir: str,
        split: str = "dev",
        max_length: int = 128
    ) -> dict:
        """
        Evaluate model on a single task.

        Returns:
            Dictionary with metrics and predictions
        """
        logger.info(f"Evaluating {task_name} ({split} split)...")

        # Load dataset
        dataset = load_glue_task(
            task_name=task_name,
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            split=split,
            max_length=max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Run inference
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                # Move to device
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != 'label'
                }

                # Forward pass
                outputs = self.model(**inputs)

                # Get predictions
                if task_name == "sts-b":
                    # Regression
                    preds = outputs.logits.squeeze(-1).cpu().numpy()
                else:
                    # Classification
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds.tolist())

                if 'label' in batch:
                    labels = batch['label'].cpu().numpy()
                    all_labels.extend(labels.tolist())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = glue_metrics(task_name, all_preds, all_labels)

        logger.info(f"{task_name} metrics: {metrics}")

        return {
            'task': task_name,
            'metrics': metrics,
            'num_examples': len(all_preds)
        }

    def run_glue(self, data_root: str, tasks: list = None) -> dict:
        """Run full GLUE benchmark."""
        if tasks is None:
            tasks = list(GLUE_TASKS.keys())

        logger.info(f"Running GLUE benchmark on {len(tasks)} tasks...")

        results = {}
        for task in tasks:
            task_dir = Path(data_root) / task.upper()
            if not task_dir.exists():
                logger.warning(f"Skipping {task}: data not found at {task_dir}")
                continue

            result = self.evaluate_task(
                task_name=task,
                data_dir=str(task_dir),
                split="dev"
            )
            results[task] = result

        # Compute average score
        # GLUE score is average of individual task metrics
        glue_score = self._compute_glue_score(results)
        results['glue_score'] = glue_score

        logger.info(f"\n{'='*80}")
        logger.info(f"GLUE SCORE: {glue_score:.2f}")
        logger.info(f"{'='*80}")

        return results

    def _compute_glue_score(self, results: dict) -> float:
        """Compute official GLUE score."""
        task_scores = []

        for task, result in results.items():
            if task == 'glue_score':
                continue

            metrics = result['metrics']

            # Extract primary metric for each task
            if task == 'cola':
                score = metrics['mcc'] * 100
            elif task == 'sts-b':
                score = (metrics['pearson'] + metrics['spearman']) / 2 * 100
            elif task in ['mrpc', 'qqp']:
                score = (metrics['acc'] + metrics['f1']) / 2 * 100
            else:
                score = metrics['acc'] * 100

            task_scores.append(score)

        return np.mean(task_scores)

    def save_results(self, results: dict, output_file: str):
        """Save results to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GLUE/SuperGLUE benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--suite", type=str, choices=["glue", "superglue"], default="glue")
    parser.add_argument("--tasks", type=str, help="Comma-separated task names")
    parser.add_argument("--all", action="store_true", help="Run all tasks in suite")
    parser.add_argument("--data-root", type=str, default="./data/glue")
    parser.add_argument("--output", type=str, default="./results/benchmark_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model}...")
    # TODO: Implement model loading
    # from tfan.model import TFANModel
    # model = TFANModel.from_pretrained(args.model)
    # tokenizer = ...

    # For now, use placeholder
    model = None
    tokenizer = None

    # Parse tasks
    if args.all:
        tasks = None  # Use all tasks
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        raise ValueError("Must specify --all or --tasks")

    # Run benchmark
    runner = BenchmarkRunner(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    start_time = time.time()

    if args.suite == "glue":
        results = runner.run_glue(
            data_root=args.data_root,
            tasks=tasks
        )
    else:
        raise NotImplementedError("SuperGLUE coming soon!")

    elapsed = time.time() - start_time

    # Add metadata
    results['metadata'] = {
        'model': args.model,
        'suite': args.suite,
        'elapsed_time': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save results
    runner.save_results(results, args.output)

    logger.info(f"\nBenchmark complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
