#!/usr/bin/env python3
"""
Model Export and Optimization Pipeline
Prepares trained TFAN models for production deployment.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.model import TFANModel
from tfan.compression import CompressionPipeline, ModelQuantizer, StructuredPruning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model export and optimization for deployment."""

    def __init__(
        self,
        model: nn.Module,
        model_config: Dict[str, Any],
        output_dir: Path
    ):
        self.model = model
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_pytorch(
        self,
        version: str = "v1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export as PyTorch checkpoint."""
        logger.info("Exporting PyTorch checkpoint...")

        checkpoint = {
            'version': version,
            'config': self.model_config,
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {}
        }

        output_path = self.output_dir / "tfan_model.pt"
        torch.save(checkpoint, output_path)

        logger.info(f"Saved PyTorch checkpoint to {output_path}")
        return output_path

    def export_torchscript(
        self,
        example_input: torch.Tensor,
        optimize: bool = True
    ) -> Path:
        """Export as TorchScript for optimized inference."""
        logger.info("Exporting TorchScript...")

        self.model.eval()

        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)

            if optimize:
                # Apply TorchScript optimizations
                traced_model = torch.jit.optimize_for_inference(traced_model)

        output_path = self.output_dir / "tfan_model_traced.pt"
        traced_model.save(str(output_path))

        logger.info(f"Saved TorchScript model to {output_path}")
        return output_path

    def export_onnx(
        self,
        example_input: torch.Tensor,
        opset_version: int = 14
    ) -> Path:
        """Export as ONNX for cross-platform deployment."""
        logger.info("Exporting ONNX...")

        self.model.eval()

        output_path = self.output_dir / "tfan_model.onnx"

        torch.onnx.export(
            self.model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"Saved ONNX model to {output_path}")
        return output_path

    def apply_compression(
        self,
        pruning_ratio: float = 0.4,
        quantization_method: str = "dynamic"
    ) -> nn.Module:
        """Apply model compression."""
        logger.info(f"Applying compression (pruning={pruning_ratio}, quant={quantization_method})...")

        pipeline = CompressionPipeline(self.model)

        compressed_model, stats = pipeline.compress(
            pruning_ratio=pruning_ratio,
            quantization_method=quantization_method,
            calibration_loader=None  # TODO: Add calibration data if using static quantization
        )

        logger.info(f"Compression complete: {stats}")

        # Save compressed model
        compressed_checkpoint = {
            'version': 'v1.0.0-compressed',
            'config': self.model_config,
            'model_state_dict': compressed_model.state_dict() if hasattr(compressed_model, 'state_dict') else {},
            'compression_stats': stats,
            'metadata': {
                'compressed': True,
                'pruning_ratio': pruning_ratio,
                'quantization_method': quantization_method
            }
        }

        compressed_path = self.output_dir / "tfan_model_compressed.pt"

        # Handle quantized models differently
        if quantization_method == "dynamic":
            # For quantized models, save the entire model
            torch.save({'model': compressed_model, **compressed_checkpoint}, compressed_path)
        else:
            torch.save(compressed_checkpoint, compressed_path)

        logger.info(f"Saved compressed model to {compressed_path}")

        return compressed_model

    def benchmark_model(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        logger.info(f"Benchmarking model ({num_iterations} iterations)...")

        model.eval()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input.to(device))

        # Benchmark
        import time
        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(example_input.to(device))
                if device == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.time() - start) * 1000)  # ms

        # Calculate stats
        latencies = np.array(latencies)
        stats = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_qps': float(1000.0 / np.mean(latencies))
        }

        logger.info(f"Benchmark results: {stats}")
        return stats

    def save_config(self):
        """Save model configuration."""
        config_path = self.output_dir / "model_config.json"

        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)

        logger.info(f"Saved model config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export and optimize TFAN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default="./deploy/models", help="Output directory")
    parser.add_argument("--format", type=str, choices=["pytorch", "torchscript", "onnx", "all"], default="all")
    parser.add_argument("--compress", action="store_true", help="Apply compression")
    parser.add_argument("--pruning-ratio", type=float, default=0.4, help="Pruning ratio")
    parser.add_argument("--quantization", type=str, choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")

    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Extract config
    config = checkpoint.get('config', {
        'input_dim': 128,
        'hidden_dim': 256,
        'output_dim': 10,
        'num_heads': 8,
        'num_layers': 6,
        'seq_len': 64,
        'dropout': 0.1
    })

    # Initialize model
    logger.info("Initializing model...")
    model = TFANModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create exporter
    exporter = ModelExporter(
        model=model,
        model_config=config,
        output_dir=Path(args.output_dir)
    )

    # Save config
    exporter.save_config()

    # Create example input for export
    batch_size = 1
    seq_len = config.get('seq_len', 64)
    input_dim = config.get('input_dim', 128)
    example_input = torch.randn(batch_size, seq_len, input_dim)

    # Export in requested formats
    if args.format in ["pytorch", "all"]:
        exporter.export_pytorch(
            version=checkpoint.get('version', 'v1.0.0'),
            metadata=checkpoint.get('metadata', {})
        )

    if args.format in ["torchscript", "all"]:
        exporter.export_torchscript(example_input, optimize=True)

    if args.format in ["onnx", "all"]:
        exporter.export_onnx(example_input, opset_version=14)

    # Apply compression
    if args.compress:
        compressed_model = exporter.apply_compression(
            pruning_ratio=args.pruning_ratio,
            quantization_method=args.quantization
        )

        if args.benchmark:
            logger.info("\nBenchmarking compressed model:")
            exporter.benchmark_model(compressed_model, example_input, num_iterations=100)

    # Benchmark original model
    if args.benchmark:
        logger.info("\nBenchmarking original model:")
        exporter.benchmark_model(model, example_input, num_iterations=100)

    logger.info("\n✅ Export complete!")
    logger.info(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
