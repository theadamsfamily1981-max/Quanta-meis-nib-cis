"""
Model compression for TFAN: Quantization and Pruning.
Enables edge deployment with reduced model size and faster inference.
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class QuantizationConfig:
    """Configuration for model quantization."""

    @staticmethod
    def get_default_qconfig(backend: str = "fbgemm"):
        """
        Get default quantization configuration.

        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

        Returns:
            Quantization config
        """
        if backend == "fbgemm":
            return quant.get_default_qconfig('fbgemm')
        elif backend == "qnnpack":
            return quant.get_default_qconfig('qnnpack')
        else:
            return quant.get_default_qconfig('fbgemm')


class ModelQuantizer:
    """
    Model quantization for INT8 and FP16 precision.

    Supports:
    - Post-training quantization (PTQ)
    - Quantization-aware training (QAT)
    - Dynamic quantization
    """

    def __init__(self, model: nn.Module, backend: str = "fbgemm"):
        """
        Args:
            model: Model to quantize
            backend: Quantization backend
        """
        self.model = model
        self.backend = backend
        torch.backends.quantized.engine = backend

    def dynamic_quantize(self, dtype=torch.qint8) -> nn.Module:
        """
        Apply dynamic quantization (weights only).

        Args:
            dtype: Quantization dtype (torch.qint8 or torch.float16)

        Returns:
            Quantized model
        """
        quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
            dtype=dtype
        )

        return quantized_model

    def static_quantize(self,
                       calibration_loader,
                       qconfig=None) -> nn.Module:
        """
        Apply static (post-training) quantization.

        Args:
            calibration_loader: DataLoader for calibration
            qconfig: Quantization configuration

        Returns:
            Quantized model
        """
        if qconfig is None:
            qconfig = QuantizationConfig.get_default_qconfig(self.backend)

        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = qconfig

        # Fuse modules if possible (Conv+BN+ReLU, etc.)
        fused_model = self._fuse_modules()

        # Prepare for calibration
        prepared_model = quant.prepare(fused_model)

        # Calibration
        print("Calibrating quantization...")
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calibration_loader):
                prepared_model(inputs)

                if batch_idx >= 100:  # Limit calibration batches
                    break

        # Convert to quantized model
        quantized_model = quant.convert(prepared_model)

        return quantized_model

    def qat_prepare(self, qconfig=None) -> nn.Module:
        """
        Prepare model for quantization-aware training.

        Args:
            qconfig: Quantization configuration

        Returns:
            Prepared model for QAT
        """
        if qconfig is None:
            qconfig = quant.get_default_qat_qconfig(self.backend)

        self.model.train()
        self.model.qconfig = qconfig

        # Fuse modules
        fused_model = self._fuse_modules()

        # Prepare for QAT
        prepared_model = quant.prepare_qat(fused_model)

        return prepared_model

    def qat_convert(self, trained_model: nn.Module) -> nn.Module:
        """
        Convert QAT model to quantized model.

        Args:
            trained_model: Model trained with QAT

        Returns:
            Quantized model
        """
        trained_model.eval()
        quantized_model = quant.convert(trained_model)

        return quantized_model

    def _fuse_modules(self) -> nn.Module:
        """Fuse consecutive modules for efficiency."""
        # This is model-specific
        # Example: fuse Conv-BN-ReLU
        try:
            fused_model = torch.quantization.fuse_modules(
                self.model,
                [['conv', 'bn', 'relu']]  # Example fusion
            )
            return fused_model
        except:
            # If fusion fails, return original model
            return self.model

    def to_fp16(self) -> nn.Module:
        """Convert model to FP16 (half precision)."""
        fp16_model = self.model.half()
        return fp16_model

    def measure_model_size(self, quantized_model: nn.Module) -> Dict[str, float]:
        """
        Measure model size before and after quantization.

        Returns:
            Dictionary with size metrics
        """
        # Original model size
        orig_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        orig_size_mb = orig_size / (1024 ** 2)

        # Quantized model size
        quant_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        quant_size_mb = quant_size / (1024 ** 2)

        compression_ratio = orig_size / quant_size if quant_size > 0 else 0

        return {
            "original_size_mb": orig_size_mb,
            "quantized_size_mb": quant_size_mb,
            "compression_ratio": compression_ratio
        }


class StructuredPruning:
    """
    Structured pruning for neural networks.

    Removes entire channels/neurons to reduce model size and latency.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to prune
        """
        self.model = model
        self.pruning_masks = {}

    def compute_importance(self,
                          layer: nn.Module,
                          method: str = "l1") -> torch.Tensor:
        """
        Compute importance scores for layer weights.

        Args:
            layer: Layer to analyze
            method: Importance metric ('l1', 'l2', 'gradient')

        Returns:
            Importance scores
        """
        if not hasattr(layer, 'weight'):
            return torch.tensor([])

        weight = layer.weight.data

        if method == "l1":
            # L1 norm of filters/neurons
            importance = torch.sum(torch.abs(weight), dim=tuple(range(1, weight.ndim)))
        elif method == "l2":
            # L2 norm of filters/neurons
            importance = torch.norm(weight, dim=tuple(range(1, weight.ndim)))
        elif method == "gradient":
            # Gradient-based importance (requires gradients)
            if layer.weight.grad is not None:
                importance = torch.abs(layer.weight.grad).sum(dim=tuple(range(1, weight.ndim)))
            else:
                importance = torch.sum(torch.abs(weight), dim=tuple(range(1, weight.ndim)))
        else:
            raise ValueError(f"Unknown method: {method}")

        return importance

    def prune_layer(self,
                    layer: nn.Module,
                    pruning_ratio: float,
                    method: str = "l1") -> Tuple[nn.Module, torch.Tensor]:
        """
        Prune a single layer.

        Args:
            layer: Layer to prune
            pruning_ratio: Ratio of channels/neurons to prune (0-1)
            method: Importance metric

        Returns:
            (pruned_layer, pruning_mask)
        """
        if not hasattr(layer, 'weight'):
            return layer, torch.tensor([])

        # Compute importance
        importance = self.compute_importance(layer, method)

        # Determine pruning threshold
        num_channels = importance.shape[0]
        num_to_prune = int(num_channels * pruning_ratio)

        if num_to_prune == 0:
            return layer, torch.ones_like(importance, dtype=torch.bool)

        # Get indices to keep
        _, sorted_indices = torch.sort(importance, descending=True)
        keep_indices = sorted_indices[:num_channels - num_to_prune]

        # Create pruning mask
        mask = torch.zeros(num_channels, dtype=torch.bool, device=importance.device)
        mask[keep_indices] = True

        # Apply pruning (create new layer with reduced channels)
        if isinstance(layer, nn.Linear):
            pruned_layer = self._prune_linear(layer, mask)
        elif isinstance(layer, nn.Conv2d):
            pruned_layer = self._prune_conv2d(layer, mask)
        else:
            # Unsupported layer type
            return layer, mask

        return pruned_layer, mask

    def _prune_linear(self, layer: nn.Linear, mask: torch.Tensor) -> nn.Linear:
        """Prune linear layer."""
        keep_indices = torch.where(mask)[0]

        # Create new layer
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=len(keep_indices),
            bias=layer.bias is not None
        )

        # Copy weights
        new_layer.weight.data = layer.weight.data[keep_indices]

        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices]

        return new_layer

    def _prune_conv2d(self, layer: nn.Conv2d, mask: torch.Tensor) -> nn.Conv2d:
        """Prune Conv2d layer."""
        keep_indices = torch.where(mask)[0]

        # Create new layer
        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=len(keep_indices),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias is not None
        )

        # Copy weights
        new_layer.weight.data = layer.weight.data[keep_indices]

        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices]

        return new_layer

    def global_pruning(self,
                      pruning_ratio: float,
                      method: str = "l1",
                      layer_types: Optional[List[type]] = None) -> nn.Module:
        """
        Apply global pruning across all layers.

        Args:
            pruning_ratio: Global pruning ratio
            method: Importance metric
            layer_types: List of layer types to prune

        Returns:
            Pruned model
        """
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]

        print(f"Applying global pruning with ratio {pruning_ratio}")

        # Collect all importances
        all_importances = []
        layer_infos = []

        for name, module in self.model.named_modules():
            if any(isinstance(module, lt) for lt in layer_types):
                importance = self.compute_importance(module, method)
                if len(importance) > 0:
                    all_importances.append(importance)
                    layer_infos.append((name, module))

        if len(all_importances) == 0:
            print("No prunable layers found")
            return self.model

        # Concatenate all importances
        global_importance = torch.cat(all_importances)

        # Global threshold
        num_total = len(global_importance)
        num_to_prune = int(num_total * pruning_ratio)

        threshold = torch.kthvalue(global_importance, num_to_prune + 1).values

        print(f"Global threshold: {threshold:.6f}")
        print(f"Pruning {num_to_prune}/{num_total} parameters")

        # Apply pruning to each layer
        for name, module in layer_infos:
            importance = self.compute_importance(module, method)
            mask = importance > threshold

            # Store mask
            self.pruning_masks[name] = mask

            print(f"  {name}: keeping {mask.sum()}/{len(mask)} channels")

        return self.model

    def iterative_pruning(self,
                         target_ratio: float,
                         num_iterations: int = 5,
                         fine_tune_fn: Optional[Callable] = None) -> nn.Module:
        """
        Iterative pruning with fine-tuning.

        Args:
            target_ratio: Target pruning ratio
            num_iterations: Number of pruning iterations
            fine_tune_fn: Function to fine-tune after each pruning step

        Returns:
            Pruned model
        """
        ratio_per_iter = 1 - (1 - target_ratio) ** (1 / num_iterations)

        print(f"Iterative pruning: {num_iterations} iterations, "
              f"{ratio_per_iter:.3f} ratio per iteration")

        for iter_idx in range(num_iterations):
            print(f"\nIteration {iter_idx + 1}/{num_iterations}")

            # Prune
            self.global_pruning(ratio_per_iter)

            # Fine-tune if function provided
            if fine_tune_fn is not None:
                print("Fine-tuning...")
                fine_tune_fn(self.model)

        return self.model


class CompressionPipeline:
    """
    Complete compression pipeline combining quantization and pruning.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to compress
        """
        self.model = model
        self.quantizer = ModelQuantizer(model)
        self.pruner = StructuredPruning(model)

    def compress(self,
                pruning_ratio: float = 0.5,
                quantization_method: str = "dynamic",
                calibration_loader=None) -> Tuple[nn.Module, Dict]:
        """
        Apply full compression pipeline.

        Args:
            pruning_ratio: Ratio of parameters to prune
            quantization_method: 'dynamic', 'static', or 'qat'
            calibration_loader: DataLoader for calibration (static quant)

        Returns:
            (compressed_model, compression_stats)
        """
        print("=== Model Compression Pipeline ===\n")

        # Original model size
        orig_params = sum(p.numel() for p in self.model.parameters())
        print(f"Original parameters: {orig_params:,}")

        # Step 1: Pruning
        print("\n1. Structured Pruning")
        pruned_model = self.pruner.global_pruning(pruning_ratio)

        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Pruning ratio achieved: {1 - pruned_params/orig_params:.2%}")

        # Step 2: Quantization
        print("\n2. Quantization")

        if quantization_method == "dynamic":
            compressed_model = self.quantizer.dynamic_quantize()
        elif quantization_method == "static":
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            compressed_model = self.quantizer.static_quantize(calibration_loader)
        elif quantization_method == "fp16":
            compressed_model = self.quantizer.to_fp16()
        else:
            compressed_model = pruned_model

        # Compute statistics
        size_stats = self.quantizer.measure_model_size(compressed_model)

        stats = {
            "original_params": orig_params,
            "pruned_params": pruned_params,
            "pruning_ratio": 1 - pruned_params / orig_params,
            "quantization_method": quantization_method,
            **size_stats
        }

        print("\n=== Compression Complete ===")
        print(f"Final parameters: {pruned_params:,}")
        print(f"Model size: {size_stats['quantized_size_mb']:.2f} MB")
        print(f"Overall compression: {size_stats['compression_ratio']:.2f}×")

        return compressed_model, stats


if __name__ == "__main__":
    # Demo model compression
    print("=== Model Compression Demo ===\n")

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleModel()

    # Quantization demo
    print("1. Quantization")
    quantizer = ModelQuantizer(model)
    quant_model = quantizer.dynamic_quantize()

    size_stats = quantizer.measure_model_size(quant_model)
    print(f"  Original size: {size_stats['original_size_mb']:.2f} MB")
    print(f"  Quantized size: {size_stats['quantized_size_mb']:.2f} MB")
    print(f"  Compression: {size_stats['compression_ratio']:.2f}×")

    # Pruning demo
    print("\n2. Pruning")
    pruner = StructuredPruning(model)

    # Prune FC1 layer
    pruned_fc1, mask = pruner.prune_layer(model.fc1, pruning_ratio=0.5)
    print(f"  FC1: {model.fc1.out_features} → {pruned_fc1.out_features} neurons")
    print(f"  Pruning ratio: {1 - pruned_fc1.out_features/model.fc1.out_features:.1%}")

    print("\nCompression demo complete!")
