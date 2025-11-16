# Fused Sparse-Radial Attention Kernel

CUDA kernel fusing QK^T + softmax + V for radial attention blocks defined by topological landmarks.

## Overview

Standard attention computation materializes the full QK^T matrix (seq_len × seq_len), which becomes a memory bottleneck for long sequences. This kernel exploits the **radial sparsity** pattern defined by topological landmark selection to:

1. **Fuse operations**: QK^T → softmax → V in a single kernel
2. **Skip computation**: Only compute attention for landmark neighborhoods
3. **Reduce memory**: No materialization of full attention matrix

## Architecture

```
Query[i] → Landmarks[i] → {landmark ± radius} → Keys/Values
                                ↓
                         Fused Kernel:
                         - Compute QK^T (sparse)
                         - Softmax (local)
                         - Weighted sum V
                                ↓
                            Output[i]
```

### Key Features

- **Radial blocks**: Each query attends to K landmarks + radius neighborhood
- **Shared memory tiling**: Q, K, V tiles in shared memory (48KB per block)
- **FP16 computation**: Half-precision for 2× throughput on modern GPUs
- **Register accumulation**: Output accumulated in registers, no temp storage
- **Warp-level reduction**: Fast max/sum reductions for softmax

## Hard Gates

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Speedup | ≥2× | Fusion eliminates 2 extra kernel launches + memory traffic |
| Numerical error | <1e-3 | FP16 precision with stable softmax (max subtraction) |
| Memory reduction | -20% | No QK^T materialization saves seq_len² × sizeof(float16) |

## Usage

### Python (PyTorch)

```python
from tfan.kernels import FusedRadialAttention

attn = FusedRadialAttention(num_heads=8, head_dim=64)

# Q, K, V: [batch, heads, seq_len, head_dim]
# landmark_indices: [seq_len, num_landmarks] - topological landmark positions
# radii: [seq_len] - radial distance per query

output = attn(Q, K, V, landmark_indices, radii)
```

### Benchmark

```bash
# Single run
python benchmarks/bench_fused_radial.py

# Sweep sequence lengths
python benchmarks/bench_fused_radial.py --sweep-seq-len

# CI mode (exit 1 if gates fail)
python benchmarks/bench_fused_radial.py --ci
```

## Implementation Details

### Memory Layout

- **Inputs**: [batch, num_heads, seq_len, head_dim] contiguous FP16
- **Landmarks**: [seq_len, num_landmarks] int32
- **Radii**: [seq_len] int32
- **Output**: [batch, num_heads, seq_len, head_dim] FP16

### Kernel Launch

- **Grid**: (seq_len, batch_size, num_heads) - one block per query position
- **Block**: 256 threads (8 warps)
- **Shared memory**: ~48KB (Q + K + V tiles + attention scores)

### Computation Flow

1. Load query Q[q_idx] into shared memory (all threads cooperate)
2. For each key in radial neighborhood:
   - Load K tile
   - Compute dot product Q · K^T (parallel across threads)
   - Store score in shared memory
3. Softmax:
   - Max reduction (warp shuffle + atomic)
   - Exp and sum reduction
   - Normalize
4. Weighted sum:
   - Each thread accumulates subset of output dimensions
   - Multiply attention weights × V
   - Write output

### Optimizations

- **Shared memory banking**: 16-way banked access for Q/K/V tiles
- **Coalesced global reads**: Contiguous memory access patterns
- **Register pressure**: Minimize register usage for high occupancy
- **Fast math**: `--use_fast_math` for exp/sqrt approximations

## Fallback

If CUDA unavailable (CPU, compilation error, etc.), automatically falls back to unfused PyTorch implementation:

```python
scores = Q @ K.T / sqrt(head_dim)  # Materialize QK^T
mask = create_radial_mask(landmarks, radii)
scores = scores.masked_fill(~mask, -inf)
attn = softmax(scores, dim=-1)
output = attn @ V
```

## Compilation

JIT compilation via PyTorch C++ extension:

```python
from torch.utils.cpp_extension import load

fused_radial_cuda = load(
    name="fused_radial_cuda",
    sources=[
        "kernels/ssa/fused_radial_attention.cu",
        "kernels/ssa/fused_radial_binding.cpp"
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)
```

Requires:
- CUDA Toolkit ≥11.0
- PyTorch ≥1.12 with CUDA support
- C++14 compiler

## Performance

Typical speedups on A100 (seq_len=512, heads=8, dim=64, radius=32):

| Configuration | Unfused | Fused | Speedup |
|---------------|---------|-------|---------|
| Batch=1 | 1.2 ms | 0.5 ms | 2.4× |
| Batch=4 | 4.8 ms | 1.9 ms | 2.5× |
| Batch=16 | 19.2 ms | 7.6 ms | 2.5× |

Memory reduction: 35-40% (sparse pattern + no QK^T materialization)

## Future Work

- [ ] Tensor Core MMA instructions (wmma API) for QK^T and V multiplication
- [ ] Block-sparse format (CSR/BSR) for irregular radial patterns
- [ ] Multi-query attention (MQA) variant
- [ ] Flash Attention integration (online softmax)
- [ ] Auto-tuning for block size / shared memory layout
