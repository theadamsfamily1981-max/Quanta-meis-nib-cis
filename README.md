# Quanta-meis-nib-cis
Research for quanta meis nib cis code

## Benchmarking selective self-attention

Use `benchmark_selective_attention.py` to measure the latency and peak memory
usage of the TensorFlow Attention Network (TFAN) selective self-attention
module.

```bash
python benchmark_selective_attention.py
```

The script prints a JSON blob that includes the sequence length (`N`), the
landmark keep ratio (`keep`), latency in milliseconds (`lat_ms`), and the peak
GPU memory usage in megabytes (`mem_peak_mb`). When CUDA is unavailable, the
script falls back to CPU execution and reports zero for the peak memory metric.
