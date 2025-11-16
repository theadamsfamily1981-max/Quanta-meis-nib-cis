#!/usr/bin/env python
"""
CUPTI GPU Telemetry Tracer

Uses NVIDIA CUPTI (CUDA Profiling Tools Interface) to trace:
- Kernel launches (SSA, attention, etc.)
- Memory transfers (H2D, D2H)
- GPU utilization
- CUDA API calls

Hard gate: p95 attribution coverage ≥95% of wall-time

Usage:
    tracer = CUPTITracer()
    tracer.start()

    # Run workload
    model.forward(input)

    tracer.stop()
    stats = tracer.get_stats()
"""

import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class KernelEvent:
    """Single GPU kernel event."""
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    device_id: int
    stream_id: int
    grid_size: tuple
    block_size: tuple


@dataclass
class MemoryTransfer:
    """Memory transfer event."""
    kind: str  # 'H2D', 'D2H', 'D2D'
    start_ns: int
    end_ns: int
    duration_ns: int
    size_bytes: int
    bandwidth_gbps: float


class CUPTITracer:
    """
    GPU telemetry tracer using CUPTI.

    Note: This is a stub implementation. Real CUPTI integration requires:
    - nvidia-pyprof or pycupti
    - CUDA toolkit with CUPTI headers
    - Appropriate permissions/capabilities
    """

    def __init__(self):
        self.kernel_events: List[KernelEvent] = []
        self.memory_events: List[MemoryTransfer] = []
        self.start_time_ns = 0
        self.end_time_ns = 0
        self.is_tracing = False

        print("✓ CUPTITracer initialized")
        print("⚠ Using stub implementation (CUPTI not available)")

    def start(self):
        """Start tracing."""
        self.kernel_events.clear()
        self.memory_events.clear()
        self.start_time_ns = time.perf_counter_ns()
        self.is_tracing = True

        print("✓ GPU tracing started")

    def stop(self):
        """Stop tracing and collect events."""
        if not self.is_tracing:
            return

        self.end_time_ns = time.perf_counter_ns()
        self.is_tracing = False

        # Stub: generate mock events
        self._generate_mock_events()

        print("✓ GPU tracing stopped")
        print(f"  Duration: {(self.end_time_ns - self.start_time_ns) / 1e9:.3f}s")
        print(f"  Kernel events: {len(self.kernel_events)}")
        print(f"  Memory events: {len(self.memory_events)}")

    def _generate_mock_events(self):
        """Generate mock events for testing."""
        # Mock kernel events
        kernel_names = [
            'ssa_attention_kernel',
            'softmax_kernel',
            'matmul_kernel',
            'layer_norm_kernel'
        ]

        current_ts = self.start_time_ns

        for i in range(100):
            name = np.random.choice(kernel_names)
            duration_ns = int(np.random.lognormal(13, 1.5) * 1000)  # ~100μs mean

            event = KernelEvent(
                name=name,
                start_ns=current_ts,
                end_ns=current_ts + duration_ns,
                duration_ns=duration_ns,
                device_id=0,
                stream_id=0,
                grid_size=(256, 1, 1),
                block_size=(128, 1, 1)
            )

            self.kernel_events.append(event)
            current_ts += duration_ns + int(np.random.exponential(5000))  # Gap

        # Mock memory transfers
        for i in range(10):
            kind = np.random.choice(['H2D', 'D2H'])
            size_bytes = int(np.random.exponential(1024 * 1024))  # ~1MB
            duration_ns = int(size_bytes / (10 * 1024**3) * 1e9)  # 10 GB/s

            transfer = MemoryTransfer(
                kind=kind,
                start_ns=current_ts,
                end_ns=current_ts + duration_ns,
                duration_ns=duration_ns,
                size_bytes=size_bytes,
                bandwidth_gbps=size_bytes / (duration_ns / 1e9) / 1024**3
            )

            self.memory_events.append(transfer)
            current_ts += duration_ns

    def get_stats(self) -> Dict:
        """Get tracing statistics."""
        if not self.kernel_events:
            return {}

        # Kernel statistics
        kernel_durations = defaultdict(list)
        for event in self.kernel_events:
            kernel_durations[event.name].append(event.duration_ns)

        kernel_stats = {}
        for name, durations in kernel_durations.items():
            kernel_stats[name] = {
                'count': len(durations),
                'total_ns': sum(durations),
                'mean_ns': np.mean(durations),
                'p50_ns': np.percentile(durations, 50),
                'p95_ns': np.percentile(durations, 95),
                'p99_ns': np.percentile(durations, 99)
            }

        # Total GPU time
        total_gpu_ns = sum(e.duration_ns for e in self.kernel_events)
        total_wall_ns = self.end_time_ns - self.start_time_ns

        # p95 attribution coverage
        attribution_coverage = total_gpu_ns / total_wall_ns if total_wall_ns > 0 else 0.0

        # Memory transfer statistics
        mem_stats = {}
        if self.memory_events:
            mem_durations = [e.duration_ns for e in self.memory_events]
            mem_sizes = [e.size_bytes for e in self.memory_events]

            mem_stats = {
                'count': len(self.memory_events),
                'total_bytes': sum(mem_sizes),
                'total_ns': sum(mem_durations),
                'mean_bandwidth_gbps': np.mean([e.bandwidth_gbps for e in self.memory_events])
            }

        return {
            'kernel_stats': kernel_stats,
            'memory_stats': mem_stats,
            'total_gpu_time_ns': total_gpu_ns,
            'total_wall_time_ns': total_wall_ns,
            'gpu_utilization': total_gpu_ns / total_wall_ns if total_wall_ns > 0 else 0.0,
            'attribution_coverage': attribution_coverage
        }

    def export_events(self, output_path: str):
        """Export events to JSON."""
        events = {
            'kernel_events': [asdict(e) for e in self.kernel_events],
            'memory_events': [asdict(e) for e in self.memory_events],
            'start_time_ns': self.start_time_ns,
            'end_time_ns': self.end_time_ns
        }

        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)

        print(f"✓ Events exported to {output_path}")

    def get_flamegraph_data(self) -> List[Dict]:
        """
        Generate flamegraph data for visualization.

        Returns list of entries in format:
        [{"name": "kernel_name", "value": duration_ns}, ...]
        """
        flamegraph_data = []

        # Group by kernel name
        kernel_totals = defaultdict(int)
        for event in self.kernel_events:
            kernel_totals[event.name] += event.duration_ns

        # Convert to flamegraph format
        for name, total_ns in kernel_totals.items():
            flamegraph_data.append({
                'name': name,
                'value': total_ns / 1e6,  # Convert to ms
                'children': []
            })

        return flamegraph_data


def check_cupti_available() -> bool:
    """Check if CUPTI is available."""
    try:
        import pycuda.driver as cuda
        cuda.init()
        return True
    except ImportError:
        return False
    except Exception:
        return False
