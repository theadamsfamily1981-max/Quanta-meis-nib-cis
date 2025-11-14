"""
Production monitoring dashboard setup for TFAN.
Provides real-time metrics tracking, alerting, and visualization.
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading


@dataclass
class MetricSnapshot:
    """Single metric snapshot at a point in time."""
    timestamp: float
    name: str
    value: float
    unit: str
    tags: Dict[str, str]


class MetricsCollector:
    """
    Collects and aggregates system metrics for monitoring.
    Thread-safe for concurrent access.
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent samples to keep per metric
        """
        self.window_size = window_size
        self.metrics = {}
        self.lock = threading.Lock()

    def record(self, name: str, value: float, unit: str = "",
               tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        timestamp = time.time()
        tags = tags or {}

        snapshot = MetricSnapshot(
            timestamp=timestamp,
            name=name,
            value=value,
            unit=unit,
            tags=tags
        )

        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)

            self.metrics[name].append(snapshot)

    def get_recent(self, name: str, n: int = 100) -> List[MetricSnapshot]:
        """Get N most recent samples for a metric."""
        with self.lock:
            if name not in self.metrics:
                return []

            samples = list(self.metrics[name])
            return samples[-n:]

    def get_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict:
        """
        Get statistics for a metric over a time window.

        Args:
            name: Metric name
            window_seconds: Time window in seconds (None = all samples)

        Returns:
            Dict with min, max, mean, std, p50, p95, p99
        """
        with self.lock:
            if name not in self.metrics:
                return {}

            samples = list(self.metrics[name])

        # Filter by time window
        if window_seconds is not None:
            cutoff_time = time.time() - window_seconds
            samples = [s for s in samples if s.timestamp >= cutoff_time]

        if not samples:
            return {}

        values = [s.value for s in samples]
        values_sorted = sorted(values)

        n = len(values)
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        import numpy as np
        stats = {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "p50": values_sorted[p50_idx],
            "p95": values_sorted[p95_idx],
            "p99": values_sorted[p99_idx]
        }

        return stats

    def export_prometheus(self, output_path: str):
        """Export metrics in Prometheus format."""
        with self.lock:
            lines = []

            for name, samples in self.metrics.items():
                if not samples:
                    continue

                latest = samples[-1]

                # Format: metric_name{tag1="value1"} value timestamp
                tags_str = ",".join([f'{k}="{v}"' for k, v in latest.tags.items()])
                if tags_str:
                    metric_line = f'{name}{{{tags_str}}} {latest.value} {int(latest.timestamp * 1000)}'
                else:
                    metric_line = f'{name} {latest.value} {int(latest.timestamp * 1000)}'

                lines.append(metric_line)

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def export_json(self, output_path: str, window_seconds: Optional[float] = None):
        """Export metrics as JSON."""
        export_data = {}

        for name in self.metrics.keys():
            stats = self.get_stats(name, window_seconds)
            export_data[name] = stats

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class TFANMonitor:
    """
    Real-time monitoring for TFAN production system.
    Tracks key metrics and generates alerts.
    """

    def __init__(self, metrics_dir: str = "metrics"):
        """
        Args:
            metrics_dir: Directory to store metric exports
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.collector = MetricsCollector(window_size=10000)

        # Alert thresholds
        self.thresholds = {
            "epr_cv": 0.15,
            "pgu_p95_latency_ms": 200.0,
            "ttw_p95_latency_ms": 5.0,
            "attention_speedup": 3.0,
            "gpu_memory_mb": 12000,  # For RTX 3090
        }

        self.alerts = []

    def record_training_step(self, step: int, train_loss: float,
                            val_loss: float, lr: float, temp: float,
                            epr_cv: float):
        """Record metrics from a training step."""
        tags = {"step": str(step)}

        self.collector.record("train_loss", train_loss, "loss", tags)
        self.collector.record("val_loss", val_loss, "loss", tags)
        self.collector.record("learning_rate", lr, "lr", tags)
        self.collector.record("temperature", temp, "temp", tags)
        self.collector.record("epr_cv", epr_cv, "cv", tags)

        # Check EPR-CV threshold
        if epr_cv > self.thresholds["epr_cv"]:
            self.alert(f"EPR-CV {epr_cv:.4f} exceeds threshold {self.thresholds['epr_cv']:.4f}",
                      severity="warning")

    def record_inference(self, latency_ms: float, batch_size: int,
                        seq_length: int):
        """Record metrics from inference."""
        tags = {
            "batch_size": str(batch_size),
            "seq_length": str(seq_length)
        }

        self.collector.record("inference_latency_ms", latency_ms, "ms", tags)

        # Compute throughput
        throughput = (batch_size * seq_length) / (latency_ms / 1000.0)  # tokens/sec
        self.collector.record("inference_throughput", throughput, "tokens/sec", tags)

    def record_pgu_metrics(self, latency_ms: float, hit: bool):
        """Record PGU cache metrics."""
        self.collector.record("pgu_latency_ms", latency_ms, "ms")
        self.collector.record("pgu_cache_hit", 1.0 if hit else 0.0, "bool")

        # Check p95 latency
        stats = self.collector.get_stats("pgu_latency_ms", window_seconds=60)
        if stats and stats["p95"] > self.thresholds["pgu_p95_latency_ms"]:
            self.alert(f"PGU p95 latency {stats['p95']:.1f}ms exceeds threshold {self.thresholds['pgu_p95_latency_ms']:.1f}ms",
                      severity="warning")

    def record_ttw_detection(self, latency_ms: float, precursor_detected: bool):
        """Record TTW-Sentry metrics."""
        self.collector.record("ttw_latency_ms", latency_ms, "ms")
        self.collector.record("ttw_precursor_detected", 1.0 if precursor_detected else 0.0, "bool")

        # Check p95 latency
        stats = self.collector.get_stats("ttw_latency_ms", window_seconds=60)
        if stats and stats["p95"] > self.thresholds["ttw_p95_latency_ms"]:
            self.alert(f"TTW p95 latency {stats['p95']:.2f}ms exceeds threshold {self.thresholds['ttw_p95_latency_ms']:.2f}ms",
                      severity="warning")

    def record_system_metrics(self, gpu_memory_mb: float, cpu_percent: float):
        """Record system resource metrics."""
        self.collector.record("gpu_memory_mb", gpu_memory_mb, "MB")
        self.collector.record("cpu_percent", cpu_percent, "%")

        if gpu_memory_mb > self.thresholds["gpu_memory_mb"]:
            self.alert(f"GPU memory {gpu_memory_mb:.0f}MB exceeds threshold {self.thresholds['gpu_memory_mb']:.0f}MB",
                      severity="warning")

    def alert(self, message: str, severity: str = "info"):
        """Generate an alert."""
        alert = {
            "timestamp": time.time(),
            "message": message,
            "severity": severity
        }

        self.alerts.append(alert)
        print(f"[ALERT {severity.upper()}] {message}")

        # Write to alert log
        alert_log = self.metrics_dir / "alerts.jsonl"
        with open(alert_log, 'a') as f:
            f.write(json.dumps(alert) + '\n')

    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data."""
        dashboard = {
            "timestamp": time.time(),
            "metrics": {},
            "alerts": self.alerts[-10:],  # Last 10 alerts
        }

        # Get stats for key metrics
        key_metrics = [
            "epr_cv", "train_loss", "val_loss",
            "pgu_latency_ms", "ttw_latency_ms",
            "inference_latency_ms", "gpu_memory_mb"
        ]

        for metric in key_metrics:
            stats = self.collector.get_stats(metric, window_seconds=300)  # Last 5 min
            if stats:
                dashboard["metrics"][metric] = stats

        return dashboard

    def export_dashboard(self, output_path: Optional[str] = None):
        """Export dashboard data to JSON."""
        if output_path is None:
            output_path = self.metrics_dir / "dashboard.json"

        dashboard_data = self.get_dashboard_data()

        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        print(f"Dashboard exported to: {output_path}")

    def start_export_loop(self, interval_seconds: int = 30):
        """Start background thread to export metrics periodically."""
        def export_loop():
            while True:
                try:
                    # Export Prometheus format
                    self.collector.export_prometheus(
                        str(self.metrics_dir / "metrics.prom")
                    )

                    # Export JSON dashboard
                    self.export_dashboard()

                except Exception as e:
                    print(f"Error exporting metrics: {e}")

                time.sleep(interval_seconds)

        thread = threading.Thread(target=export_loop, daemon=True)
        thread.start()

        print(f"Metrics export loop started (interval: {interval_seconds}s)")


def create_grafana_dashboard_json() -> Dict:
    """
    Create Grafana dashboard JSON configuration.

    Returns:
        Grafana dashboard configuration dict
    """
    dashboard = {
        "dashboard": {
            "title": "TFAN Production Monitoring",
            "tags": ["tfan", "ml", "production"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Training Loss",
                    "type": "graph",
                    "targets": [
                        {"expr": "train_loss", "legendFormat": "Train"},
                        {"expr": "val_loss", "legendFormat": "Validation"}
                    ]
                },
                {
                    "id": 2,
                    "title": "EPR-CV",
                    "type": "graph",
                    "targets": [
                        {"expr": "epr_cv", "legendFormat": "EPR-CV"}
                    ],
                    "thresholds": [
                        {"value": 0.15, "color": "red", "op": "gt"}
                    ]
                },
                {
                    "id": 3,
                    "title": "PGU Latency p95",
                    "type": "stat",
                    "targets": [
                        {"expr": "quantile(0.95, pgu_latency_ms)"}
                    ],
                    "thresholds": [
                        {"value": 200, "color": "red", "op": "gt"}
                    ]
                },
                {
                    "id": 4,
                    "title": "GPU Memory",
                    "type": "gauge",
                    "targets": [
                        {"expr": "gpu_memory_mb"}
                    ],
                    "max": 24000
                },
                {
                    "id": 5,
                    "title": "Inference Throughput",
                    "type": "graph",
                    "targets": [
                        {"expr": "inference_throughput", "legendFormat": "Tokens/sec"}
                    ]
                }
            ]
        }
    }

    return dashboard


if __name__ == "__main__":
    # Demo usage
    print("=== TFAN Monitoring Dashboard Setup ===\n")

    monitor = TFANMonitor(metrics_dir="artifacts/metrics")

    # Simulate training
    for step in range(100):
        train_loss = 1.0 - (step / 100) * 0.5 + (step % 10) * 0.01
        val_loss = train_loss + 0.1
        lr = 1e-4 * (0.95 ** (step // 10))
        temp = 1.0
        epr_cv = 0.2 - (step / 100) * 0.1

        monitor.record_training_step(step, train_loss, val_loss, lr, temp, epr_cv)

        time.sleep(0.01)

    # Export dashboard
    monitor.export_dashboard()

    # Save Grafana config
    grafana_config = create_grafana_dashboard_json()
    with open("monitoring/grafana_dashboard.json", "w") as f:
        json.dump(grafana_config, f, indent=2)

    print("\nMonitoring setup complete!")
