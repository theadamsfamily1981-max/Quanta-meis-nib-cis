#!/usr/bin/env python3
"""
Load Testing Suite for TFAN Model Server
Uses Locust for distributed load testing.

Usage:
    # Web UI mode
    locust -f load_test.py --host=http://localhost:8000

    # Headless mode
    locust -f load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless
"""

import json
import random
import time
from typing import List

import numpy as np
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


class TFANUser(FastHttpUser):
    """
    Simulates a user making predictions against TFAN server.
    Uses FastHttpUser for better performance.
    """

    # Wait time between requests (1-3 seconds)
    wait_time = between(1, 3)

    # Model configuration
    feature_dim = 128
    batch_sizes = [1, 2, 4, 8, 16]

    def on_start(self):
        """Called when a simulated user starts."""
        # Check server health
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(10)
    def predict_single(self):
        """Single batch prediction (most common scenario)."""
        batch_size = random.choice([1, 2, 4])
        input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

        payload = {
            'input_data': input_data.tolist(),
            'return_embeddings': False,
            'return_attention': False
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [batch_size=small]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response
                if 'predictions' in data and 'model_version' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Request failed: {response.status_code}")

    @task(5)
    def predict_medium_batch(self):
        """Medium batch prediction."""
        batch_size = random.choice([8, 16])
        input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

        payload = {
            'input_data': input_data.tolist(),
            'return_embeddings': False,
            'return_attention': False
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [batch_size=medium]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Request failed: {response.status_code}")

    @task(2)
    def predict_with_embeddings(self):
        """Prediction with embeddings (more expensive)."""
        batch_size = random.choice([1, 2])
        input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

        payload = {
            'input_data': input_data.tolist(),
            'return_embeddings': True,
            'return_attention': False
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [with_embeddings]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'embeddings' in data and data['embeddings'] is not None:
                    response.success()
                else:
                    response.failure("Missing embeddings in response")
            else:
                response.failure(f"Request failed: {response.status_code}")

    @task(1)
    def predict_with_attention(self):
        """Prediction with attention weights (most expensive)."""
        batch_size = 1
        input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

        payload = {
            'input_data': input_data.tolist(),
            'return_embeddings': True,
            'return_attention': True
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict [with_attention]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'attention_weights' in data and data['attention_weights'] is not None:
                    response.success()
                else:
                    response.failure("Missing attention weights in response")
            else:
                response.failure(f"Request failed: {response.status_code}")

    @task(1)
    def health_check(self):
        """Periodic health checks."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"Health check failed: {response.status_code}")


# ============================================================================
# Custom Event Handlers for Advanced Metrics
# ============================================================================

# Track latency percentiles
latencies = []

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Record request metrics."""
    if exception is None:
        latencies.append(response_time)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Calculate and print statistics at test end."""
    if latencies:
        latencies_array = np.array(latencies)
        print("\n" + "="*80)
        print("LOAD TEST RESULTS")
        print("="*80)
        print(f"Total requests: {len(latencies)}")
        print(f"Mean latency: {np.mean(latencies_array):.2f}ms")
        print(f"Median latency: {np.median(latencies_array):.2f}ms")
        print(f"P95 latency: {np.percentile(latencies_array, 95):.2f}ms")
        print(f"P99 latency: {np.percentile(latencies_array, 99):.2f}ms")
        print(f"Min latency: {np.min(latencies_array):.2f}ms")
        print(f"Max latency: {np.max(latencies_array):.2f}ms")
        print("="*80)


# ============================================================================
# Stress Test Scenarios
# ============================================================================

class StressTestUser(FastHttpUser):
    """Stress test with constant load."""
    wait_time = between(0.1, 0.5)  # Minimal wait time
    feature_dim = 128

    @task
    def predict_stress(self):
        """Constant prediction requests."""
        batch_size = random.choice([1, 2, 4, 8])
        input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

        payload = {'input_data': input_data.tolist()}

        self.client.post("/predict", json=payload, name="/predict [stress]")


class SpikeTestUser(FastHttpUser):
    """Spike test with burst traffic."""
    wait_time = between(5, 10)  # Long wait, then burst
    feature_dim = 128

    @task
    def predict_burst(self):
        """Send burst of requests."""
        for _ in range(10):  # Burst of 10 requests
            batch_size = random.choice([1, 2, 4])
            input_data = np.random.randn(batch_size, self.feature_dim).astype(np.float32)

            payload = {'input_data': input_data.tolist()}

            self.client.post("/predict", json=payload, name="/predict [spike]")


# ============================================================================
# Standalone Benchmark Script
# ============================================================================

def run_benchmark(
    url: str = "http://localhost:8000",
    num_requests: int = 1000,
    batch_size: int = 4,
    feature_dim: int = 128
):
    """
    Run standalone benchmark without Locust.

    Args:
        url: Server URL
        num_requests: Number of requests to send
        batch_size: Batch size for predictions
        feature_dim: Feature dimension
    """
    import requests
    from tqdm import tqdm

    print(f"Running benchmark: {num_requests} requests with batch_size={batch_size}")

    session = requests.Session()
    latencies = []

    for _ in tqdm(range(num_requests)):
        input_data = np.random.randn(batch_size, feature_dim).astype(np.float32)
        payload = {'input_data': input_data.tolist()}

        start_time = time.time()
        response = session.post(f"{url}/predict", json=payload)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            latencies.append(latency)

    # Calculate stats
    latencies_array = np.array(latencies)

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Total requests: {len(latencies)}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Mean latency: {np.mean(latencies_array):.2f}ms")
    print(f"Median latency: {np.median(latencies_array):.2f}ms")
    print(f"P95 latency: {np.percentile(latencies_array, 95):.2f}ms")
    print(f"P99 latency: {np.percentile(latencies_array, 99):.2f}ms")
    print(f"Min latency: {np.min(latencies_array):.2f}ms")
    print(f"Max latency: {np.max(latencies_array):.2f}ms")
    print(f"Throughput: {len(latencies) / (sum(latencies) / 1000):.2f} req/s")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TFAN Load Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--benchmark", action="store_true", help="Run standalone benchmark")
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            url=args.url,
            num_requests=args.num_requests,
            batch_size=args.batch_size
        )
    else:
        print("Use 'locust -f load_test.py --host=<url>' to run load tests")
        print("Or use --benchmark flag to run standalone benchmark")
