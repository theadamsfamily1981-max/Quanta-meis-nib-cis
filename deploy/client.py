#!/usr/bin/env python3
"""
TFAN Client SDK
Easy-to-use client for interacting with TFAN Model Server.
"""

import logging
from typing import List, Optional, Dict, Any, Union
import time
from dataclasses import dataclass

import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray
    embeddings: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    model_version: str = ""
    inference_time_ms: float = 0.0
    batch_size: int = 0
    request_time_ms: float = 0.0


class TFANClient:
    """Client for TFAN Model Server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize TFAN client.

        Args:
            base_url: Base URL of TFAN server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TFAN-Client/1.0.0'
        })

        logger.info(f"TFAN Client initialized with base_url={base_url}")

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            Health status dictionary
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        input_data: Union[List[List[float]], np.ndarray],
        model_version: Optional[str] = None,
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> PredictionResult:
        """
        Run prediction on input data.

        Args:
            input_data: Input features as 2D array [batch_size, feature_dim]
            model_version: Specific model version (None for latest)
            return_embeddings: Return intermediate embeddings
            return_attention: Return attention weights

        Returns:
            PredictionResult containing predictions and optional outputs
        """
        # Convert numpy to list if needed
        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()

        # Build request
        payload = {
            'input_data': input_data,
            'return_embeddings': return_embeddings,
            'return_attention': return_attention
        }

        if model_version:
            payload['model_version'] = model_version

        # Make request with retries
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                break

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Prediction failed after {self.max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Prediction attempt {attempt + 1} failed, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))

        request_time = (time.time() - start_time) * 1000

        # Parse response
        data = response.json()

        return PredictionResult(
            predictions=np.array(data['predictions']),
            embeddings=np.array(data['embeddings']) if data.get('embeddings') else None,
            attention_weights=np.array(data['attention_weights']) if data.get('attention_weights') else None,
            model_version=data['model_version'],
            inference_time_ms=data['inference_time_ms'],
            batch_size=data['batch_size'],
            request_time_ms=request_time
        )

    def predict_batch(
        self,
        batch_inputs: List[Union[List[List[float]], np.ndarray]],
        **kwargs
    ) -> List[PredictionResult]:
        """
        Run predictions on multiple batches.

        Args:
            batch_inputs: List of input batches
            **kwargs: Additional arguments passed to predict()

        Returns:
            List of PredictionResult objects
        """
        results = []
        for input_data in batch_inputs:
            result = self.predict(input_data, **kwargs)
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get server metrics.

        Returns:
            Metrics dictionary
        """
        response = self.session.get(
            f"{self.base_url}/metrics",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.text  # Prometheus format

    def close(self):
        """Close the client session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncTFANClient:
    """Async client for TFAN Model Server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize async TFAN client.

        Requires: pip install aiohttp
        """
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError("AsyncTFANClient requires aiohttp: pip install aiohttp")

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = None

        logger.info(f"Async TFAN Client initialized with base_url={base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = self.aiohttp.ClientSession(
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'TFAN-AsyncClient/1.0.0'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check server health (async)."""
        async with self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def predict(
        self,
        input_data: Union[List[List[float]], np.ndarray],
        model_version: Optional[str] = None,
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> PredictionResult:
        """Run prediction (async)."""
        # Convert numpy to list if needed
        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()

        # Build request
        payload = {
            'input_data': input_data,
            'return_embeddings': return_embeddings,
            'return_attention': return_attention
        }

        if model_version:
            payload['model_version'] = model_version

        # Make request with retries
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    break

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Prediction failed after {self.max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Prediction attempt {attempt + 1} failed, retrying...")
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        request_time = (time.time() - start_time) * 1000

        return PredictionResult(
            predictions=np.array(data['predictions']),
            embeddings=np.array(data['embeddings']) if data.get('embeddings') else None,
            attention_weights=np.array(data['attention_weights']) if data.get('attention_weights') else None,
            model_version=data['model_version'],
            inference_time_ms=data['inference_time_ms'],
            batch_size=data['batch_size'],
            request_time_ms=request_time
        )


# ============================================================================
# CLI for testing
# ============================================================================
def main():
    """CLI for testing TFAN client."""
    import argparse

    parser = argparse.ArgumentParser(description="TFAN Client CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--health", action="store_true", help="Check health")
    parser.add_argument("--predict", action="store_true", help="Run test prediction")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--feature-dim", type=int, default=128, help="Feature dimension")

    args = parser.parse_args()

    # Create client
    with TFANClient(base_url=args.url) as client:

        # Health check
        if args.health:
            logger.info("Checking server health...")
            health = client.health_check()
            logger.info(f"Health: {health}")

        # Test prediction
        if args.predict:
            logger.info("Running test prediction...")

            # Generate random test data
            test_data = np.random.randn(args.batch_size, args.feature_dim).astype(np.float32)

            # Run prediction
            result = client.predict(
                test_data,
                return_embeddings=True,
                return_attention=True
            )

            logger.info(f"Predictions shape: {result.predictions.shape}")
            logger.info(f"Model version: {result.model_version}")
            logger.info(f"Inference time: {result.inference_time_ms:.2f}ms")
            logger.info(f"Total request time: {result.request_time_ms:.2f}ms")

            if result.embeddings is not None:
                logger.info(f"Embeddings shape: {result.embeddings.shape}")

            if result.attention_weights is not None:
                logger.info(f"Attention weights shape: {result.attention_weights.shape}")


if __name__ == "__main__":
    main()
