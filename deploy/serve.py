#!/usr/bin/env python3
"""
TFAN Model Serving API
Production-grade FastAPI server with batching, caching, monitoring, and health checks.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.model import TFANModel
from tfan.compression import CompressionPipeline

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================
REQUEST_COUNT = Counter(
    'tfan_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'tfan_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint']
)
BATCH_SIZE = Histogram(
    'tfan_batch_size',
    'Batch size distribution'
)
MODEL_INFERENCE_TIME = Histogram(
    'tfan_inference_duration_seconds',
    'Model inference time in seconds'
)
QUEUE_SIZE = Gauge(
    'tfan_queue_size',
    'Number of requests in queue'
)
CACHE_HITS = Counter(
    'tfan_cache_hits_total',
    'Number of cache hits'
)
CACHE_MISSES = Counter(
    'tfan_cache_misses_total',
    'Number of cache misses'
)

# ============================================================================
# Request/Response Models
# ============================================================================
class PredictionRequest(BaseModel):
    """Request schema for model predictions."""
    input_data: List[List[float]] = Field(
        ...,
        description="Input features as 2D array [batch_size, feature_dim]"
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version (defaults to latest)"
    )
    return_embeddings: bool = Field(
        False,
        description="Return intermediate embeddings"
    )
    return_attention: bool = Field(
        False,
        description="Return attention weights"
    )

    @validator('input_data')
    def validate_input(cls, v):
        if not v or not all(isinstance(row, list) for row in v):
            raise ValueError("input_data must be a non-empty 2D list")
        # Check all rows have same length
        lengths = [len(row) for row in v]
        if len(set(lengths)) > 1:
            raise ValueError("All input rows must have same length")
        return v

class PredictionResponse(BaseModel):
    """Response schema for model predictions."""
    predictions: List[List[float]] = Field(
        ...,
        description="Model predictions"
    )
    embeddings: Optional[List[List[float]]] = Field(
        None,
        description="Intermediate embeddings (if requested)"
    )
    attention_weights: Optional[List[List[List[float]]]] = Field(
        None,
        description="Attention weights (if requested)"
    )
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    batch_size: int = Field(
        ...,
        description="Batch size processed"
    )

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_requests: int
    cache_hit_rate: float

class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    requests_total: int
    average_latency_ms: float
    average_batch_size: float
    cache_hit_rate: float
    queue_size: int

# ============================================================================
# Model Manager
# ============================================================================
class ModelManager:
    """Manages model loading, versioning, and caching."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_compression: bool = True,
        warmup_samples: int = 10
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.use_compression = use_compression
        self.warmup_samples = warmup_samples

        self.model: Optional[nn.Module] = None
        self.model_version: str = "unknown"
        self.model_config: Dict[str, Any] = {}

        # Simple LRU cache for predictions
        self.cache_size = 1000
        self.cache: Dict[str, Any] = {}
        self.cache_queue = deque(maxlen=self.cache_size)

        logger.info(f"ModelManager initialized with device={device}, compression={use_compression}")

    def load_model(self):
        """Load model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract metadata
            self.model_version = checkpoint.get('version', 'v1.0.0')
            self.model_config = checkpoint.get('config', {})

            # Initialize model architecture
            model = TFANModel(**self.model_config)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Apply compression if enabled
            if self.use_compression:
                logger.info("Applying model compression...")
                from tfan.compression import ModelQuantizer
                quantizer = ModelQuantizer(model)
                model = quantizer.dynamic_quantize(dtype=torch.qint8)
                logger.info("Model compressed with INT8 quantization")

            model = model.to(self.device)
            model.eval()

            self.model = model

            # Warmup
            self._warmup()

            logger.info(f"Model loaded successfully (version={self.model_version})")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _warmup(self):
        """Warmup model with dummy inputs."""
        if self.model is None:
            return

        logger.info(f"Warming up model with {self.warmup_samples} samples...")

        # Get input dimension from config
        input_dim = self.model_config.get('input_dim', 128)
        seq_len = self.model_config.get('seq_len', 64)

        with torch.no_grad():
            for _ in range(self.warmup_samples):
                dummy_input = torch.randn(1, seq_len, input_dim, device=self.device)
                _ = self.model(dummy_input)

        logger.info("Model warmup complete")

    @torch.no_grad()
    def predict(
        self,
        input_data: np.ndarray,
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """Run inference on input data."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Check cache
        cache_key = self._compute_cache_key(input_data)
        if cache_key in self.cache and not (return_embeddings or return_attention):
            CACHE_HITS.inc()
            return self.cache[cache_key]

        CACHE_MISSES.inc()

        # Convert to tensor
        x = torch.from_numpy(input_data).float().to(self.device)

        # Inference
        start_time = time.time()

        if return_embeddings or return_attention:
            # Forward with intermediate outputs
            outputs = self.model.forward_with_intermediates(x)
            predictions = outputs['output']
            embeddings = outputs.get('embeddings', None)
            attention = outputs.get('attention', None)
        else:
            predictions = self.model(x)
            embeddings = None
            attention = None

        inference_time = time.time() - start_time
        MODEL_INFERENCE_TIME.observe(inference_time)

        # Convert to numpy
        result = {
            'predictions': predictions.cpu().numpy(),
            'embeddings': embeddings.cpu().numpy() if embeddings is not None else None,
            'attention': attention.cpu().numpy() if attention is not None else None,
            'inference_time_ms': inference_time * 1000,
            'batch_size': len(input_data)
        }

        # Cache result (only if no extra outputs requested)
        if not (return_embeddings or return_attention):
            self._cache_result(cache_key, result)

        return result

    def _compute_cache_key(self, input_data: np.ndarray) -> str:
        """Compute cache key from input data."""
        # Simple hash of input array
        return str(hash(input_data.tobytes()))

    def _cache_result(self, key: str, result: Dict[str, Any]):
        """Add result to cache with LRU eviction."""
        if len(self.cache_queue) >= self.cache_size:
            # Evict oldest
            oldest_key = self.cache_queue.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]

        self.cache[key] = result
        self.cache_queue.append(key)

# ============================================================================
# Request Batcher
# ============================================================================
class RequestBatcher:
    """Batches incoming requests for efficient processing."""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: int = 50
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.queue: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()

        logger.info(f"RequestBatcher initialized (max_batch={max_batch_size}, max_wait={max_wait_ms}ms)")

    async def add_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add request to batch and wait for processing."""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append({
                'data': request_data,
                'future': future
            })
            QUEUE_SIZE.set(len(self.queue))

        # Wait for result
        result = await future
        return result

    async def process_batch(self, model_manager: ModelManager):
        """Process batched requests."""
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000.0)

            if not self.queue:
                continue

            async with self.lock:
                if not self.queue:
                    continue

                # Get batch
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]
                QUEUE_SIZE.set(len(self.queue))

            if not batch:
                continue

            # Combine inputs
            batch_inputs = [item['data']['input_data'] for item in batch]
            batch_array = np.concatenate(batch_inputs, axis=0)

            BATCH_SIZE.observe(len(batch_array))

            # Run inference
            try:
                result = model_manager.predict(
                    batch_array,
                    return_embeddings=batch[0]['data'].get('return_embeddings', False),
                    return_attention=batch[0]['data'].get('return_attention', False)
                )

                # Split results
                offset = 0
                for item in batch:
                    batch_size = len(item['data']['input_data'])
                    item_result = {
                        'predictions': result['predictions'][offset:offset+batch_size],
                        'embeddings': result['embeddings'][offset:offset+batch_size] if result['embeddings'] is not None else None,
                        'attention': result['attention'][offset:offset+batch_size] if result['attention'] is not None else None,
                        'inference_time_ms': result['inference_time_ms'],
                        'batch_size': batch_size
                    }
                    item['future'].set_result(item_result)
                    offset += batch_size

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for item in batch:
                    item['future'].set_exception(e)

# ============================================================================
# Application Lifespan
# ============================================================================
app_state = {
    'model_manager': None,
    'batcher': None,
    'start_time': None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting TFAN Model Server...")
    app_state['start_time'] = time.time()

    # Initialize model manager
    model_path = Path(__file__).parent / "models" / "tfan_model.pt"
    app_state['model_manager'] = ModelManager(
        model_path=str(model_path),
        use_compression=True
    )

    # Load model
    if not app_state['model_manager'].load_model():
        logger.error("Failed to load model, but continuing...")

    # Initialize batcher
    app_state['batcher'] = RequestBatcher(
        max_batch_size=32,
        max_wait_ms=50
    )

    # Start batch processing task
    batch_task = asyncio.create_task(
        app_state['batcher'].process_batch(app_state['model_manager'])
    )

    logger.info("TFAN Model Server started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down TFAN Model Server...")
    batch_task.cancel()
    try:
        await batch_task
    except asyncio.CancelledError:
        pass

# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="TFAN Model Serving API",
    description="Production-grade model serving with batching, caching, and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Endpoints
# ============================================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "TFAN Model Serving API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_manager = app_state['model_manager']

    # Calculate metrics
    uptime = time.time() - app_state['start_time']

    cache_hits = CACHE_HITS._value.get()
    cache_misses = CACHE_MISSES._value.get()
    cache_total = cache_hits + cache_misses
    cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

    # Get total requests
    total_requests = 0
    for metric in REGISTRY.collect():
        if metric.name == 'tfan_requests_total':
            for sample in metric.samples:
                if sample.name == 'tfan_requests_total_total':
                    total_requests += sample.value

    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_version=model_manager.model_version,
        uptime_seconds=uptime,
        total_requests=int(total_requests),
        cache_hit_rate=cache_hit_rate
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest, bg_tasks: BackgroundTasks):
    """
    Run model inference on input data.

    Supports batching and caching for optimal performance.
    """
    start_time = time.time()

    try:
        model_manager = app_state['model_manager']

        if model_manager.model is None:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert input to numpy
        input_array = np.array(request.input_data, dtype=np.float32)

        # Add to batch queue
        result = await app_state['batcher'].add_request({
            'input_data': input_array,
            'return_embeddings': request.return_embeddings,
            'return_attention': request.return_attention
        })

        # Build response
        response = PredictionResponse(
            predictions=result['predictions'].tolist(),
            embeddings=result['embeddings'].tolist() if result['embeddings'] is not None else None,
            attention_weights=result['attention'].tolist() if result['attention'] is not None else None,
            model_version=model_manager.model_version,
            inference_time_ms=result['inference_time_ms'],
            batch_size=result['batch_size']
        )

        # Record metrics
        REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint for multiple inputs."""
    results = []
    for req in requests:
        result = await predict(req, BackgroundTasks())
        results.append(result)
    return results

# Fix: Import Response from starlette
from starlette.responses import Response

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TFAN Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )
