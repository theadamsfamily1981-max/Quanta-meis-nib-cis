# TFAN Model Deployment Guide

**Complete production-ready deployment infrastructure for TFAN models**

This directory contains everything needed to deploy TFAN models to production, from local development to cloud-scale Kubernetes clusters.

## 🎯 Quick Start

### Option 1: Local Deployment (Docker Compose)

**Fastest way to get started:**

```bash
# 1. Train and export model
python examples/train_example.py --dataset cifar10 --epochs 50
python deploy/export_model.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output-dir ./deploy/models \
    --compress \
    --benchmark

# 2. Deploy locally
cd deploy
./deploy.sh build
./deploy.sh deploy-local

# 3. Test
./deploy.sh test

# 4. Access services
# - API: http://localhost:8000/docs
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Option 2: Kubernetes Deployment

**For production scale:**

```bash
# 1. Build and push to registry
export REGISTRY=gcr.io/your-project
export IMAGE_TAG=v1.0.0
./deploy.sh build
./deploy.sh push

# 2. Deploy to K8s
./deploy.sh deploy-k8s

# 3. Check status
kubectl get pods -l app=tfan-server
kubectl logs -f deployment/tfan-server
```

---

## 📦 What's Included

### Core Components

| Component | Description | File |
|-----------|-------------|------|
| **FastAPI Server** | Production API with batching & caching | `serve.py` |
| **Docker Image** | Multi-stage optimized container | `Dockerfile` |
| **K8s Manifests** | Production orchestration | `k8s/*.yaml` |
| **Client SDK** | Python client library | `client.py` |
| **Load Testing** | Locust-based load tests | `load_test.py` |
| **Monitoring** | Prometheus + Grafana | `monitoring/` |
| **Automation** | Deployment scripts | `deploy.sh` |

### Infrastructure Features

✅ **High Performance**
- Request batching (up to 32 samples)
- LRU caching
- Async processing
- TorchScript optimization
- INT8 quantization support

✅ **Production Ready**
- Health checks & readiness probes
- Graceful shutdown
- Auto-scaling (HPA)
- Zero-downtime updates
- Comprehensive logging

✅ **Monitoring & Observability**
- Prometheus metrics export
- Grafana dashboards
- Alert rules
- Request tracing
- Performance profiling

✅ **Security**
- Non-root containers
- Resource limits
- CORS configuration
- TLS support (K8s ingress)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
│                    (Nginx / K8s Service)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼───┐    ┌───▼────┐   ┌───▼────┐
    │ TFAN   │    │ TFAN   │   │ TFAN   │
    │ Server │    │ Server │   │ Server │
    │  Pod 1 │    │  Pod 2 │   │  Pod 3 │
    └────┬───┘    └───┬────┘   └───┬────┘
         │            │            │
         └────────────┼────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼───┐   ┌───▼────┐  ┌───▼────┐
    │Prometheus  │Grafana │  │ Redis  │
    │(Metrics) │ │(Dash)  │  │(Cache) │
    └──────────┘ └────────┘  └────────┘
```

### Request Flow

1. **Client** → Sends prediction request
2. **Load Balancer** → Routes to healthy pod
3. **API Server** → Validates request
4. **Batcher** → Combines requests for efficiency
5. **Model** → Runs inference (GPU/CPU)
6. **Cache** → Stores recent results
7. **Response** → Returns predictions
8. **Metrics** → Records performance data

---

## 📖 Detailed Documentation

### 1. Model Export

Export trained models for deployment:

```bash
python deploy/export_model.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir ./deploy/models \
    --format all \                    # pytorch, torchscript, onnx
    --compress \                      # Apply quantization + pruning
    --pruning-ratio 0.4 \            # 40% pruning
    --quantization dynamic \          # INT8 quantization
    --benchmark                       # Run performance tests
```

**Outputs:**
- `tfan_model.pt` - PyTorch checkpoint
- `tfan_model_traced.pt` - TorchScript (faster inference)
- `tfan_model.onnx` - ONNX (cross-platform)
- `tfan_model_compressed.pt` - Compressed (8-20× smaller)
- `model_config.json` - Model configuration

**Compression Results:**
- **Size**: 8-20× reduction
- **Latency**: 2-4× faster
- **Accuracy**: <1% degradation

### 2. API Server

**Endpoints:**

```python
# Health check
GET /health
Response: {
    "status": "healthy",
    "model_loaded": true,
    "model_version": "v1.0.0",
    "uptime_seconds": 3600,
    "cache_hit_rate": 0.75
}

# Prediction
POST /predict
Request: {
    "input_data": [[...], [...]],      # 2D array [batch, features]
    "return_embeddings": false,         # Optional
    "return_attention": false           # Optional
}
Response: {
    "predictions": [[...], [...]],
    "model_version": "v1.0.0",
    "inference_time_ms": 15.2,
    "batch_size": 2
}

# Metrics (Prometheus format)
GET /metrics
```

**Features:**

- **Batching**: Automatically combines requests (max 32, 50ms wait)
- **Caching**: LRU cache (1000 entries)
- **Health Checks**: Liveness & readiness probes
- **Monitoring**: Prometheus metrics export

**Configuration (Environment Variables):**

```bash
TFAN_MODEL_PATH=/app/models/tfan_model.pt
LOG_LEVEL=INFO
MAX_BATCH_SIZE=32
MAX_WAIT_MS=50
```

### 3. Client SDK

**Basic Usage:**

```python
from deploy.client import TFANClient
import numpy as np

# Create client
with TFANClient(base_url="http://localhost:8000") as client:

    # Health check
    health = client.health_check()
    print(f"Status: {health['status']}")

    # Prediction
    input_data = np.random.randn(4, 128)  # [batch, features]
    result = client.predict(input_data)

    print(f"Predictions: {result.predictions.shape}")
    print(f"Latency: {result.inference_time_ms:.2f}ms")
    print(f"Model: {result.model_version}")
```

**Advanced Features:**

```python
# Get embeddings and attention
result = client.predict(
    input_data,
    return_embeddings=True,
    return_attention=True
)

# Batch predictions
results = client.predict_batch([
    input_batch_1,
    input_batch_2,
    input_batch_3
])

# Async client
from deploy.client import AsyncTFANClient

async with AsyncTFANClient(base_url="http://localhost:8000") as client:
    result = await client.predict(input_data)
```

**CLI Testing:**

```bash
# Health check
python deploy/client.py --url http://localhost:8000 --health

# Test prediction
python deploy/client.py --url http://localhost:8000 --predict \
    --batch-size 8 --feature-dim 128
```

### 4. Docker Deployment

**Build:**

```bash
# Standard build
docker build -f deploy/Dockerfile -t tfan-server:latest .

# Multi-platform build
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f deploy/Dockerfile \
    -t tfan-server:latest .

# With build args
docker build \
    --build-arg PYTHON_VERSION=3.10 \
    -f deploy/Dockerfile \
    -t tfan-server:latest .
```

**Run:**

```bash
# Single container
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/deploy/models:/app/models:ro \
    -e TFAN_MODEL_PATH=/app/models/tfan_model.pt \
    tfan-server:latest

# Docker Compose (full stack)
cd deploy
docker-compose up -d

# Check logs
docker-compose logs -f tfan-server

# Stop
docker-compose down
```

**GPU Support:**

```bash
# Enable GPU in docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 5. Kubernetes Deployment

**Prerequisites:**

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Configure cluster access
export KUBECONFIG=~/.kube/config
```

**Deploy:**

```bash
# Apply all manifests
kubectl apply -f deploy/k8s/

# Or individually
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/hpa.yaml

# Check status
kubectl get pods -l app=tfan-server
kubectl get svc tfan-server

# View logs
kubectl logs -f deployment/tfan-server

# Port forward for testing
kubectl port-forward svc/tfan-server 8000:80
```

**Scaling:**

```bash
# Manual scaling
kubectl scale deployment tfan-server --replicas=5

# Auto-scaling (HPA)
# Configured in k8s/hpa.yaml
# Scales 3-10 pods based on CPU/memory/RPS

# Check HPA status
kubectl get hpa tfan-server-hpa
```

**Update Deployment:**

```bash
# Rolling update
kubectl set image deployment/tfan-server \
    tfan-server=tfan-server:v1.1.0

# Monitor rollout
kubectl rollout status deployment/tfan-server

# Rollback if needed
kubectl rollout undo deployment/tfan-server
```

**GPU Nodes:**

Uncomment GPU configuration in `k8s/deployment.yaml`:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1

affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: accelerator
          operator: In
          values:
          - nvidia-tesla-v100
```

### 6. Monitoring

**Prometheus Metrics:**

Available at `/metrics`:

- `tfan_requests_total` - Total requests (by endpoint, status)
- `tfan_request_duration_seconds` - Request latency histogram
- `tfan_batch_size` - Batch size distribution
- `tfan_inference_duration_seconds` - Model inference time
- `tfan_queue_size` - Current queue size
- `tfan_cache_hits_total` / `tfan_cache_misses_total` - Cache stats

**Grafana Dashboards:**

Access at `http://localhost:3000` (admin/admin)

Pre-configured panels:
- Request rate & error rate
- Latency percentiles (P50, P95, P99)
- Batch size distribution
- Queue depth
- Cache hit rate
- Resource usage (CPU, memory)

**Alerts:**

Configured in `monitoring/alerts/tfan_alerts.yml`:

- High error rate (>5%)
- High latency (P95 >1s)
- Server down (>2min)
- High memory usage (>3.5GB)
- Large queue (>100 requests)
- Low cache hit rate (<50%)

**Custom Metrics:**

```python
from prometheus_client import Counter, Histogram

custom_metric = Counter('custom_requests', 'Custom request counter')
custom_metric.inc()
```

### 7. Load Testing

**Locust (Web UI):**

```bash
# Start Locust
locust -f deploy/load_test.py --host=http://localhost:8000

# Open browser to http://localhost:8089
# Set users: 100, spawn rate: 10
# Click "Start swarming"
```

**Headless Mode:**

```bash
# Run 2-minute load test
locust -f deploy/load_test.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 2m \
    --headless
```

**Standalone Benchmark:**

```bash
python deploy/load_test.py \
    --benchmark \
    --url http://localhost:8000 \
    --num-requests 1000 \
    --batch-size 4
```

**Test Scenarios:**

- `TFANUser` - Realistic traffic (10% small batch, 5% medium, 2% w/ embeddings)
- `StressTestUser` - Constant high load (minimal wait)
- `SpikeTestUser` - Burst traffic (long wait, then 10× spike)

**Expected Performance:**

- **Latency**: P95 <100ms (CPU), <50ms (GPU)
- **Throughput**: 100-1000 req/s (depends on batch size, hardware)
- **Batch efficiency**: 3-10× speedup vs individual requests

### 8. CI/CD Integration

**GitHub Actions Example:**

```yaml
name: Deploy TFAN Model

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          cd deploy
          ./deploy.sh build

      - name: Run tests
        run: |
          cd deploy
          ./deploy.sh deploy-local
          sleep 30
          ./deploy.sh test

      - name: Push to registry
        env:
          REGISTRY: gcr.io/my-project
        run: |
          cd deploy
          echo ${{ secrets.GCR_KEY }} | docker login -u _json_key --password-stdin gcr.io
          ./deploy.sh push

      - name: Deploy to K8s
        run: |
          echo "${{ secrets.KUBECONFIG }}" > kubeconfig
          export KUBECONFIG=./kubeconfig
          cd deploy
          ./deploy.sh deploy-k8s
```

---

## 🔧 Configuration

### Model Configuration

Edit `model_config.json`:

```json
{
  "input_dim": 128,
  "hidden_dim": 256,
  "output_dim": 10,
  "num_heads": 8,
  "num_layers": 6,
  "seq_len": 64,
  "dropout": 0.1
}
```

### Server Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TFAN_MODEL_PATH` | - | Path to model file |
| `LOG_LEVEL` | INFO | Logging level |
| `MAX_BATCH_SIZE` | 32 | Maximum batch size |
| `MAX_WAIT_MS` | 50 | Max batching wait time |
| `CACHE_SIZE` | 1000 | LRU cache size |
| `WORKERS` | 1 | Number of workers |

---

## 🚀 Performance Optimization

### 1. Model Optimization

**Quantization:**
```bash
# INT8 (fastest, 4× smaller)
python deploy/export_model.py --quantization dynamic

# FP16 (good balance)
python deploy/export_model.py --quantization fp16
```

**Pruning:**
```bash
# 40% pruning (recommended)
python deploy/export_model.py --pruning-ratio 0.4

# Aggressive 60% pruning
python deploy/export_model.py --pruning-ratio 0.6
```

**TorchScript:**
```bash
# Traced model (faster inference)
python deploy/export_model.py --format torchscript
```

### 2. Server Optimization

**Batching:**
```python
# Larger batches = higher throughput
MAX_BATCH_SIZE=64
MAX_WAIT_MS=100
```

**Caching:**
```python
# Larger cache = higher hit rate
CACHE_SIZE=5000
```

**Workers:**
```bash
# Multi-worker deployment
uvicorn deploy.serve:app --workers 4
```

### 3. Infrastructure Optimization

**Horizontal Scaling:**
```yaml
# K8s HPA
minReplicas: 3
maxReplicas: 20
targetCPUUtilization: 70
```

**Vertical Scaling:**
```yaml
# Increase resources
resources:
  requests:
    cpu: "2000m"
    memory: "8Gi"
```

**GPU Acceleration:**
```yaml
# Enable GPU
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## 🐛 Troubleshooting

### Server Won't Start

```bash
# Check logs
docker-compose logs tfan-server

# Common issues:
# 1. Model not found
ls -la deploy/models/tfan_model.pt

# 2. Port conflict
lsof -i :8000
kill -9 <PID>

# 3. Permission denied
chmod 644 deploy/models/tfan_model.pt
```

### High Latency

```bash
# Check metrics
curl http://localhost:8000/metrics | grep latency

# Possible causes:
# 1. Large batch size → Reduce MAX_BATCH_SIZE
# 2. CPU bottleneck → Enable GPU or scale horizontally
# 3. Network issues → Check load balancer config
```

### Out of Memory

```bash
# Monitor memory
docker stats tfan-server

# Solutions:
# 1. Use compressed model
python deploy/export_model.py --compress

# 2. Reduce batch size
MAX_BATCH_SIZE=16

# 3. Increase container memory
docker run -m 8g tfan-server:latest
```

### Low Cache Hit Rate

```bash
# Check cache stats
curl http://localhost:8000/health | jq '.cache_hit_rate'

# Solutions:
# 1. Increase cache size
CACHE_SIZE=10000

# 2. Check request patterns (too diverse?)
# 3. Warm up cache with common requests
```

---

## 📊 Production Checklist

Before going to production:

- [ ] Model trained and validated
- [ ] Model exported and compressed
- [ ] Load testing completed (expected traffic + 2×)
- [ ] Monitoring dashboards configured
- [ ] Alert rules tested
- [ ] Health checks passing
- [ ] Auto-scaling configured
- [ ] Backup/rollback plan in place
- [ ] Security review completed
- [ ] Documentation updated
- [ ] On-call team trained
- [ ] Incident runbook prepared

---

## 🔗 Additional Resources

- **API Documentation**: http://localhost:8000/docs (OpenAPI/Swagger)
- **TFAN Documentation**: `../docs/PRODUCTION_GUIDE.md`
- **Model Training**: `../examples/train_example.py`
- **Issue Tracker**: GitHub Issues

---

## 📝 License

See main project LICENSE file.

---

**Ready to deploy? Start with `./deploy.sh build` and follow the Quick Start guide above!** 🚀
