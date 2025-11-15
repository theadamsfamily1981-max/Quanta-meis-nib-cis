#!/bin/bash
# Automated Deployment Script for TFAN Model Server
# Supports: Docker, Docker Compose, Kubernetes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"
IMAGE_NAME="tfan-server"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

# ============================================================================
# Build Functions
# ============================================================================

build_docker() {
    log_info "Building Docker image..."

    cd "$PROJECT_ROOT"

    docker build \
        -f deploy/Dockerfile \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        .

    log_info "Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"
}

push_docker() {
    if [ -z "$REGISTRY" ]; then
        log_warn "REGISTRY not set, skipping push"
        return 0
    fi

    log_info "Pushing to registry: $REGISTRY"

    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    log_info "Image pushed: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
}

# ============================================================================
# Local Deployment (Docker Compose)
# ============================================================================

deploy_local() {
    log_info "Deploying locally with Docker Compose..."

    check_command docker-compose || check_command docker

    cd "$DEPLOY_DIR"

    # Check if model exists
    if [ ! -f "$DEPLOY_DIR/models/tfan_model.pt" ]; then
        log_error "Model not found at $DEPLOY_DIR/models/tfan_model.pt"
        log_info "Please export your model first:"
        log_info "  python deploy/export_model.py --checkpoint /path/to/checkpoint.pt --output-dir deploy/models"
        exit 1
    fi

    # Start services
    docker-compose up -d

    log_info "Services starting..."
    log_info "  - TFAN Server: http://localhost:8000"
    log_info "  - API Docs: http://localhost:8000/docs"
    log_info "  - Grafana: http://localhost:3000 (admin/admin)"
    log_info "  - Prometheus: http://localhost:9090"

    # Wait for health check
    log_info "Waiting for server to be healthy..."
    max_attempts=30
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "Server is healthy!"
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    if [ $attempt -eq $max_attempts ]; then
        log_error "Server failed to become healthy"
        docker-compose logs tfan-server
        exit 1
    fi
}

stop_local() {
    log_info "Stopping local deployment..."

    cd "$DEPLOY_DIR"
    docker-compose down

    log_info "Services stopped"
}

# ============================================================================
# Kubernetes Deployment
# ============================================================================

deploy_k8s() {
    log_info "Deploying to Kubernetes..."

    check_command kubectl || { log_error "kubectl not found"; exit 1; }

    cd "$DEPLOY_DIR/k8s"

    # Apply manifests
    log_info "Creating PVC..."
    kubectl apply -f pvc.yaml

    log_info "Creating Deployment..."
    kubectl apply -f deployment.yaml

    log_info "Creating Service..."
    kubectl apply -f service.yaml

    log_info "Creating HPA..."
    kubectl apply -f hpa.yaml

    # Wait for rollout
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/tfan-server --timeout=5m

    # Get service endpoint
    log_info "Deployment complete!"
    kubectl get svc tfan-server-loadbalancer

    log_info "To check status:"
    log_info "  kubectl get pods -l app=tfan-server"
    log_info "  kubectl logs -f deployment/tfan-server"
}

undeploy_k8s() {
    log_info "Removing Kubernetes deployment..."

    check_command kubectl || { log_error "kubectl not found"; exit 1; }

    cd "$DEPLOY_DIR/k8s"

    kubectl delete -f hpa.yaml || true
    kubectl delete -f service.yaml || true
    kubectl delete -f deployment.yaml || true
    kubectl delete -f pvc.yaml || true

    log_info "Kubernetes resources removed"
}

# ============================================================================
# Testing Functions
# ============================================================================

test_deployment() {
    log_info "Testing deployment..."

    # Health check
    log_info "Running health check..."
    python3 "$DEPLOY_DIR/client.py" --url http://localhost:8000 --health

    # Test prediction
    log_info "Running test prediction..."
    python3 "$DEPLOY_DIR/client.py" --url http://localhost:8000 --predict

    log_info "Tests passed!"
}

run_load_test() {
    log_info "Running load test..."

    check_command locust || {
        log_error "locust not found. Install with: pip install locust"
        exit 1
    }

    locust -f "$DEPLOY_DIR/load_test.py" \
        --host http://localhost:8000 \
        --users 50 \
        --spawn-rate 5 \
        --run-time 2m \
        --headless

    log_info "Load test complete!"
}

# ============================================================================
# Model Export
# ============================================================================

export_model() {
    local checkpoint_path="$1"

    if [ -z "$checkpoint_path" ]; then
        log_error "Usage: $0 export /path/to/checkpoint.pt"
        exit 1
    fi

    log_info "Exporting model from $checkpoint_path..."

    python3 "$DEPLOY_DIR/export_model.py" \
        --checkpoint "$checkpoint_path" \
        --output-dir "$DEPLOY_DIR/models" \
        --format all \
        --compress \
        --benchmark

    log_info "Model exported to $DEPLOY_DIR/models"
}

# ============================================================================
# Main
# ============================================================================

show_usage() {
    cat << EOF
TFAN Model Server Deployment Script

Usage: $0 <command> [options]

Commands:
  build                   Build Docker image
  push                    Push Docker image to registry
  deploy-local            Deploy locally with Docker Compose
  deploy-k8s              Deploy to Kubernetes
  undeploy-local          Stop local deployment
  undeploy-k8s            Remove Kubernetes deployment
  test                    Test deployment
  load-test               Run load test
  export <checkpoint>     Export model from checkpoint
  logs                    Show logs (local deployment)
  status                  Check deployment status

Environment Variables:
  IMAGE_TAG              Docker image tag (default: latest)
  REGISTRY               Docker registry for push (e.g., gcr.io/project)

Examples:
  # Local deployment
  $0 build
  $0 export /path/to/checkpoint.pt
  $0 deploy-local
  $0 test
  $0 logs

  # Kubernetes deployment
  export REGISTRY=gcr.io/my-project
  $0 build
  $0 push
  $0 deploy-k8s

  # Load testing
  $0 load-test

EOF
}

# Parse command
case "${1:-}" in
    build)
        build_docker
        ;;
    push)
        push_docker
        ;;
    deploy-local)
        deploy_local
        ;;
    deploy-k8s)
        deploy_k8s
        ;;
    undeploy-local)
        stop_local
        ;;
    undeploy-k8s)
        undeploy_k8s
        ;;
    test)
        test_deployment
        ;;
    load-test)
        run_load_test
        ;;
    export)
        export_model "${2:-}"
        ;;
    logs)
        cd "$DEPLOY_DIR"
        docker-compose logs -f tfan-server
        ;;
    status)
        cd "$DEPLOY_DIR"
        docker-compose ps
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

log_info "Done!"
