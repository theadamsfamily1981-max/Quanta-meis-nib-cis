# ============================================================================
# Multi-Stage Docker Build for TFAN Model Serving
# Optimized for production deployment with minimal image size
# ============================================================================

# ============================================================================
# Stage 1: Builder - Compile dependencies and prepare environment
# ============================================================================
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
COPY deploy/requirements-serve.txt /tmp/requirements-serve.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements-serve.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TFAN_MODEL_PATH="/app/models/tfan_model.pt"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY tfan/ /app/tfan/
COPY deploy/serve.py /app/deploy/serve.py
COPY deploy/healthcheck.sh /app/deploy/healthcheck.sh

# Make healthcheck script executable
RUN chmod +x /app/deploy/healthcheck.sh

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs && \
    chmod 755 /app/models /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 tfan && \
    chown -R tfan:tfan /app

# Switch to non-root user
USER tfan

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/deploy/healthcheck.sh"]

# Run server
CMD ["python", "-m", "uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000"]
