# Multi-stage Dockerfile for AI-Nexus with GPU support
# Based on NVIDIA CUDA runtime for optimal GPU performance

# Stage 1: Build stage
FROM nvidia/cuda:13.0.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] pyjwt mlflow

# Install PyTorch with CUDA 13.0 support
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Copy application code
COPY core/ ./core/
COPY services/ ./services/
COPY config/ ./config/
COPY proto/ ./proto/
COPY scripts/ ./scripts/
COPY __init__.py .

# Stage 2: Runtime stage
FROM nvidia/cuda:13.0.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash ainexus && \
    mkdir -p /app /data /models /logs && \
    chown -R ainexus:ainexus /app /data /models /logs

WORKDIR /app

# Copy from builder
COPY --from=builder --chown=ainexus:ainexus /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder --chown=ainexus:ainexus /usr/local/bin /usr/local/bin
COPY --from=builder --chown=ainexus:ainexus /app /app

USER ainexus

# Expose ports
EXPOSE 8000 8888 9090 29500

# Health check - verify CUDA availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "services.api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
