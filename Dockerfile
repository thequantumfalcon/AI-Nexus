# AI-Nexus Platform Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] pyjwt mlflow

# Copy application code
COPY core/ ./core/
COPY services/ ./services/
COPY config/ ./config/
COPY proto/ ./proto/
COPY scripts/ ./scripts/
COPY __init__.py .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Expose ports
EXPOSE 8000 8888 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "uvicorn", "services.api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
