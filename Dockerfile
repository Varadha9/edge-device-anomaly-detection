# Multi-stage lightweight Dockerfile for Edge Computer Vision API
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install essential runtime dependencies for OpenCV, Git, and DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install with uv for ultra-fast layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r requirements.txt

# Copy application source code and models
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY tests/ tests/

# Expose FastAPI Edge Port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch Edge API Service
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
