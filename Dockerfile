# PENIN Evolution System - Production Docker Image
# Multi-stage build for optimized production deployment

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash penin

# Stage 2: Dependencies
FROM base as dependencies

# Set work directory
WORKDIR /tmp

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Set work directory
WORKDIR /app

# Copy application code
COPY . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/backups && \
    chown -R penin:penin /app

# Switch to non-root user
USER penin

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "penin.api.server"]

# Stage 4: Development image
FROM application as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install -e ".[dev,all]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to penin user
USER penin

# Development command
CMD ["python", "-m", "penin.api.server", "--reload"]

# Stage 5: Production image (default)
FROM application as production

# Production optimizations
ENV PYTHONOPTIMIZE=1

# Production command with gunicorn
CMD ["gunicorn", "penin.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]