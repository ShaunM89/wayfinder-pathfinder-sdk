# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Install system dependencies for curl_cffi and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash pathfinder

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e "."

USER pathfinder

# Default cache directory for model downloads
ENV PATHFINDER_CACHE_DIR=/home/pathfinder/.cache/pathfinder

ENTRYPOINT ["python", "-m", "pathfinder_sdk"]
CMD ["--help"]

# --- Playwright variant ---
FROM base AS playwright

USER root
RUN pip install --no-cache-dir playwright && \
    playwright install chromium
USER pathfinder
