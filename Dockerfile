# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=2 \
    PORT=8400

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast python installs if present; otherwise fallback to pip
COPY pyproject.toml uv.lock ./
# Prefer uv.lock if present for reproducible installs; fallback to pip install from pyproject
RUN pip install --upgrade pip uv && \
    if [ -f uv.lock ]; then \
      uv pip sync --system uv.lock; \
    else \
      uv pip install --system -r pyproject.toml || pip install -r requirements.txt; \
    fi

# Copy source
COPY . .

# Create a non-root user
RUN useradd -m app && chown -R app:app /app
USER app

# Default to SQLite inside container unless DATABASE_URL provided
ENV DATA_DIR=/app/data

EXPOSE 8400

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8400"]
