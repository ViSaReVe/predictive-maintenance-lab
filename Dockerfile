# Multi-stage build:
#   stage 1 (builder) — install deps into a venv
#   stage 2 (runtime) — copy venv + source, run uvicorn

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY api/ api/
COPY pipelines/ pipelines/
COPY configs/ configs/

# Expose API port
EXPOSE 8000

# Start inference API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
