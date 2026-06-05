# syntax=docker/dockerfile:1
# Vocalie-TTS — Production Docker build
# Multi-stage: frontend build → backend deps → runtime

# ─── Stage 1: Build Next.js frontend ───────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --include=optional --no-audit \
    && npm install --no-save \
        lightningcss-linux-arm64-gnu \
        lightningcss-linux-x64-gnu \
        @tailwindcss/oxide-linux-arm64-gnu \
        @tailwindcss/oxide-linux-x64-gnu \
    || true
COPY frontend/ ./
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN npm run build

# ─── Stage 2: Install Python backend deps ───────────────────────────────
FROM python:3.11-slim AS backend-builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# ─── Stage 3: Runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r vocalie && useradd -r -g vocalie vocalie

WORKDIR /app

# Python deps from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Node.js runtime from frontend builder (required for Next.js standalone server)
COPY --from=frontend-builder /usr/local/bin/node /usr/local/bin/node

# Backend source
COPY backend/ ./backend/
COPY backend_install/ ./backend_install/
COPY tts_backends/ ./tts_backends/
COPY presets/ ./presets/

# Shared root shims (still needed by some test imports and backend.shared)
COPY tts_pipeline.py text_tools.py audio_defaults.py output_paths.py refs.py \
     session_manager.py pyproject.toml ./

# Frontend built output (standalone mode)
COPY --from=frontend-builder /app/frontend/.next/standalone ./frontend/
COPY --from=frontend-builder /app/frontend/.next/static ./frontend/.next/static
COPY --from=frontend-builder /app/frontend/public ./frontend/public

# Create data directories
RUN mkdir -p /app/output /app/work /app/.state /app/.assets /app/Ref_audio \
    && chown -R vocalie:vocalie /app

USER vocalie

# --- Environment ---
ENV VOCALIE_TRUST_LOCALHOST=0 \
    VOCALIE_ENABLE_API_DOCS=0 \
    VOCALIE_EXPOSE_SYSTEM_INFO=0 \
    VOCALIE_MAX_UPLOAD_BYTES=26214400 \
    VOCALIE_API_KEY="" \
    VOCALIE_CORS_ORIGINS="" \
    VOCALIE_ALLOWED_HOSTS="127.0.0.1,localhost"

# Backend
EXPOSE 8018
# Frontend
EXPOSE 3018

# Health check on backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8018/v1/health')" || exit 1

# Use our process manager
COPY docker/entrypoint.sh /entrypoint.sh

CMD ["/entrypoint.sh"]