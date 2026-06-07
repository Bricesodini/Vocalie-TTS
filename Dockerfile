# syntax=docker/dockerfile:1
# Vocalie-TTS — Production Docker build
# Multi-stage: frontend build → backend deps → backend venvs (chatterbox, qwen3) → runtime

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

# ─── Stage 3: Build Chatterbox venv (heavy: torch + chatterbox-tts) ─────
FROM python:3.11-slim AS chatterbox-venv-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-chatterbox.txt ./

# Create venv and install all chatterbox deps.
# This is the slowest step of the build (~5-10 min, mostly torch wheel).
ENV PIP_NO_BUILD_ISOLATION=1
RUN python -m venv /opt/venvs/chatterbox
RUN /opt/venvs/chatterbox/bin/pip install --no-cache-dir --quiet --upgrade pip setuptools wheel
RUN /opt/venvs/chatterbox/bin/pip install --no-cache-dir "numpy<1.26,>=1.24"
RUN /opt/venvs/chatterbox/bin/pip install --no-cache-dir -r requirements-chatterbox.txt
# Verify the install is importable before baking into the runtime image
RUN /opt/venvs/chatterbox/bin/python -c "import chatterbox; print('chatterbox OK')"

# ─── Stage 4: Build Qwen3 venv (heavy: qwen-tts + torch) ────────────────
FROM python:3.11-slim AS qwen3-venv-builder

WORKDIR /build

RUN python -m venv /opt/venvs/qwen3
RUN /opt/venvs/qwen3/bin/pip install --no-cache-dir --quiet --upgrade pip setuptools wheel
RUN /opt/venvs/qwen3/bin/pip install --no-cache-dir "qwen-tts==0.0.4" "torch" "torchaudio"
RUN /opt/venvs/qwen3/bin/python -c "import qwen_tts; print('qwen3 OK')"

# ─── Stage 5: Runtime ───────────────────────────────────────────────────
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

# Pre-baked backend venvs (chatterbox, qwen3) — ready to use, no runtime install
COPY --from=chatterbox-venv-builder /opt/venvs/chatterbox /app/.venvs/chatterbox
COPY --from=qwen3-venv-builder /opt/venvs/qwen3 /app/.venvs/qwen3

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

# Fallback runtime venv installer (used only if a pre-baked venv is missing,
# e.g. user added a custom backend at runtime)
COPY docker/install-venvs.sh /app/docker/install-venvs.sh
RUN chmod +x /app/docker/install-venvs.sh

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
    VOCALIE_ALLOWED_HOSTS="127.0.0.1,localhost" \
    VOCALIE_SKIP_VENV_INSTALL=1

# Backend
EXPOSE 8018
# Frontend
EXPOSE 3018

# Health check on backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8018/v1/health')" || exit 1

# Use our process manager
COPY docker/entrypoint.sh /entrypoint.sh

CMD ["/entrypoint.sh"]
