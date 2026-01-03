#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to generate a Linux lockfile." >&2
  echo "Install Docker Desktop (macOS) or Docker Engine (Linux), then re-run." >&2
  exit 1
fi

if [ ! -d "${FRONTEND_DIR}" ]; then
  echo "frontend directory not found at: ${FRONTEND_DIR}" >&2
  exit 1
fi

echo "Generating frontend/package-lock.json using a Linux (Docker) environment..."
docker run --rm \
  -v "${FRONTEND_DIR}:/app" \
  -w /app \
  node:20-bookworm \
  bash -lc "rm -rf node_modules package-lock.json && npm install --include=optional --package-lock-only --no-audit --progress=false"

echo "Validating lockfile on Linux (Docker): npm ci && npm run build"
docker run --rm \
  -v "${FRONTEND_DIR}:/app" \
  -w /app \
  node:20-bookworm \
  bash -lc "npm ci --include=optional --no-audit --progress=false && npm run build"

echo "Done. Commit frontend/package-lock.json."
