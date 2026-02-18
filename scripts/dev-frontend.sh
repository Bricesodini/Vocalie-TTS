#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v node >/dev/null 2>&1; then
  echo "node not found. Install Node.js >= 20 and try again." >&2
  exit 1
fi

cd "$ROOT_DIR/frontend"

if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies (npm ci)"
  npm ci
fi

if [[ -z "${NEXT_PUBLIC_API_BASE:-}" ]]; then
  export NEXT_PUBLIC_API_BASE="http://127.0.0.1:8018"
fi

export PORT="${PORT:-3018}"
npm run dev
