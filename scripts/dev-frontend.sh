#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v node >/dev/null 2>&1; then
  echo "node not found. Install Node.js >= 20 and try again." >&2
  exit 1
fi

cd "$ROOT_DIR/frontend"

OS_NAME="$(uname -s)"

install_frontend_deps() {
  echo "Installing frontend dependencies (npm ci --include=optional)"
  npm ci --include=optional --prefer-offline --no-audit --progress=false
}

if [[ ! -d node_modules ]]; then
  if [[ "$OS_NAME" == "Darwin" ]]; then
    cat <<'EOF'
On macOS the frontend lockfile targets Linux binaries (strict CI policy).
Install dependencies manually before running this script:

  cd frontend
  npm install --include=optional --no-audit --progress=false

Then rerun \`scripts/dev.sh\`. Don't forget to discard any lockfile changes:

  git checkout -- frontend/package-lock.json

EOF
    exit 1
  fi
  install_frontend_deps
fi

npm run dev
