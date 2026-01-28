#!/usr/bin/env bash
set -euo pipefail

# Keyboard Maestro can run with a minimal PATH; set a reliable one.
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

./scripts/stop.sh || true

# Kill by port (handle multiple PIDs safely).
for port in 3100 8100; do
  for pid in $(/usr/sbin/lsof -ti tcp:"$port" 2>/dev/null || true); do
    kill -9 "$pid" || true
  done
done

# Kill any leftover dev processes (safe targets).
/usr/bin/pkill -f "uvicorn backend.app:app" || true
/usr/bin/pkill -f "next dev" || true
/usr/bin/pkill -f "next-server" || true
