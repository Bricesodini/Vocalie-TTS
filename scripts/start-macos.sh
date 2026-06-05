#!/usr/bin/env bash
# Vocalie-TTS — Launch script for Keyboard Maestro
# Starts backend first, waits for health, then starts frontend, opens browser.
set -euo pipefail

cd "/Users/bricesodini/01_ai-stack/Chatterbox" || exit 1

# --- Node via nvm ---
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 20 >/dev/null

# --- Ports ---
export BACKEND_HOST="127.0.0.1"
export BACKEND_PORT="8018"
export FRONTEND_PORT="3018"
export PORT="$FRONTEND_PORT"

# --- Dev defaults ---
export VOCALIE_TRUST_LOCALHOST=1

# Auto-detect LAN IP for CORS
if [[ -z "${VOCALIE_CORS_ORIGINS:-}" ]]; then
  CORS_ORIGINS="http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}"
  LAN_IP=""
  if command -v ipconfig >/dev/null 2>&1; then
    IFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' | head -n1)"
    if [[ -n "$IFACE" ]]; then
      LAN_IP="$(ipconfig getifaddr "$IFACE" 2>/dev/null || true)"
    fi
    for IF in en0 en1 en2; do
      [[ -z "$LAN_IP" ]] && LAN_IP="$(ipconfig getifaddr "$IF" 2>/dev/null || true)"
    done
  fi
  if [[ -z "$LAN_IP" ]] && command -v ifconfig >/dev/null 2>&1; then
    LAN_IP="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -n1)"
  fi
  if [[ -n "$LAN_IP" ]]; then
    CORS_ORIGINS="${CORS_ORIGINS},http://${LAN_IP}:${FRONTEND_PORT}"
  fi
  export VOCALIE_CORS_ORIGINS="$CORS_ORIGINS"
fi

# IMPORTANT: Do NOT set NEXT_PUBLIC_API_BASE.
# The frontend uses the Next.js rewrite proxy (/v1/* → backend:8018).
unset NEXT_PUBLIC_API_BASE 2>/dev/null || true

# --- Stop existing processes ---
echo "Stopping existing processes..."
./scripts/stop.sh >/dev/null 2>&1 || true
sleep 1

# ── Step 1: Install AudioSR venv if needed (can take several minutes on first run) ──
if [[ ! -d ".venvs/audiosr" ]]; then
  echo "Installing AudioSR dependencies (first run, this may take a few minutes)..."
  ./scripts/install-audiosr-venv.sh
fi

# ── Step 2: Install Python venv + deps if needed ──
if [[ ! -d ".venv" ]]; then
  echo "Creating Python venv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip -r requirements.txt
fi

# ── Step 3: Start backend ──
mkdir -p .run
echo "Starting backend..."
./scripts/dev-backend.sh &>/tmp/vocalie-backend.log &
BACKEND_PID=$!
echo "$BACKEND_PID" > .run/backend.pid

# Wait for backend (long timeout — first run installs deps)
echo -n "Waiting for backend (port ${BACKEND_PORT})... "
BACKEND_READY=false
for i in {1..180}; do
  if curl -sf -m 5 "http://127.0.0.1:${BACKEND_PORT}/v1/health" >/dev/null 2>&1; then
    echo " ✓ (${i}s)"
    BACKEND_READY=true
    break
  fi
  echo -n "."
  sleep 1
done

if [[ "$BACKEND_READY" != "true" ]]; then
  echo " FAILED"
  echo "Backend did not start. Last 30 lines of /tmp/vocalie-backend.log:"
  tail -30 /tmp/vocalie-backend.log 2>/dev/null || true
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi

# Show engine availability (can be slow on first call)
echo -n "  Probing engines..."
ENGINE_COUNT=$(curl -s -m 30 "http://127.0.0.1:${BACKEND_PORT}/v1/tts/engines" 2>/dev/null \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("engines",[])))' 2>/dev/null || echo '?')
echo " ${ENGINE_COUNT} TTS engines available"

# ── Step 4: Start frontend (backend is ready) ──
echo "Starting frontend..."
cd frontend
PORT=$FRONTEND_PORT npm run dev &>/tmp/vocalie-frontend.log &
FRONTEND_PID=$!
cd ..
echo "$FRONTEND_PID" > .run/frontend.pid

echo -n "Waiting for frontend (port ${FRONTEND_PORT})... "
FRONTEND_READY=false
for i in {1..60}; do
  if curl -sf -m 5 "http://127.0.0.1:${FRONTEND_PORT}" >/dev/null 2>&1; then
    echo " ✓"
    FRONTEND_READY=true
    break
  fi
  echo -n "."
  sleep 1
done

if [[ "$FRONTEND_READY" != "true" ]]; then
  echo " FAILED"
  echo "Frontend did not start. Last 20 lines of /tmp/vocalie-frontend.log:"
  tail -20 /tmp/vocalie-frontend.log 2>/dev/null || true
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 1
fi

# ── Open browser ──
echo ""
echo "✓ Vocalie-TTS is running:"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT}/v1/health"
echo "  Logs:     /tmp/vocalie-backend.log, /tmp/vocalie-frontend.log"
echo ""
open "http://127.0.0.1:${FRONTEND_PORT}"