#!/usr/bin/env bash
# Vocalie-TTS — Start via Docker Compose
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Building Vocalie-TTS Docker image..."
docker compose build

echo ""
echo "Starting Vocalie-TTS..."
docker compose up -d

echo ""
echo "Waiting for services to be ready..."
for i in {1..30}; do
  if curl -sf -m 2 "http://localhost:3018" >/dev/null 2>&1; then
    echo "✓ Frontend ready on http://localhost:3018"
    break
  fi
  sleep 2
done

echo ""
echo "Vocalie-TTS is running:"
echo "  Frontend: http://localhost:3018"
echo "  Backend:  http://127.0.0.1:8018 (internal to container)"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f     # Follow logs"
echo "  docker compose down        # Stop and remove containers"
echo "  docker compose restart     # Restart"