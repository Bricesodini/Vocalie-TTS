#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT_DIR/openapi.json"

python - <<'PY'
import json
import backend.app as backend_app

schema = backend_app.app.openapi()
with open("openapi.json", "w", encoding="utf-8") as fh:
    json.dump(schema, fh, indent=2, ensure_ascii=True)
    fh.write("\n")
PY

echo "Wrote $OUTPUT"
