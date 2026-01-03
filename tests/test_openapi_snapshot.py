from __future__ import annotations

import json
from pathlib import Path

import backend.app as backend_app


def test_openapi_snapshot():
    snapshot_path = Path(__file__).resolve().parents[1] / "openapi.json"
    assert snapshot_path.exists(), "openapi.json snapshot missing (run scripts/update-openapi.sh)"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    current = backend_app.app.openapi()
    assert current == snapshot
