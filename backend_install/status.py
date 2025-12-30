"""Backend installation status helpers."""

from __future__ import annotations

import subprocess
from typing import List, Tuple

from .manifests import get_manifest
from .paths import python_path, venv_dir


def venv_exists(engine_id: str) -> bool:
    return venv_dir(engine_id).exists()


def import_ok(engine_id: str, import_probes: List[str]) -> Tuple[bool, str]:
    py = python_path(engine_id)
    if not py.exists():
        return False, "python introuvable dans le venv"
    for probe in import_probes:
        result = subprocess.run(
            [str(py), "-c", f"import {probe}; print('OK')"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, "OK"
    return False, "import probe failed"


def backend_status(engine_id: str) -> dict:
    if engine_id == "chatterbox":
        return {"installed": True, "reason": "venv principal"}
    manifest = get_manifest(engine_id)
    if manifest is None:
        return {"installed": False, "reason": "manifest introuvable"}
    if not venv_exists(engine_id):
        return {"installed": False, "reason": "venv manquante"}
    ok, reason = import_ok(engine_id, manifest.import_probes)
    return {"installed": ok, "reason": reason}
