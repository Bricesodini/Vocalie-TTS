"""Backend installation status helpers."""

from __future__ import annotations

import subprocess
from typing import List, Tuple

from .manifests import get_manifest
from .paths import ROOT, python_path, venv_dir

XTTS_ASSETS_DIR = ROOT / ".assets" / "xtts"


def _xtts_model_downloaded() -> bool:
    candidates = [
        XTTS_ASSETS_DIR / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2",
        XTTS_ASSETS_DIR / "tts" / "tts_models" / "multilingual" / "multi-dataset" / "xtts_v2",
        XTTS_ASSETS_DIR / "tts_models--multilingual--multi-dataset--xtts_v2",
        XTTS_ASSETS_DIR / "tts_models" / "multilingual" / "multi-dataset" / "xtts_v2",
    ]
    return any(path.exists() for path in candidates)


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
    if engine_id == "xtts" and ok:
        model_ok = _xtts_model_downloaded()
        if not model_ok:
            return {
                "installed": True,
                "reason": "poids manquants",
                "model_downloaded": False,
            }
        return {"installed": True, "reason": "OK", "model_downloaded": True}
    return {"installed": ok, "reason": reason}
