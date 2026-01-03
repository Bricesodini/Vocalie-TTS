"""Backend installation manifests."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class BackendManifest:
    engine_id: str
    python: str
    pip_packages: List[str]
    system_hints: List[str]
    import_probes: List[str]
    post_install_checks: List[List[str]]


MANIFESTS: Dict[str, BackendManifest] = {
    "chatterbox": BackendManifest(
        engine_id="chatterbox",
        python="python3.11",
        pip_packages=["-r", "requirements-chatterbox.txt"],
        system_hints=["ffmpeg (recommandé)"],
        import_probes=["chatterbox"],
        post_install_checks=[["-c", "import chatterbox; print('OK')"]],
    ),
    "xtts": BackendManifest(
        engine_id="xtts",
        python=sys.executable,
        pip_packages=[
            "TTS==0.22.0",
            "torch==2.2.2",
            "torchaudio==2.2.2",
            "transformers==4.39.3",
            "huggingface_hub<1.0",
            "sentencepiece",
            "soundfile",
            "scipy",
            "numpy<2.4",
        ],
        system_hints=["ffmpeg (recommandé)"],
        import_probes=["TTS"],
        post_install_checks=[["-c", "import TTS; print('OK')"]],
    ),
    "piper": BackendManifest(
        engine_id="piper",
        python="python3.11",
        pip_packages=["piper-tts"],
        system_hints=["espeak-ng (parfois requis selon les voix)", "ffmpeg (optionnel)"],
        import_probes=["piper", "piper_tts"],
        post_install_checks=[["-c", "import piper; print('OK')"]],
    ),
    "bark": BackendManifest(
        engine_id="bark",
        python="python3.11",
        pip_packages=["-r", "requirements-bark.txt"],
        system_hints=["ffmpeg (recommandé)"],
        import_probes=["bark"],
        post_install_checks=[["-c", "import bark; print('OK')"]],
    ),
}


def _overrides_path() -> Path:
    return Path(__file__).resolve().parent / "local_overrides.json"


def load_overrides() -> Dict[str, List[str]]:
    path = _overrides_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def get_manifest(engine_id: str) -> BackendManifest | None:
    manifest = MANIFESTS.get(engine_id)
    if manifest is None:
        return None
    overrides = load_overrides()
    packages = overrides.get(engine_id)
    if packages:
        manifest = BackendManifest(
            engine_id=manifest.engine_id,
            python=manifest.python,
            pip_packages=list(packages),
            system_hints=manifest.system_hints,
            import_probes=manifest.import_probes,
            post_install_checks=manifest.post_install_checks,
        )
    return manifest
