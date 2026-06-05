"""Backend installation manifests."""

from __future__ import annotations

import json
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

    "qwen3": BackendManifest(
        engine_id="qwen3",
        python="python3.11",
        pip_packages=[
            "qwen-tts==0.0.4",
            "torch",
            "torchaudio",
        ],
        system_hints=["ffmpeg (optionnel)"],
        import_probes=["qwen_tts"],
        post_install_checks=[["-c", "import qwen_tts; print('OK')"]],
    ),
    "cosyvoice": BackendManifest(
        engine_id="cosyvoice",
        python="python3.10",
        pip_packages=[
            "torch",
            "torchaudio",
            "soundfile",
            "conformer",
            "diffusers",
            "inflect",
            "pydantic",
            "numpy",
            "huggingface_hub",
        ],
        system_hints=["NVIDIA GPU ≥ 8GB VRAM", "ffmpeg (recommended)", "sox + libsox-dev (Ubuntu)"],
        import_probes=["cosyvoice"],
        post_install_checks=[["-c", "from cosyvoice.cli.cosyvoice import AutoModel; print('OK')"]],
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
