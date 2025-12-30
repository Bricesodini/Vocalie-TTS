"""Installer for per-backend virtual environments."""

from __future__ import annotations

import subprocess
import time
from typing import List, Tuple

from .manifests import get_manifest
from .paths import pip_path, python_path, venv_dir


def _stamp(message: str) -> str:
    return f"[{time.strftime('%H:%M:%S')}] {message}"


def create_venv(engine_id: str, python: str = "python3.11") -> None:
    target = venv_dir(engine_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, "-m", "venv", str(target)], check=True)


def pip_install(engine_id: str, packages: List[str]) -> None:
    pip = pip_path(engine_id)
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip), "install", *packages], check=True)


def run_install(engine_id: str) -> Tuple[bool, List[str]]:
    logs: List[str] = []
    manifest = get_manifest(engine_id)
    if manifest is None:
        return False, [f"Manifest introuvable: {engine_id}"]
    try:
        logs.append(_stamp("Création du venv..."))
        create_venv(engine_id, python=manifest.python)
        logs.append(_stamp("Installation des dépendances..."))
        pip_install(engine_id, manifest.pip_packages)
        for check in manifest.post_install_checks:
            logs.append(_stamp(f"Check: {' '.join(check)}"))
            subprocess.run([str(python_path(engine_id)), *check], check=True)
        logs.append(_stamp("Installation terminée."))
        return True, logs
    except subprocess.CalledProcessError as exc:
        logs.append(_stamp(f"Erreur install: {exc}"))
        return False, logs
