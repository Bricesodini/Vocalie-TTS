"""Installer for per-backend virtual environments."""

from __future__ import annotations

import os
import subprocess
import time
from typing import List, Tuple

from .manifests import get_manifest
from .paths import pip_path, python_path, venv_dir, ROOT


def _stamp(message: str) -> str:
    return f"[{time.strftime('%H:%M:%S')}] {message}"


def create_venv(engine_id: str, python: str = "python3.11") -> None:
    target = venv_dir(engine_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, "-m", "venv", str(target)], check=True)


def pip_install(engine_id: str, packages: List[str], env: dict | None = None) -> None:
    pip = pip_path(engine_id)
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, env=env)
    subprocess.run([str(pip), "install", *packages], check=True, env=env)


def _run_xtts_prefetch(engine_id: str) -> Tuple[bool, str]:
    py = python_path(engine_id)
    if not py.exists():
        return False, "python introuvable dans le venv"
    runner = ROOT / "tts_backends" / "xtts_prefetch.py"
    if not runner.exists():
        return False, "runner xtts_prefetch.py introuvable"
    env = dict(**os.environ)
    env["TTS_HOME"] = str(ROOT / ".assets" / "xtts")
    env["COQUI_TOS_AGREED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    result = subprocess.run(
        [str(py), str(runner)],
        capture_output=True,
        text=True,
        env=env,
    )
    output = "\n".join(
        [line for line in (result.stdout or "").splitlines() if line.strip()]
        + [line for line in (result.stderr or "").splitlines() if line.strip()]
    ).strip()
    if result.returncode != 0:
        return False, output or "prefetch failed"
    return True, output or "prefetch ok"


def _run_bark_prefetch(engine_id: str) -> Tuple[bool, str]:
    py = python_path(engine_id)
    if not py.exists():
        return False, "python introuvable dans le venv"
    runner = ROOT / "tts_backends" / "bark_prefetch.py"
    if not runner.exists():
        return False, "runner bark_prefetch.py introuvable"
    assets_dir = ROOT / ".assets" / "bark"
    env = dict(**os.environ)
    env["XDG_CACHE_HOME"] = str(assets_dir)
    env["HF_HOME"] = str(assets_dir / ".hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(assets_dir / ".hf" / "hub")
    env["SUNO_ENABLE_MPS"] = "False"
    if env.get("VOCALIE_BARK_SMALL_MODELS") in {"1", "true", "True", "yes", "YES"}:
        env["SUNO_USE_SMALL_MODELS"] = "True"
    result = subprocess.run(
        [str(py), str(runner)],
        capture_output=True,
        text=True,
        env=env,
    )
    output = "\n".join(
        [line for line in (result.stdout or "").splitlines() if line.strip()]
        + [line for line in (result.stderr or "").splitlines() if line.strip()]
    ).strip()
    if result.returncode != 0:
        return False, output or "prefetch failed"
    return True, output or "prefetch ok"


def _run_qwen3_prefetch(engine_id: str) -> Tuple[bool, str]:
    py = python_path(engine_id)
    if not py.exists():
        return False, "python introuvable dans le venv"
    runner = ROOT / "tts_backends" / "qwen3_prefetch.py"
    if not runner.exists():
        return False, "runner qwen3_prefetch.py introuvable"
    assets_dir = ROOT / ".assets" / "qwen3"
    env = dict(**os.environ)
    env["VOCALIE_QWEN3_ASSETS_DIR"] = str(assets_dir)
    result = subprocess.run(
        [str(py), str(runner)],
        capture_output=True,
        text=True,
        env=env,
    )
    output = "\n".join(
        [line for line in (result.stdout or "").splitlines() if line.strip()]
        + [line for line in (result.stderr or "").splitlines() if line.strip()]
    ).strip()
    if result.returncode != 0:
        return False, output or "prefetch failed"
    return True, output or "prefetch ok"


def run_install(engine_id: str) -> Tuple[bool, List[str]]:
    logs: List[str] = []
    manifest = get_manifest(engine_id)
    if manifest is None:
        return False, [f"Manifest introuvable: {engine_id}"]
    try:
        logs.append(_stamp("Création du venv..."))
        create_venv(engine_id, python=manifest.python)
        logs.append(_stamp("Installation des dépendances..."))
        if engine_id == "chatterbox":
            env = dict(**os.environ)
            env["PIP_NO_BUILD_ISOLATION"] = "1"
            pip = pip_path(engine_id)
            requirements_path = str(ROOT / "requirements-chatterbox.txt")
            subprocess.run([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, env=env)
            subprocess.run([str(pip), "install", "numpy<1.26,>=1.24"], check=True, env=env)
            subprocess.run([str(pip), "install", "-r", requirements_path], check=True, env=env)
        else:
            pip_install(engine_id, manifest.pip_packages)
        for check in manifest.post_install_checks:
            logs.append(_stamp(f"Check: {' '.join(check)}"))
            subprocess.run([str(python_path(engine_id)), *check], check=True)
        if engine_id == "xtts":
            logs.append(_stamp("Téléchargement des poids XTTS..."))
            ok, output = _run_xtts_prefetch(engine_id)
            if ok:
                logs.append(_stamp("Poids XTTS OK (cache)."))
            else:
                logs.append(_stamp(f"⚠️ Préchargement XTTS échoué: {output}"))
        if engine_id == "bark":
            logs.append(_stamp("Téléchargement des poids Bark..."))
            ok, output = _run_bark_prefetch(engine_id)
            if ok:
                logs.append(_stamp("Poids Bark OK (cache)."))
            else:
                logs.append(_stamp(f"⚠️ Préchargement Bark échoué: {output}"))
        if engine_id == "qwen3":
            logs.append(_stamp("Téléchargement des poids Qwen3..."))
            ok, output = _run_qwen3_prefetch(engine_id)
            if ok:
                logs.append(_stamp("Poids Qwen3 OK (cache)."))
            else:
                logs.append(_stamp(f"⚠️ Préchargement Qwen3 échoué: {output}"))
        logs.append(_stamp("Installation terminée."))
        return True, logs
    except subprocess.CalledProcessError as exc:
        logs.append(_stamp(f"Erreur install: {exc}"))
        return False, logs
