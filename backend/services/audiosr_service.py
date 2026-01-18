from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import soundfile as sf

import backend.config as backend_config
from output_paths import ensure_unique_path, sanitize_filename


LOGGER = logging.getLogger("chatterbox_api.audiosr")

_AUDIOSR_IMPORT_ERROR: Optional[Exception] = None


class FeatureDisabledError(RuntimeError):
    pass


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def audiosr_python_path() -> Optional[Path]:
    venv_root = backend_config.BASE_DIR / ".venvs" / "audiosr"
    if (venv_root / "bin" / "python").exists():
        return venv_root / "bin" / "python"
    if (venv_root / "Scripts" / "python.exe").exists():
        return venv_root / "Scripts" / "python.exe"
    return None


def _audiosr_import_ok() -> bool:
    global _AUDIOSR_IMPORT_ERROR
    python_path = audiosr_python_path()
    if not python_path:
        _AUDIOSR_IMPORT_ERROR = FileNotFoundError("audiosr_python_missing")
        return False
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import audiosr; print('ok')"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        _AUDIOSR_IMPORT_ERROR = exc
        return False
    if result.returncode == 0 and "ok" in result.stdout:
        _AUDIOSR_IMPORT_ERROR = None
        return True
    _AUDIOSR_IMPORT_ERROR = RuntimeError(result.stderr.strip() or "audiosr_import_failed")
    return False


def audiosr_is_available() -> bool:
    if not backend_config.VOCALIE_ENABLE_AUDIOSR:
        return False
    return _audiosr_import_ok()


def audiosr_available_details() -> dict:
    python_path = audiosr_python_path()
    available = audiosr_is_available()
    return {
        "enabled": backend_config.VOCALIE_ENABLE_AUDIOSR,
        "available": available,
        "python": str(python_path) if python_path else None,
        "error": str(_AUDIOSR_IMPORT_ERROR) if _AUDIOSR_IMPORT_ERROR else None,
    }


def log_audiosr_status() -> None:
    details = audiosr_available_details()
    LOGGER.info(
        "AudioSR enabled=%s available=%s python=%s error=%s",
        details["enabled"],
        details["available"],
        details["python"],
        details["error"],
    )


def build_output_paths(input_name: str) -> tuple[Path, Path]:
    date_folder = _utc_now().strftime("%Y-%m-%d")
    output_dir = backend_config.OUTPUT_DIR / date_folder / "audiosr"
    sanitized = sanitize_filename(input_name) or "audio"
    filename = f"{sanitized}.audiosr.wav"
    output_path = ensure_unique_path(output_dir, filename)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    return output_path, meta_path


def write_sidecar(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_audiosr(input_path: str, output_path: str, params: dict) -> dict:
    if not backend_config.VOCALIE_ENABLE_AUDIOSR:
        raise FeatureDisabledError("audiosr_disabled")
    python_path = audiosr_python_path()
    if not python_path or not _audiosr_import_ok():
        raise FeatureDisabledError("audiosr_not_installed")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_path),
        "-m",
        "backend.workers.audiosr_runner",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--ddim_steps",
        str(params["ddim_steps"]),
        "--guidance_scale",
        str(params["guidance_scale"]),
        "--seed",
        str(params["seed"]),
        "--chunk_size",
        str(params["chunk_size"]),
        "--overlap",
        str(params["overlap"]),
        "--multiband_ensemble",
        "1" if params["multiband_ensemble"] else "0",
        "--input_cutoff",
        str(params["input_cutoff"]),
    ]

    env = os.environ.copy()
    if "HF_HOME" in os.environ:
        env["HF_HOME"] = os.environ["HF_HOME"]
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        env["HUGGINGFACE_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
    env.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=backend_config.VOCALIE_AUDIOSR_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        LOGGER.error("AudioSR runner timed out after %ss", backend_config.VOCALIE_AUDIOSR_TIMEOUT_S)
        raise RuntimeError("audiosr_timeout") from exc
    except Exception as exc:
        LOGGER.error("AudioSR subprocess failed", exc_info=exc)
        raise RuntimeError("audiosr_subprocess_failed") from exc

    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            LOGGER.error("AudioSR runner stdout: %s", stdout)
        if stderr:
            LOGGER.error("AudioSR runner stderr: %s", stderr)
        message = stderr or stdout or "audiosr_runner_failed"
        raise RuntimeError(f"audiosr_runner_failed: {message}")

    info = sf.info(str(output_path))
    duration_s = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "output_path": str(output_path),
        "sample_rate": int(info.samplerate) if info.samplerate else 48000,
        "duration_s": duration_s,
        "engine": "audiosr",
    }
