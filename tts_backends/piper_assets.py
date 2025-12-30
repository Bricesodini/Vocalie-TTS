"""Piper voice assets helper."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from tts_backends.base import VoiceInfo
from backend_install.paths import python_path


DEFAULT_VOICE_ID = "fr_FR-upmc-medium"
DEFAULT_VOICE_URL_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium"
)
DEFAULT_VOICE_ONNX_URL = f"{DEFAULT_VOICE_URL_BASE}/fr_FR-upmc-medium.onnx"
DEFAULT_VOICE_JSON_URL = f"{DEFAULT_VOICE_URL_BASE}/fr_FR-upmc-medium.onnx.json"
_PIPER_CAPS_CACHE: dict[str, bool] = {}
_PIPER_CAPS_LOADED = False


def get_piper_voices_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".assets" / "piper" / "voices"


def _caps_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".state" / "piper_voice_caps.json"


def _load_caps_cache() -> None:
    global _PIPER_CAPS_LOADED
    if _PIPER_CAPS_LOADED:
        return
    _PIPER_CAPS_LOADED = True
    path = _caps_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, bool):
                _PIPER_CAPS_CACHE[key] = value


def _save_caps_cache() -> None:
    path = _caps_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_PIPER_CAPS_CACHE, indent=2), encoding="utf-8")
    except OSError:
        return


def _probe_length_scale(voice_id: str) -> bool:
    info = get_voice_info(voice_id)
    if info is None:
        return False
    py = python_path("piper")
    if not py.exists():
        return False
    runner = Path(__file__).resolve().parent / "piper_runner.py"
    out_path = _caps_path().with_name(f"probe_{voice_id}.wav")
    cmd = [
        str(py),
        str(runner),
        "--text",
        "test",
        "--out_wav",
        str(out_path),
        "--voice",
        info["voice_id"],
        "--model_dir",
        info["model_dir"],
        "--length_scale",
        "1.1",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        out_path.unlink(missing_ok=True)


def piper_voice_supports_length_scale(voice_id: str) -> bool:
    if not voice_id:
        return False
    _load_caps_cache()
    if voice_id in _PIPER_CAPS_CACHE:
        return _PIPER_CAPS_CACHE[voice_id]
    supports = _probe_length_scale(voice_id)
    _PIPER_CAPS_CACHE[voice_id] = supports
    _save_caps_cache()
    return supports

@dataclass(frozen=True)
class InstallResult:
    ok: bool
    voice_id: str
    message: str


def list_installed_voices() -> list[str]:
    root = get_piper_voices_dir()
    if not root.exists():
        return []
    voices = set()
    for file in root.rglob("*.onnx"):
        json_path = file.with_suffix(".onnx.json")
        if not json_path.exists():
            continue
        rel = file.relative_to(root).as_posix()
        if not rel.endswith(".onnx"):
            continue
        voice_id = rel[: -len(".onnx")]
        voices.add(voice_id)
    return sorted(voices)


def _voice_lang_codes(voice_id: str) -> list[str]:
    if "fr_FR" in voice_id:
        return ["fr-FR"]
    if "en_US" in voice_id:
        return ["en-US"]
    if "en_GB" in voice_id:
        return ["en-GB"]
    if "es_ES" in voice_id:
        return ["es-ES"]
    if "de_DE" in voice_id:
        return ["de-DE"]
    if "it_IT" in voice_id:
        return ["it-IT"]
    if "pt_PT" in voice_id:
        return ["pt-PT"]
    if "nl_NL" in voice_id:
        return ["nl-NL"]
    return []


def list_piper_voices() -> list[VoiceInfo]:
    voices = []
    for voice_id in list_installed_voices():
        lang_codes = _voice_lang_codes(voice_id)
        label = voice_id
        if lang_codes:
            label = f"{lang_codes[0]} — {voice_id}"
        voices.append(
            VoiceInfo(
                id=voice_id,
                label=label,
                lang_codes=lang_codes or None,
                installed=True,
            )
        )
    return voices


def get_voice_info(voice_id: str) -> dict | None:
    root = get_piper_voices_dir()
    model_path = root / f"{voice_id}.onnx"
    config_path = root / f"{voice_id}.onnx.json"
    if model_path.exists() and config_path.exists():
        return {
            "voice_id": voice_id,
            "model_dir": str(root),
            "model_path": str(model_path),
            "config_path": str(config_path),
        }
    return None


def is_voice_installed(voice_id: str) -> bool:
    root = get_piper_voices_dir()
    model_path = root / f"{voice_id}.onnx"
    config_path = root / f"{voice_id}.onnx.json"
    return model_path.exists() and config_path.exists()


def install_voice_from_catalog(voice_id: str, onnx_url: str, json_url: str) -> InstallResult:
    root = get_piper_voices_dir()
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / f"{voice_id}.onnx"
    config_path = root / f"{voice_id}.onnx.json"
    model_tmp = model_path.with_suffix(".onnx.tmp")
    config_tmp = config_path.with_suffix(".onnx.json.tmp")
    try:
        urllib.request.urlretrieve(onnx_url, model_tmp)
        urllib.request.urlretrieve(json_url, config_tmp)
        if model_tmp.stat().st_size <= 0:
            return InstallResult(False, voice_id, "download_failed: onnx empty")
        if config_tmp.stat().st_size <= 0:
            return InstallResult(False, voice_id, "download_failed: json empty")
        with config_tmp.open("r", encoding="utf-8") as fh:
            json.load(fh)
        model_tmp.replace(model_path)
        config_tmp.replace(config_path)
        return InstallResult(True, voice_id, "OK")
    except Exception as exc:
        return InstallResult(False, voice_id, f"download_failed: {exc}")
    finally:
        if model_tmp.exists() and not model_path.exists():
            model_tmp.unlink(missing_ok=True)
        if config_tmp.exists() and not config_path.exists():
            config_tmp.unlink(missing_ok=True)


def ensure_default_voice_installed() -> InstallResult:
    if is_voice_installed(DEFAULT_VOICE_ID):
        return InstallResult(True, DEFAULT_VOICE_ID, "OK")
    return install_voice_from_catalog(
        DEFAULT_VOICE_ID,
        DEFAULT_VOICE_ONNX_URL,
        DEFAULT_VOICE_JSON_URL,
    )


def ensure_default_voice(download: bool = True) -> Tuple[bool, dict, str]:
    if is_voice_installed(DEFAULT_VOICE_ID):
        info = get_voice_info(DEFAULT_VOICE_ID)
        return True, info or {}, "OK"
    if not download:
        return False, {}, "aucune voix disponible"
    result = ensure_default_voice_installed()
    if not result.ok:
        return False, {}, result.message
    info = get_voice_info(DEFAULT_VOICE_ID)
    return True, info or {}, "OK"
