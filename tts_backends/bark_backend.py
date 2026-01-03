"""Bark backend wrapper (optional dependency)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf

from backend.config import VOCALIE_BARK_SMALL_MODELS, VOCALIE_BARK_TIMEOUT_S
from backend_install.paths import python_path
from backend_install.status import backend_status

from .base import BackendUnavailableError, ParamSpec, TTSBackend


def _probe_bark_presets(timeout_s: float = 2.0) -> list[str]:
    py = python_path("bark")
    if not py.exists():
        return []
    script = r"""
import json
from pathlib import Path
try:
    import bark
except Exception:
    print("[]")
    raise SystemExit(0)
root = Path(bark.__file__).resolve().parent
prompts = root / "assets" / "prompts"
items = []
if prompts.exists():
    for path in prompts.rglob("*.npz"):
        try:
            rel = path.relative_to(prompts).with_suffix("")
        except Exception:
            continue
        items.append(rel.as_posix())
items = sorted(set(items))
print(json.dumps(items))
"""
    try:
        proc = subprocess.run(
            [str(py), "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    stdout = (proc.stdout or "").strip()
    try:
        data = json.loads(stdout) if stdout else []
    except json.JSONDecodeError:
        return []
    return [str(item) for item in data if isinstance(item, str)]


def _format_preset_label(key: str) -> str:
    parts = key.split("/")
    if len(parts) >= 2:
        tail = parts[-1]
        if "_speaker_" in tail:
            lang = parts[-2].upper()
            speaker = tail.replace("_", " ")
            return f"{lang} {speaker}"
    return key


class BarkBackend(TTSBackend):
    id = "bark"
    display_name = "Bark (creative)"
    supports_ref_audio = False
    uses_internal_voices = True

    @classmethod
    def is_available(cls) -> bool:
        return backend_status("bark").get("installed", False)

    @classmethod
    def unavailable_reason(cls) -> str | None:
        return backend_status("bark").get("reason")

    def supported_languages(self) -> list[str]:
        return ["fr-FR", "en-US", "es-ES", "de-DE", "it-IT", "pt-PT", "nl-NL"]

    def default_language(self) -> str:
        return "fr-FR"

    def params_schema(self) -> dict[str, ParamSpec]:
        installed = _probe_bark_presets()
        if installed:
            presets = [(_format_preset_label(key), key) for key in installed]
            default_preset = "v2/en_speaker_6" if "v2/en_speaker_6" in installed else installed[0]
        else:
            presets = [
                ("EN speaker 6 (default)", "v2/en_speaker_6"),
                ("EN speaker 9", "v2/en_speaker_9"),
                ("EN speaker 3", "v2/en_speaker_3"),
                ("DE speaker 0", "v2/de_speaker_0"),
                ("ES speaker 0", "v2/es_speaker_0"),
            ]
            default_preset = "v2/en_speaker_6"
        return {
            "voice_preset": ParamSpec(
                key="voice_preset",
                type="choice",
                default=default_preset,
                choices=presets,
                label="Voice preset",
                help="Historique/preset Bark (history_prompt).",
            ),
            "text_temp": ParamSpec(
                key="text_temp",
                type="float",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Text temp",
            ),
            "waveform_temp": ParamSpec(
                key="waveform_temp",
                type="float",
                default=0.9,
                min=0.0,
                max=1.0,
                step=0.05,
                label="Waveform temp",
            ),
            "seed": ParamSpec(
                key="seed",
                type="int",
                default=0,
                min=0,
                max=2_147_483_647,
                step=1,
                label="Seed",
                help="0 = random (no seed).",
            ),
            "device": ParamSpec(
                key="device",
                type="choice",
                default="cpu",
                choices=[("CPU (recommended)", "cpu")],
                label="Device",
            ),
        }

    def map_language(self, bcp47: Optional[str]) -> Optional[str]:
        if not bcp47:
            return None
        return bcp47.split("-")[0]

    def synthesize(
        self,
        script: str,
        out_path: str,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        raise BackendUnavailableError("Bark backend should use synthesize_chunk.")

    def synthesize_chunk(
        self,
        text: str,
        *,
        voice_ref_path: Optional[str] = None,
        lang: Optional[str] = None,
        **params: Any,
    ):
        py = python_path("bark")
        if not py.exists():
            raise BackendUnavailableError("Bark venv introuvable.")

        assets_dir = Path(__file__).resolve().parents[1] / ".assets" / "bark"
        tmp_dir = assets_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"bark_{abs(hash(text))}.wav"
        runner = Path(__file__).resolve().parent / "bark_runner.py"

        seed = params.get("seed")
        if seed in (0, "0", None, ""):
            seed = None

        runner_payload = {
            "text": text,
            "out_path": str(tmp_wav),
            "assets_dir": str(assets_dir),
            "use_small_models": bool(VOCALIE_BARK_SMALL_MODELS),
            "params": {
                "voice_preset": params.get("voice_preset"),
                "text_temp": params.get("text_temp"),
                "waveform_temp": params.get("waveform_temp"),
                "seed": seed,
                "device": "cpu",
            },
        }
        env = dict(**os.environ)
        env["XDG_CACHE_HOME"] = str(assets_dir)
        env["HF_HOME"] = str(assets_dir / ".hf")
        env["HUGGINGFACE_HUB_CACHE"] = str(assets_dir / ".hf" / "hub")
        env["SUNO_ENABLE_MPS"] = "False"
        if VOCALIE_BARK_SMALL_MODELS:
            env["SUNO_USE_SMALL_MODELS"] = "True"
        try:
            proc = subprocess.run(
                [str(py), str(runner)],
                input=json.dumps(runner_payload),
                text=True,
                capture_output=True,
                check=True,
                env=env,
                timeout=float(VOCALIE_BARK_TIMEOUT_S),
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailableError(
                "Bark timeout (CPU). Premier lancement = téléchargement des poids; "
                "augmente `VOCALIE_BARK_TIMEOUT_S` (ex: 600) ou active `VOCALIE_BARK_SMALL_MODELS=1`."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            message = stderr or stdout or "Bark failed"
            raise BackendUnavailableError(message) from exc

        meta = {}
        stdout = (proc.stdout or "").strip()
        if stdout:
            try:
                meta = json.loads(stdout)
            except json.JSONDecodeError:
                meta = {"ok": False, "stdout": stdout}
        if not tmp_wav.exists():
            raise BackendUnavailableError("Bark did not produce audio.")
        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        return audio, int(sr), meta
