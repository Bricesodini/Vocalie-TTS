from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_error(message: str, *, detail: str | None = None) -> None:
    payload = {"ok": False, "error": message}
    if detail:
        payload["detail"] = detail
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        _write_error("invalid_json", detail=str(exc))
        return 2

    text = str(payload.get("text") or "")
    out_path = str(payload.get("out_path") or "")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if not text.strip():
        _write_error("empty_text")
        return 2
    if not out_path:
        _write_error("missing_out_path")
        return 2

    assets_dir = Path(payload.get("assets_dir") or "").expanduser()
    if not assets_dir:
        assets_dir = Path.cwd() / ".assets" / "bark"
    assets_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(assets_dir))
    os.environ.setdefault("HF_HOME", str(assets_dir / ".hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(assets_dir / ".hf" / "hub"))

    voice_preset = str(params.get("voice_preset") or "v2/en_speaker_6")
    text_temp = _coerce_float(params.get("text_temp"), 0.7)
    waveform_temp = _coerce_float(params.get("waveform_temp"), 0.7)
    seed = _coerce_int(params.get("seed"), None)
    device = str(params.get("device") or "cpu")
    if device != "cpu":
        device = "cpu"

    if payload.get("use_small_models") is True:
        os.environ.setdefault("SUNO_USE_SMALL_MODELS", "True")
    os.environ.setdefault("SUNO_ENABLE_MPS", "False")

    start = time.time()
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
    except Exception as exc:  # noqa: BLE001
        _write_error("bark_import_failed", detail=str(exc))
        return 3

    try:
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch

                torch.manual_seed(seed)
            except Exception:
                pass

        preload_models()
        audio = generate_audio(
            text,
            history_prompt=voice_preset,
            text_temp=float(max(0.0, min(1.0, text_temp))),
            waveform_temp=float(max(0.0, min(1.0, waveform_temp))),
        )
        audio = np.asarray(audio, dtype=np.float32)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, int(SAMPLE_RATE), subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        _write_error("bark_generation_failed", detail=str(exc))
        return 4

    elapsed_ms = int((time.time() - start) * 1000)
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "sample_rate": int(SAMPLE_RATE),
                "duration_ms": elapsed_ms,
                "voice_preset": voice_preset,
            }
        )
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
