#!/usr/bin/env python3
"""Chatterbox runner invoked inside the Chatterbox venv (JSON stdin/stdout)."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ── HF cache MUST be configured BEFORE importing any of the heavy deps ──
# chatterbox_impl imports huggingface_hub and torch at module load time,
# and those modules read HF_HOME / HUGGINGFACE_HUB_CACHE at import.
# Fallback to a local project dir so it works outside Docker (macOS, dev).
_DEFAULT_ASSETS = str(Path(__file__).resolve().parents[1] / ".assets" / "chatterbox")
HF_ASSETS_DIR = os.environ.get("VOCALIE_HF_ASSETS_DIR", _DEFAULT_ASSETS)
Path(HF_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
Path(HF_ASSETS_DIR, ".hf").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(Path(HF_ASSETS_DIR) / ".hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(HF_ASSETS_DIR) / ".hf" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(HF_ASSETS_DIR) / ".hf" / "hub"))
os.environ.setdefault("TORCH_HOME", str(Path(HF_ASSETS_DIR) / ".torch"))

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tts_backends.chatterbox_impl import ChatterboxEngine
from tts_backends.base_runner import BaseSubprocessRunner


def _read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw:
        raise ValueError("missing JSON payload on stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    return data


def _write_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    sys.stdout.flush()


def _synthesize_chunk(
    engine: ChatterboxEngine,
    text: str,
    *,
    voice_ref_path: str | None,
    lang: str | None,
    tts_model_mode: str,
    multilang_cfg_weight: float,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    tts = engine._get_backend(tts_model_mode)
    requested_language = engine._resolve_language(tts_model_mode, lang or "fr-FR")
    backend_language = requested_language
    effective_cfg = float(cfg_weight)
    language_kw = None
    if tts_model_mode == "multilang":
        backend_language = engine._map_multilang_language(requested_language)
        effective_cfg = float(multilang_cfg_weight)
        language_kw = "language_id"

    audio, _ = engine.synthesize_text(
        tts,
        text,
        voice_ref_path,
        float(exaggeration),
        float(effective_cfg),
        float(temperature),
        float(repetition_penalty),
        backend_language,
        language_kw,
    )
    sr = int(engine.sample_rate or tts.sr)
    retried = False
    duration = len(audio) / sr if sr else 0.0
    if len(text) > 80 and duration < 1.2:
        retried = True
        retry_audio, _ = engine._synthesize_text(
            tts,
            text,
            voice_ref_path,
            float(exaggeration),
            min(1.5, float(effective_cfg) + 0.05),
            max(0.2, float(temperature) - 0.05),
            float(repetition_penalty),
            backend_language,
            language_kw,
            False,
        )
        retry_duration = len(retry_audio) / sr if sr else 0.0
        if retry_duration > duration:
            audio = retry_audio
            duration = retry_duration
    meta = {
        "retry": retried,
        "requested_language_bcp47": requested_language,
        "backend_language_id": backend_language,
    }
    return np.asarray(audio, dtype=np.float32), sr, meta


def main() -> int:
    try:
        print("[chatterbox_runner] start", file=sys.stderr, flush=True)
        payload = _read_payload()
        text = str(payload.get("text") or "")
        out_path = payload.get("out_wav_path") or payload.get("out_path")
        if not out_path:
            raise ValueError("out_wav_path is required")
        tts_model_mode = str(payload.get("tts_model_mode") or payload.get("chatterbox_mode") or "fr_finetune")
        voice_ref_path = payload.get("ref_audio_path") or payload.get("voice_ref_path")
        lang = payload.get("lang") or payload.get("language") or payload.get("tts_language")
        multilang_cfg_weight = float(payload.get("multilang_cfg_weight", 0.5))
        exaggeration = float(payload.get("exaggeration", 0.5))
        cfg_weight = float(payload.get("cfg_weight", 0.6))
        temperature = float(payload.get("temperature", 0.5))
        repetition_penalty = float(payload.get("repetition_penalty", 1.35))

        print(f"[chatterbox_runner] mode={tts_model_mode} text_len={len(text)} out={out_path}", file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            print("[chatterbox_runner] loading engine...", file=sys.stderr, flush=True)
            engine = ChatterboxEngine()
            print("[chatterbox_runner] engine loaded, synthesizing...", file=sys.stderr, flush=True)
            audio, sr, meta = _synthesize_chunk(
                engine,
                text,
                voice_ref_path=voice_ref_path,
                lang=lang,
                tts_model_mode=tts_model_mode,
                multilang_cfg_weight=multilang_cfg_weight,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        out_path = str(Path(out_path).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, sr)
        print(f"[chatterbox_runner] done wav={out_path}", file=sys.stderr, flush=True)
        _write_response(
            {
                "ok": True,
                "out_path": out_path,
                "duration_s": len(audio) / sr if sr else 0.0,
                "retry": bool(meta.get("retry")),
                "logs": [],
            }
        )
        return 0
    except Exception as exc:
        _write_response(
            {
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
