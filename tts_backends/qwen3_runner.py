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


def _pick_dtype(torch_mod, dtype_value: str | None):
    if not dtype_value:
        return None
    dtype_key = str(dtype_value).lower()
    mapping = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    name = mapping.get(dtype_key)
    if not name:
        return None
    return getattr(torch_mod, name, None)


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _log(message: str, *, log_path: str | None = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}\n"
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        _write_error("invalid_json", detail=str(exc))
        return 2

    text = str(payload.get("text") or "")
    out_path = str(payload.get("out_path") or "")
    mode = str(payload.get("mode") or "custom_voice")
    model_id = payload.get("model_id")
    language = payload.get("language") or "Auto"
    speaker = payload.get("speaker")
    instruct = payload.get("instruct") or ""
    ref_text = payload.get("ref_text") or ""
    x_vector_only_mode = _coerce_bool(payload.get("x_vector_only_mode"), True)
    voice_ref_path = payload.get("voice_ref_path")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    assets_dir = Path(payload.get("assets_dir") or "").expanduser()
    debug_log_path = payload.get("debug_log_path")

    if not text.strip():
        _write_error("empty_text")
        return 2
    if not out_path:
        _write_error("missing_out_path")
        return 2

    if assets_dir:
        assets_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(assets_dir / ".hf"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(assets_dir / ".hf" / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(assets_dir / ".hf" / "hub"))
        os.environ.setdefault("TORCH_HOME", str(assets_dir / ".torch"))

    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:  # noqa: BLE001
        _write_error("qwen_import_failed", detail=str(exc))
        return 3

    device = params.get("device") or "auto"
    dtype = _pick_dtype(torch, params.get("dtype"))
    attn_implementation = params.get("attn_implementation")

    model_kwargs = {"device_map": device}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    start = time.time()
    _log(
        f"start mode={mode} model_id={model_id} language={language} speaker={speaker} "
        f"x_vector_only={x_vector_only_mode} ref_audio={bool(voice_ref_path)} "
        f"ref_text_len={len(str(ref_text))}",
        log_path=debug_log_path,
    )
    try:
        _log("loading model...", log_path=debug_log_path)
        model = Qwen3TTSModel.from_pretrained(model_id, **model_kwargs)
        _log("model loaded", log_path=debug_log_path)
        if mode == "voice_design":
            _log("generate_voice_design", log_path=debug_log_path)
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
        elif mode == "voice_clone":
            if not voice_ref_path:
                _write_error("missing_ref_audio")
                return 4
            if not x_vector_only_mode and not str(ref_text).strip():
                _write_error("missing_ref_text")
                return 4
            _log("generate_voice_clone", log_path=debug_log_path)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=voice_ref_path,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
        else:
            if speaker is None and hasattr(model, "get_supported_speakers"):
                speakers = model.get_supported_speakers()
                if speakers:
                    speaker = speakers[0]
            _log("generate_custom_voice", log_path=debug_log_path)
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or None,
            )
    except Exception as exc:  # noqa: BLE001
        _write_error("qwen_generation_failed", detail=str(exc))
        return 5

    try:
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio = np.asarray(audio, dtype=np.float32)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, int(sr), subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        _write_error("write_failed", detail=str(exc))
        return 6

    elapsed_ms = int((time.time() - start) * 1000)
    _log(f"done elapsed_ms={elapsed_ms}", log_path=debug_log_path)
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "sample_rate": int(sr),
                "duration_ms": elapsed_ms,
                "mode": mode,
                "model_id": model_id,
            }
        )
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
