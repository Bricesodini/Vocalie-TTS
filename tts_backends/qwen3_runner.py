#!/usr/bin/env python3
"""Qwen3 runner invoked inside the Qwen3 venv (JSON stdin/stdout).

Uses BaseSubprocessRunner for protocol handling.
Only the synthesis logic is engine-specific.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# The runner is launched with only the venv's site-packages on sys.path
# (no /app). Add the project root so the `tts_backends` package can be
# imported from inside the Qwen3 venv, which is shipped separately
# from the chatterbox venv in Docker.
_BASE_DIR = Path(__file__).resolve().parents[1]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from tts_backends.base_runner import BaseSubprocessRunner


class Qwen3Runner(BaseSubprocessRunner):
    """Qwen3-TTS subprocess runner."""

    def run_synthesis(self, payload: dict) -> dict:
        text = str(payload.get("text") or "")
        out_path = str(payload.get("out_path") or "")
        mode = str(payload.get("mode") or "custom_voice")
        model_id = payload.get("model_id")
        language = payload.get("language") or "Auto"
        speaker = payload.get("speaker")
        instruct = payload.get("instruct") or ""
        ref_text = payload.get("ref_text") or ""
        x_vector_only_mode = self.coerce_bool(payload.get("x_vector_only_mode"), True)
        voice_ref_path = payload.get("voice_ref_path")
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        debug_log_path = payload.get("debug_log_path")

        if not text.strip():
            raise ValueError("empty_text")
        if not out_path:
            raise ValueError("missing_out_path")

        # Setup HuggingFace cache
        assets_dir = payload.get("assets_dir")
        if assets_dir:
            self.setup_hf_cache(assets_dir)

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:
            raise RuntimeError(f"qwen_import_failed: {exc}") from exc

        device = params.get("device") or "auto"
        # Default to float16: 1.7B params @ fp32 = ~6.8 GB, which OOMs the
        # 7.6 GB Docker Desktop container on macOS. fp16 = ~3.4 GB, leaves
        # room for the code_predictor and inference activations.
        dtype_value = params.get("dtype") or "float16"
        dtype = self.resolve_dtype(torch, dtype_value)
        attn_implementation = params.get("attn_implementation")

        # Hugging Face's device_map="auto" silently offloads the model to the
        # meta device when no accelerator is visible (no CUDA, no MPS, no
        # XPU). On Mac/amd64 hosts this manifests as
        # "Tensor.item() cannot be called on meta tensors" at the first
        # forward pass of generate(). Detect that and pin to CPU instead.
        if device == "auto":
            has_cuda = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
            has_mps = bool(getattr(getattr(torch, "backends", None), "mps", None)) and torch.backends.mps.is_available()
            if not has_cuda and not has_mps:
                device = "cpu"

        # Resolve the *actual* device that HF will land on, so the debug
        # log tells the user whether they're getting CPU, MPS, or CUDA.
        # On Apple Silicon hosts (M-series) this is typically "mps" and
        # gives a meaningful speedup over CPU.
        effective_device = device
        if device == "auto":
            if has_cuda:
                effective_device = "cuda"
            elif has_mps:
                effective_device = "mps"
            else:
                effective_device = "cpu"

        model_kwargs = {"device_map": device}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        start = time.time()
        self.log(
            f"start mode={mode} model_id={model_id} language={language} speaker={speaker} "
            f"x_vector_only={x_vector_only_mode} ref_audio={bool(voice_ref_path)} "
            f"ref_text_len={len(str(ref_text))} device={effective_device} dtype={dtype_value}",
            log_path=debug_log_path,
        )

        self.log("loading model...", log_path=debug_log_path)
        model = Qwen3TTSModel.from_pretrained(model_id, **model_kwargs)
        self.log("model loaded", log_path=debug_log_path)

        if mode == "voice_design":
            self.log("generate_voice_design", log_path=debug_log_path)
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
        elif mode == "voice_clone":
            if not voice_ref_path:
                raise RuntimeError("missing_ref_audio")
            if not x_vector_only_mode and not str(ref_text).strip():
                raise RuntimeError("missing_ref_text")
            self.log("generate_voice_clone", log_path=debug_log_path)
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
            self.log("generate_custom_voice", log_path=debug_log_path)
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or None,
            )

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio = np.asarray(audio, dtype=np.float32)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, int(sr), subtype="PCM_16")

        elapsed_ms = int((time.time() - start) * 1000)
        self.log(f"done elapsed_ms={elapsed_ms}", log_path=debug_log_path)

        return {
            "ok": True,
            "sample_rate": int(sr),
            "duration_ms": elapsed_ms,
            "mode": mode,
            "model_id": model_id,
        }

    @staticmethod
    def coerce_bool(value, default: bool) -> bool:
        """Backward-compatible alias for BaseSubprocessRunner.coerce_bool."""
        from tts_backends.base import coerce_bool as _cb
        return _cb(value, default)


if __name__ == "__main__":
    raise SystemExit(Qwen3Runner.main())