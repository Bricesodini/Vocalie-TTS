#!/usr/bin/env python3
"""CosyVoice runner invoked inside the CosyVoice venv (JSON stdin/stdout).

Uses BaseSubprocessRunner for protocol handling.
Only the synthesis logic is engine-specific.

CosyVoice supports three modes:
- instruct: inference_instruct2(text, instruct_text, ref_audio)
- clone: inference_zero_shot(text, prompt_text, ref_audio)
- cross_lingual: inference_cross_lingual(text, ref_audio)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_backends.base_runner import BaseSubprocessRunner


class CosyVoiceRunner(BaseSubprocessRunner):
    """CosyVoice-TTS subprocess runner."""

    def run_synthesis(self, payload: dict) -> dict:
        text = str(payload.get("text") or "")
        out_path = str(payload.get("out_path") or "")
        mode = str(payload.get("mode") or "clone")
        model_id = payload.get("model_id") or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        language = payload.get("language") or "French"
        instruct_text = payload.get("instruct_text") or ""
        prompt_text = payload.get("prompt_text") or ""
        voice_ref_path = payload.get("voice_ref_path")
        streaming = self.coerce_bool(payload.get("streaming"), False)
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
        except Exception as exc:
            raise RuntimeError(f"torch_import_failed: {exc}") from exc

        start = time.time()
        self.log(
            f"start mode={mode} model_id={model_id} language={language} "
            f"ref_audio={bool(voice_ref_path)} streaming={streaming} "
            f"instruct_len={len(instruct_text)} prompt_len={len(str(prompt_text))}",
            log_path=debug_log_path,
        )

        # Load CosyVoice model
        self.log("loading model...", log_path=debug_log_path)

        try:
            # CosyVoice requires adding its submodule to sys.path
            cosyvoice_root = os.environ.get("COSYVOICE_ROOT")
            if cosyvoice_root:
                matcha_path = os.path.join(cosyvoice_root, "third_party", "Matcha-TTS")
                if os.path.isdir(matcha_path) and matcha_path not in sys.path:
                    sys.path.insert(0, matcha_path)

            from cosyvoice.cli.cosyvoice import AutoModel
            model = AutoModel(model_dir=model_id)
        except ImportError as exc:
            # Try without AutoModel — some CosyVoice versions have different API
            raise RuntimeError(f"cosyvoice_import_failed: {exc}") from exc

        self.log("model loaded", log_path=debug_log_path)

        # Generate audio
        sr = 22050  # Default; will be overridden by model.sample_rate
        chunks = []

        if mode == "instruct":
            self.log("generate_instruct", log_path=debug_log_path)
            if voice_ref_path:
                gen = model.inference_instruct2(
                    text, instruct_text, voice_ref_path,
                    stream=streaming,
                )
            else:
                gen = model.inference_instruct2(
                    text, instruct_text, None,
                    stream=streaming,
                )
            for chunk in gen:
                if "tts_speech" in chunk:
                    chunks.append(chunk["tts_speech"])

        elif mode == "cross_lingual":
            if not voice_ref_path:
                raise RuntimeError("cross_lingual_requires_ref_audio")
            self.log("generate_cross_lingual", log_path=debug_log_path)
            gen = model.inference_cross_lingual(
                text, voice_ref_path,
                stream=streaming,
            )
            for chunk in gen:
                if "tts_speech" in chunk:
                    chunks.append(chunk["tts_speech"])

        else:  # clone (zero-shot)
            if not voice_ref_path:
                raise RuntimeError("clone_requires_ref_audio")
            self.log("generate_clone", log_path=debug_log_path)
            gen = model.inference_zero_shot(
                text, prompt_text, voice_ref_path,
                stream=streaming,
            )
            for chunk in gen:
                if "tts_speech" in chunk:
                    chunks.append(chunk["tts_speech"])

        if not chunks:
            raise RuntimeError("no_audio_generated")

        # Concatenate chunks
        sr = getattr(model, "sample_rate", 22050)
        import torch
        audio_tensor = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        audio = np.asarray(audio_tensor.cpu().numpy(), dtype=np.float32)
        # Flatten if multi-dimensional
        if audio.ndim > 1:
            audio = audio.squeeze()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, int(sr), subtype="PCM_16")

        duration_s = len(audio) / sr
        elapsed_ms = int((time.time() - start) * 1000)
        self.log(
            f"done duration_s={duration_s:.2f} elapsed_ms={elapsed_ms}",
            log_path=debug_log_path,
        )

        return {
            "ok": True,
            "sample_rate": int(sr),
            "duration_ms": int(duration_s * 1000),
            "mode": mode,
            "model_id": model_id,
            "streaming": streaming,
        }


if __name__ == "__main__":
    raise SystemExit(CosyVoiceRunner.main())