"""Audio editing utilities — trim, normalize, metadata extraction.

These functions are the canonical implementation.  Both the API route
(``backend/routes/audio.py``) and the legacy surface use this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def apply_minimal_edit(
    raw_path: Path,
    output_path: Path,
    *,
    trim_enabled: bool,
    normalize_enabled: bool,
    target_dbfs: float,
    silence_threshold: float = 0.002,
    silence_min_ms: int = 20,
    zero_cross_radius_ms: int = 10,
    fade_ms: int = 10,
) -> dict[str, Any]:
    """Apply minimal post-processing: optional trim + normalize.

    This is the public, canonical implementation.  It does NOT modify
    the input file — it writes to *output_path*.
    """
    from backend.shared.tts_pipeline import _find_active_range

    if raw_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from input.")

    audio, sr = sf.read(str(raw_path), always_2d=False)
    if not isinstance(sr, (int, float)):
        raise ValueError("Sample rate invalide pour l'édition.")
    sr = int(sr)
    audio = np.asarray(audio, dtype=np.float32)

    # Trim silence
    trimmed = False
    if trim_enabled:
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        min_silence_frames = int(sr * (int(silence_min_ms) / 1000.0))
        start_idx, end_idx = _find_active_range(
            mono,
            threshold=float(silence_threshold),
            min_silence_frames=min_silence_frames,
        )
        if 0 <= start_idx < end_idx <= len(audio):
            audio = audio[start_idx:end_idx]
            trimmed = True

    # Normalize peak
    normalized = False
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    target_peak = 10 ** (float(target_dbfs) / 20.0)
    gain = 1.0
    if normalize_enabled and peak_before > 0.0 and target_peak > 0.0:
        gain = target_peak / peak_before
        audio = audio * gain
        normalized = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")

    return {
        "trimmed": trimmed,
        "normalized": normalized,
        "target_dbfs": float(target_dbfs),
        "peak_before": peak_before,
        "peak_after": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "gain": gain,
    }


def audio_meta(path: Path) -> dict[str, Any]:
    """Return audio metadata (duration, sample rate, file size)."""
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "duration_s": duration,
        "sample_rate": int(info.samplerate) if info.samplerate else None,
        "size_bytes": int(path.stat().st_size),
    }