"""Render a chunk / segment back to a clean, engine-ready string.

These functions are the final step before the text is sent to the TTS
backend. The contract is: the output is normalized, free of legacy
directives, and safe to feed to the engine as-is.
"""

from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np

from backend.shared.text_models import SpeechSegment
from backend.shared.text_normalize import count_words, normalize_text


def render_clean_text(text: str) -> str:
    """Return the script without pause directives, lightly normalized."""
    return normalize_text(text or "")


def render_clean_text_from_segments(segments: Iterable[SpeechSegment]) -> str:
    cleaned = "".join(seg.content for seg in segments if seg.kind == "text")
    return normalize_text(cleaned)


def stitch_segments(
    segments: Iterable[SpeechSegment],
    sr: int,
    synth_fn: Callable[[str], np.ndarray],
) -> np.ndarray:
    """Concatenate synthesized audio chunks, with explicit silence gaps.

    synth_fn is provided by the caller (e.g. the chatterbox backend); this
    keeps the renderer free of any TTS-engine import.
    """
    audio_chunks: List[np.ndarray] = []
    for segment in segments:
        if segment.kind == "silence":
            frames = int(sr * (segment.duration_ms / 1000.0))
            if frames <= 0:
                continue
            audio_chunks.append(np.zeros(frames, dtype=np.float32))
            continue
        clean = segment.content.strip()
        if not clean:
            continue
        audio_chunks.append(synth_fn(clean).astype(np.float32))
    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audio_chunks)


__all__ = [
    "render_clean_text",
    "render_clean_text_from_segments",
    "stitch_segments",
]
