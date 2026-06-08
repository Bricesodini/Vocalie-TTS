"""Duration estimation and adjustment helpers."""

from __future__ import annotations

import re
from typing import Optional

from backend.shared.text_constants import AVERAGE_WPS
from backend.shared.text_models import DurationAdjustment
from backend.shared.text_normalize import normalize_whitespace


def estimate_duration(text: str, words_per_sec: float = AVERAGE_WPS) -> float:
    """Roughly estimate how long *text* will take to speak."""
    words = re.findall(r"\w+", text)
    if words_per_sec <= 0:
        words_per_sec = AVERAGE_WPS
    return max(len(words) / words_per_sec, 0.0)


def adjust_text_to_duration(
    text: str,
    target_seconds: float,
    tolerance: float = 0.2,
) -> DurationAdjustment:
    """Naive text resize helper to hit a rough target duration.

    The implementation intentionally keeps things deterministic and local by
    trimming/duplicating phrases rather than calling an LLM. The UI warns the
    user when edits are aggressive so they may tweak the prose manually.
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        return DurationAdjustment("", 0.0, target_seconds, warning="Texte vide.")

    est = estimate_duration(normalized)
    if target_seconds <= 0:
        return DurationAdjustment(normalized, est, target_seconds)

    if est == 0:
        return DurationAdjustment(normalized, est, target_seconds)

    ratio = target_seconds / est
    warning: Optional[str] = None

    if abs(1 - ratio) <= tolerance:
        return DurationAdjustment(normalized, est, target_seconds)

    words = normalized.split()
    desired_len = max(int(len(words) * ratio), 3)

    if desired_len < len(words):
        new_words = words[:desired_len]
        warning = "Texte raccourci automatiquement, vérifiez le sens."
    else:
        repeat_words = words
        while len(repeat_words) < desired_len:
            repeat_words += words
        new_words = repeat_words[:desired_len]
        warning = "Texte allongé en dupliquant certains segments, ajustez manuellement."

    adjusted = " ".join(new_words)
    new_est = estimate_duration(adjusted)
    return DurationAdjustment(adjusted, new_est, target_seconds, warning)


__all__ = [
    "estimate_duration",
    "adjust_text_to_duration",
]
