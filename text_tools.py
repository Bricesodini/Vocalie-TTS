"""Utilities for preparing text prior to passing it to the TTS engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


AVERAGE_WPS = 2.6  # empiric speaking speed for French promo VO

PAUSE_PATTERN = re.compile(r"\{(?P<token>pause:\s*\d+|breath|beat)\}", re.IGNORECASE)
PAUSE_ALIASES = {"breath": 180, "beat": 250}
MAX_PAUSE_MS = 2000


def normalize_whitespace(text: str) -> str:
    """Collapse trailing spaces while keeping manual line breaks."""

    cleaned = []
    for block in text.splitlines():
        cleaned.append(re.sub(r"\s+", " ", block).strip())
    return "\n".join(filter(None, cleaned)).strip()


def estimate_duration(text: str, words_per_sec: float = AVERAGE_WPS) -> float:
    """Roughly estimate how long *text* will take to speak."""

    words = re.findall(r"\w+", text)
    if words_per_sec <= 0:
        words_per_sec = AVERAGE_WPS
    return max(len(words) / words_per_sec, 0.0)


@dataclass
class DurationAdjustment:
    text: str
    estimated_duration: float
    target_duration: float
    warning: str | None = None


@dataclass
class SpeechSegment:
    kind: str  # "text" or "silence"
    content: str
    duration_ms: int = 0


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
    warning = None

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


def _pause_duration(token: str) -> int:
    token = token.strip().lower()
    if token in PAUSE_ALIASES:
        return PAUSE_ALIASES[token]
    if token.startswith("pause:"):
        digits = re.findall(r"\d+", token)
        if digits:
            value = int(digits[0])
            value = max(0, min(value, MAX_PAUSE_MS))
            return value
    return 0


def split_text_and_pauses(text: str) -> List[SpeechSegment]:
    """Split the script into text chunks and silence directives."""

    if not text:
        return []

    segments: List[SpeechSegment] = []
    last = 0
    for match in PAUSE_PATTERN.finditer(text):
        chunk = text[last:match.start()]
        if chunk:
            segments.append(SpeechSegment("text", chunk))
        duration = _pause_duration(match.group("token"))
        if duration > 0:
            segments.append(SpeechSegment("silence", "", duration))
        last = match.end()

    tail = text[last:]
    if tail:
        segments.append(SpeechSegment("text", tail))
    if not segments:
        segments.append(SpeechSegment("text", text))
    return segments


def render_clean_text(text: str) -> str:
    """Return the script without pause directives, lightly normalized."""

    segments = split_text_and_pauses(text)
    cleaned = "".join(seg.content for seg in segments if seg.kind == "text")
    return normalize_whitespace(cleaned)


__all__ = [
    "AVERAGE_WPS",
    "DurationAdjustment",
    "SpeechSegment",
    "MAX_PAUSE_MS",
    "adjust_text_to_duration",
    "estimate_duration",
    "normalize_whitespace",
    "render_clean_text",
    "split_text_and_pauses",
]
