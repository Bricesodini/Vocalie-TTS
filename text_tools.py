"""Utilities for preparing text prior to passing it to the TTS engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np


AVERAGE_WPS = 2.6  # empiric speaking speed for French promo VO
DEFAULT_MAX_CHARS_PER_CHUNK = 380
DEFAULT_MAX_PHRASES_PER_CHUNK = 3
FINAL_MERGE_EST_SECONDS = 3.5

PAUSE_PATTERN = re.compile(r"\{(?P<token>pause:\s*\d+|breath|beat)\}", re.IGNORECASE)
PAUSE_ALIASES = {"breath": 180, "beat": 250}
MAX_PAUSE_MS = 4000


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


@dataclass
class TextUnit:
    text: str
    sentence_end: bool = False
    hard_break: bool = False
    char_fallback: bool = False


@dataclass
class ChunkInfo:
    segments: List[SpeechSegment]
    sentence_count: int
    char_count: int
    estimated_duration: float
    reason: str
    oversize_sentence: bool = False


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


def _split_by_length(text: str, max_chars: int) -> List[str]:
    parts: List[str] = []
    text = text.strip()
    while len(text) > max_chars:
        cut = max(text.rfind(" ", 0, max_chars), text.rfind("\n", 0, max_chars))
        if cut <= 0:
            cut = max_chars
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def _text_units(
    text: str,
    sentence_endings: Sequence[str],
    prefer_newlines: bool,
) -> List[TextUnit]:
    units: List[TextUnit] = []
    if not text:
        return units
    content = text.replace("\r\n", "\n")
    i = 0
    buffer = []
    while i < len(content):
        if prefer_newlines and content[i : i + 2] == "\n\n":
            buffer.append("\n\n")
            units.append(TextUnit("".join(buffer), hard_break=True))
            buffer = []
            i += 2
            continue
        ch = content[i]
        buffer.append(ch)
        if ch in sentence_endings:
            units.append(TextUnit("".join(buffer), sentence_end=True))
            buffer = []
        i += 1
    if buffer:
        units.append(TextUnit("".join(buffer)))
    return units


def ensure_strong_ending(
    segments: List[SpeechSegment],
    sentence_endings: Sequence[str] = (".", "!", "?", "…"),
) -> None:
    for segment in reversed(segments):
        if segment.kind != "text":
            continue
        stripped = segment.content.rstrip()
        if not stripped:
            continue
        if stripped[-1] not in sentence_endings:
            segment.content = f"{segment.content.rstrip()}{sentence_endings[0]}"
        break


def render_clean_text_from_segments(segments: Iterable[SpeechSegment]) -> str:
    cleaned = "".join(seg.content for seg in segments if seg.kind == "text")
    return normalize_whitespace(cleaned)


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


def chunk_script(
    script: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_sentences: int = DEFAULT_MAX_PHRASES_PER_CHUNK,
    prefer_newlines: bool = True,
    sentence_endings: Sequence[str] = (".", "!", "?", "…"),
) -> List[ChunkInfo]:
    segments = split_text_and_pauses(script)
    if not segments:
        return []

    expanded: List[tuple[SpeechSegment, TextUnit]] = []
    for seg in segments:
        if seg.kind == "silence":
            expanded.append((seg, TextUnit("")))
            continue
        units = _text_units(seg.content, sentence_endings, prefer_newlines)
        for unit in units:
            if len(unit.text) > max_chars:
                pieces = _split_by_length(unit.text, max_chars)
                for idx, piece in enumerate(pieces):
                    is_last = idx == len(pieces) - 1
                    expanded.append(
                        (
                            SpeechSegment("text", piece),
                            TextUnit(
                                piece,
                                sentence_end=unit.sentence_end and is_last,
                                hard_break=unit.hard_break and is_last,
                                char_fallback=True,
                            ),
                        )
                    )
            else:
                expanded.append((SpeechSegment("text", unit.text), unit))

    chunks: List[ChunkInfo] = []
    current: List[SpeechSegment] = []
    char_count = 0
    sentence_count = 0
    oversize_sentence = False

    def finalize_chunk(reason: str) -> None:
        nonlocal current, char_count, sentence_count, oversize_sentence
        if not current:
            return
        clean = render_clean_text_from_segments(current)
        chunks.append(
            ChunkInfo(
                segments=list(current),
                sentence_count=sentence_count,
                char_count=char_count,
                estimated_duration=estimate_duration(clean),
                reason=reason,
                oversize_sentence=oversize_sentence,
            )
        )
        current = []
        char_count = 0
        sentence_count = 0
        oversize_sentence = False

    for seg, unit in expanded:
        current.append(seg)
        if seg.kind == "text":
            char_count += len(seg.content)
            if unit.sentence_end:
                sentence_count += 1

        if unit.hard_break and prefer_newlines:
            finalize_chunk("paragraph")
            continue

        if unit.sentence_end and sentence_count >= max_sentences:
            finalize_chunk("phrase-limit")
            continue

        if unit.char_fallback:
            oversize_sentence = True
            finalize_chunk("char-fallback")

    finalize_chunk("end")

    if len(chunks) >= 2:
        last = chunks[-1]
        prev = chunks[-2]
        if last.estimated_duration < FINAL_MERGE_EST_SECONDS and prev.sentence_count < max_sentences:
            merged_segments = prev.segments + last.segments
            clean = render_clean_text_from_segments(merged_segments)
            merged = ChunkInfo(
                segments=merged_segments,
                sentence_count=prev.sentence_count + last.sentence_count,
                char_count=prev.char_count + last.char_count,
                estimated_duration=estimate_duration(clean),
                reason="merged-final",
                oversize_sentence=prev.oversize_sentence or last.oversize_sentence,
            )
            chunks = chunks[:-2] + [merged]

    return chunks


def stitch_segments(
    segments: Iterable[SpeechSegment],
    sr: int,
    synth_fn: Callable[[str], np.ndarray],
) -> np.ndarray:
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

def render_clean_text(text: str) -> str:
    """Return the script without pause directives, lightly normalized."""

    segments = split_text_and_pauses(text)
    cleaned = "".join(seg.content for seg in segments if seg.kind == "text")
    return normalize_whitespace(cleaned)


__all__ = [
    "AVERAGE_WPS",
    "DEFAULT_MAX_CHARS_PER_CHUNK",
    "DEFAULT_MAX_PHRASES_PER_CHUNK",
    "FINAL_MERGE_EST_SECONDS",
    "DurationAdjustment",
    "SpeechSegment",
    "ChunkInfo",
    "TextUnit",
    "MAX_PAUSE_MS",
    "adjust_text_to_duration",
    "chunk_script",
    "estimate_duration",
    "ensure_strong_ending",
    "normalize_whitespace",
    "render_clean_text",
    "render_clean_text_from_segments",
    "split_text_and_pauses",
    "stitch_segments",
]
