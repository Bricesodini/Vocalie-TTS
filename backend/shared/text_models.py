"""Dataclasses passed between the chunking / rendering / synthesis stages.

These types are the lingua franca of the text preparation pipeline.
Keeping them in their own module avoids circular imports between the
behaviour modules (chunk / render / duration) and makes it easy for
the API layer to type-annotate the responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DurationAdjustment:
    """Result of resizing a script to hit a target speaking duration."""

    text: str
    estimated_duration: float
    target_duration: float
    warning: str | None = None


@dataclass
class SpeechSegment:
    """One chunk of audio to be synthesized (or a silence gap)."""

    kind: str  # "text" or "silence"
    content: str
    duration_ms: int = 0


@dataclass
class TextUnit:
    """Internal tokenized unit used by some chunking helpers."""

    text: str
    sentence_end: bool = False
    hard_break: bool = False
    char_fallback: bool = False


@dataclass
class ChunkInfo:
    """One chunk of the prepared script, with all the metadata the
    pipeline needs to schedule synthesis and stitch the audio back."""

    segments: List[SpeechSegment]
    sentence_count: int
    char_count: int
    word_count: int
    comma_count: int
    estimated_duration: float
    reason: str
    boundary_kind: str | None = None
    pivot: bool = False
    ends_with_suspended: bool = False
    oversize_sentence: bool = False
    warnings: List[str] = field(default_factory=list)


__all__ = [
    "DurationAdjustment",
    "SpeechSegment",
    "TextUnit",
    "ChunkInfo",
]
