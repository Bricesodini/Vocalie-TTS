"""Chunking logic: turn a long script into a list of ChunkInfo ready
to be synthesized. This module is the heaviest in the pipeline; the
helpers are split by responsibility (tokenization, state, splitting,
oversize handling, short-chunk merging, pivot splitting).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from backend.shared.text_constants import (
    AVERAGE_WPS,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    DEFAULT_MIN_WORDS_PER_CHUNK,
    DETERMINERS,
    FALLBACK_PUNCTUATION,
    MANUAL_CHUNK_MARKER,
    PIVOT_WORDS,
    TERMINATOR_CHARS,
)
from backend.shared.text_models import ChunkInfo, SpeechSegment
from backend.shared.text_normalize import (
    _first_word,
    count_words,
    normalize_text,
)
from backend.shared.text_render import (
    render_clean_text,
    render_clean_text_from_segments,
)


# ── token-level helpers ─────────────────────────────────────────────────


def _tokenize_for_chunking(text: str) -> List[str]:
    return re.findall(r"\w+|\n|[^\w\n]", text)


def _is_word_token(token: str) -> bool:
    return bool(re.fullmatch(r"\w+", token))


def _is_dash_token(tokens: Sequence[str], idx: int) -> bool:
    if idx < 0 or idx >= len(tokens):
        return False
    if tokens[idx] == "—":
        return True
    if tokens[idx] != "-":
        return False
    before = tokens[idx - 1] if idx > 0 else ""
    after = tokens[idx + 1] if idx + 1 < len(tokens) else ""
    return bool(before.isspace() and after.isspace())


def _recalculate_chunk_state(
    tokens: Sequence[str],
) -> Tuple[int, int, Optional[int], Dict[str, Optional[int]]]:
    word_count = 0
    words_since_terminator = 0
    last_terminator_idx: Optional[int] = None
    fallback_indices: Dict[str, Optional[int]] = {punct: None for punct in FALLBACK_PUNCTUATION}
    for idx, token in enumerate(tokens):
        if _is_word_token(token):
            word_count += 1
            words_since_terminator += 1
            continue
        if token in TERMINATOR_CHARS:
            last_terminator_idx = idx
            words_since_terminator = 0
            continue
        if token == "\n":
            continue
        if token in (":", ";", "—", ","):
            fallback_indices[token] = idx
            continue
        if token == "-" and _is_dash_token(tokens, idx):
            fallback_indices["-"] = idx
    return word_count, words_since_terminator, last_terminator_idx, fallback_indices


def _select_fallback_index(
    fallback_indices: Dict[str, Optional[int]],
) -> Tuple[Optional[str], Optional[int]]:
    for punct in FALLBACK_PUNCTUATION:
        idx = fallback_indices.get(punct)
        if idx is not None:
            return punct, idx
    return None, None


def _find_word_split_index(
    tokens: Sequence[str],
    target_words: int,
    min_words: int,
) -> Optional[int]:
    word_positions: List[Tuple[int, int, str]] = []
    word_count = 0
    for idx, token in enumerate(tokens):
        if _is_word_token(token):
            word_count += 1
            word_positions.append((word_count, idx, token))
    if not word_positions:
        return None
    if target_words <= 0:
        target_words = word_positions[-1][0]
    if target_words < min_words:
        target_words = min_words
    if target_words > word_positions[-1][0]:
        target_words = word_positions[-1][0]
    split_idx = word_positions[-1][1]
    split_word = word_positions[-1][2]
    for count, idx, token in word_positions:
        if count >= target_words:
            split_idx = idx
            split_word = token
            break
    if split_word.lower() in DETERMINERS:
        for count, idx, _ in word_positions:
            if count == min(target_words + 1, word_positions[-1][0]):
                split_idx = idx
                break
    return split_idx


# ── public entry point ──────────────────────────────────────────────────


def chunk_script(
    script: str,
    *,
    min_words_per_chunk: int = DEFAULT_MIN_WORDS_PER_CHUNK,
    max_words_without_terminator: int = DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    max_est_seconds_per_chunk: float = DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    split_on_newline: bool = True,
) -> List[ChunkInfo]:
    """Split a script into chunks suitable for sequential synthesis.

    Splits on terminators first, then falls back to weaker punctuation
    when a chunk gets too long, then to hard word splits if nothing
    else is available.
    """
    cleaned = normalize_text(script)
    if not cleaned:
        return []
    min_words_per_chunk = max(0, min(int(min_words_per_chunk), 20))
    tokens = _tokenize_for_chunking(cleaned)
    if not tokens:
        return []
    max_words_per_chunk = int(max_est_seconds_per_chunk * AVERAGE_WPS) if max_est_seconds_per_chunk > 0 else 0

    buffer: List[str] = []
    word_count = 0
    words_since_terminator = 0
    last_terminator_idx: Optional[int] = None
    fallback_indices = {punct: None for punct in FALLBACK_PUNCTUATION}
    warnings_current: List[str] = []
    chunks: List[ChunkInfo] = []
    tail_from_forced_split = False

    def finalize_chunk(split_idx: int, reason: str, boundary_kind: Optional[str], warnings: List[str]) -> None:
        nonlocal buffer, word_count, words_since_terminator, last_terminator_idx, fallback_indices, warnings_current
        nonlocal tail_from_forced_split
        if split_idx < 0:
            return
        chunk_tokens = buffer[: split_idx + 1]
        chunk_text = "".join(chunk_tokens)
        if boundary_kind == "newline":
            chunk_text = chunk_text.rstrip("\n")
        chunk_text = chunk_text.strip()
        if chunk_text:
            clean = render_clean_text(chunk_text)
            sentence_count = len(re.findall(r"[.!?]", clean))
            chunks.append(
                ChunkInfo(
                    segments=[SpeechSegment("text", chunk_text)],
                    sentence_count=sentence_count,
                    char_count=len(chunk_text),
                    word_count=count_words(clean),
                    comma_count=clean.count(","),
                    estimated_duration=_estimate_duration_local(clean),
                    reason=reason,
                    boundary_kind=boundary_kind,
                    pivot=False,
                    ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
                    oversize_sentence=False,
                    warnings=list(warnings),
                )
            )
        tail_from_forced_split = reason == "hard" or reason.startswith("fallback(")
        buffer = buffer[split_idx + 1 :]
        while buffer and buffer[0].isspace():
            buffer.pop(0)
        word_count, words_since_terminator, last_terminator_idx, fallback_indices = _recalculate_chunk_state(buffer)
        warnings_current = []

    for idx, token in enumerate(tokens):
        buffer.append(token)
        if _is_word_token(token):
            word_count += 1
            words_since_terminator += 1
        elif token in TERMINATOR_CHARS:
            last_terminator_idx = len(buffer) - 1
            words_since_terminator = 0
        if token == "\n" and split_on_newline:
            if word_count >= min_words_per_chunk:
                finalize_chunk(len(buffer) - 1, "newline", "newline", warnings_current)
                continue
            warnings_current.append("newline_boundary_skipped_min_words")
        if token in (":", ";", "—", ","):
            fallback_indices[token] = len(buffer) - 1
        elif token == "-" and _is_dash_token(tokens, idx):
            fallback_indices["-"] = len(buffer) - 1

        if max_words_without_terminator > 0 and words_since_terminator > max_words_without_terminator:
            punct, split_idx = _select_fallback_index(fallback_indices)
            if split_idx is not None and punct is not None:
                warnings = warnings_current + [f"fallback_split_used:{punct}"]
                finalize_chunk(split_idx, f"fallback({punct})", punct, warnings)
                continue
            split_idx = _find_word_split_index(buffer, max_words_without_terminator, min_words_per_chunk)
            if split_idx is not None:
                warnings = warnings_current + ["hard_split_no_punct"]
                finalize_chunk(split_idx, "hard", "hard", warnings)
                continue

        if max_words_per_chunk > 0 and word_count > max_words_per_chunk:
            if last_terminator_idx is not None:
                finalize_chunk(last_terminator_idx, "terminator", "terminator", warnings_current)
                continue
            punct, split_idx = _select_fallback_index(fallback_indices)
            if split_idx is not None and punct is not None:
                warnings = warnings_current + [f"fallback_split_used:{punct}"]
                finalize_chunk(split_idx, f"fallback({punct})", punct, warnings)
                continue
            split_idx = _find_word_split_index(buffer, max_words_per_chunk, min_words_per_chunk)
            if split_idx is not None:
                warnings = warnings_current + ["hard_split_no_punct"]
                finalize_chunk(split_idx, "hard", "hard", warnings)
                continue

    if buffer:
        if (
            tail_from_forced_split
            and word_count < min_words_per_chunk
            and chunks
        ):
            merged_text = render_clean_text_from_segments(chunks[-1].segments) + "".join(buffer)
            clean = render_clean_text(merged_text)
            chunks[-1] = ChunkInfo(
                segments=[SpeechSegment("text", merged_text)],
                sentence_count=len(re.findall(r"[.!?]", clean)),
                char_count=len(merged_text),
                word_count=count_words(clean),
                comma_count=clean.count(","),
                estimated_duration=_estimate_duration_local(clean),
                reason=chunks[-1].reason,
                boundary_kind=chunks[-1].boundary_kind,
                pivot=chunks[-1].pivot,
                ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
                oversize_sentence=chunks[-1].oversize_sentence,
                warnings=list(chunks[-1].warnings),
            )
        else:
            finalize_chunk(len(buffer) - 1, "end", None, warnings_current)
    return chunks


def parse_manual_chunks(
    snapshot: str,
    *,
    marker: str = MANUAL_CHUNK_MARKER,
) -> Tuple[List[ChunkInfo], int]:
    """Honor explicit user-provided [[CHUNK]] markers."""
    if not snapshot:
        return [], 0
    marker_count = snapshot.count(marker)
    if marker_count == 0:
        return [], 0
    parts = snapshot.split(marker)
    chunks: List[ChunkInfo] = []
    for part in parts:
        chunk_text = part.strip()
        if not chunk_text:
            continue
        clean = render_clean_text(chunk_text)
        sentence_count = len(re.findall(r"[.!?]", clean))
        chunks.append(
            ChunkInfo(
                segments=[SpeechSegment("text", chunk_text)],
                sentence_count=sentence_count,
                char_count=len(chunk_text),
                word_count=count_words(clean),
                comma_count=clean.count(","),
                estimated_duration=_estimate_duration_local(clean),
                reason="manual_marker",
                boundary_kind="manual_marker",
                pivot=False,
                ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
                oversize_sentence=False,
                warnings=[],
            )
        )
    return chunks, marker_count


# ── post-processing: oversize / short / pivot ───────────────────────────


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


def _split_text_by_punctuation(text: str, punct: str) -> List[str]:
    if punct not in text:
        return [text]
    parts = []
    start = 0
    for idx, ch in enumerate(text):
        if ch == punct:
            parts.append(text[start : idx + 1])
            start = idx + 1
    tail = text[start:]
    if tail:
        parts.append(tail)
    return parts


def _split_by_word_count(text: str, max_words: int, safe_tail_words: int = 2) -> List[str]:
    if max_words <= 0:
        return [text]
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts = []
    idx = 0
    while idx < len(words):
        end = min(idx + max_words, len(words))
        remaining = len(words) - end
        if 0 < remaining < safe_tail_words:
            end = max(len(words) - safe_tail_words, idx + 1)
        chunk = " ".join(words[idx:end])
        parts.append(chunk)
        idx = end
    return parts


def _split_oversize_chunks(
    chunks: List[ChunkInfo],
    max_est_seconds: float,
    max_chars: int,
) -> List[ChunkInfo]:
    if max_est_seconds <= 0:
        return chunks
    refined: List[ChunkInfo] = []
    for chunk in chunks:
        if chunk.estimated_duration <= max_est_seconds:
            refined.append(chunk)
            continue
        text = render_clean_text_from_segments(chunk.segments)
        candidates: List[str] = []
        for punct in [".", "!", "?", "…"]:
            if punct in text:
                candidates = _split_text_by_punctuation(text, punct)
                if len(candidates) > 1:
                    break
                candidates = []
        if not candidates:
            for punct in [";", ":"]:
                if punct in text:
                    candidates = _split_text_by_punctuation(text, punct)
                    if len(candidates) > 1:
                        break
                    candidates = []
        if not candidates and "\n" in text:
            candidates = text.split("\n")
        if not candidates:
            candidates = _split_by_length(text, max_chars if max_chars > 0 else DEFAULT_MAX_CHARS_PER_CHUNK)
        max_words = max(int(max_est_seconds * AVERAGE_WPS), 1)
        for part in candidates:
            clean = render_clean_text(part)
            if max_words > 0 and count_words(clean) > max_words:
                subparts = _split_by_word_count(clean, max_words)
            else:
                subparts = [part]
            for sub in subparts:
                clean_sub = render_clean_text(sub)
                word_count = count_words(clean_sub)
                comma_count = clean_sub.count(",")
                boundary_kind: Optional[str] = None
                stripped = clean_sub.rstrip()
                if stripped.endswith((".", "!", "?", "…")):
                    boundary_kind = "period"
                elif "\n" in clean_sub:
                    boundary_kind = "newline"
                else:
                    boundary_kind = "hard"
                refined.append(
                    ChunkInfo(
                        segments=[SpeechSegment("text", sub)],
                        sentence_count=1,
                        char_count=len(sub),
                        word_count=word_count,
                        comma_count=comma_count,
                        estimated_duration=_estimate_duration_local(clean_sub),
                        reason="max-est-split",
                        boundary_kind=boundary_kind,
                        pivot=False,
                        ends_with_suspended=False,
                        oversize_sentence=chunk.oversize_sentence,
                    )
                )
    return refined


def _merge_short_chunks(
    chunks: List[ChunkInfo],
    min_words: int,
    max_est_seconds: float,
) -> List[ChunkInfo]:
    if min_words <= 0 or len(chunks) <= 1:
        return chunks
    merged: List[ChunkInfo] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        clean_current = render_clean_text_from_segments(current.segments)
        words = count_words(clean_current)
        est = _estimate_duration_local(clean_current)
        is_short = words < min_words or est < 2.0
        if not is_short:
            merged.append(current)
            idx += 1
            continue
        if current.pivot:
            merged.append(current)
            idx += 1
            continue
        if idx < len(chunks) - 1:
            next_chunk = chunks[idx + 1]
            if next_chunk.pivot:
                merged.append(current)
                idx += 1
                continue
            merged_text = clean_current + " " + render_clean_text_from_segments(next_chunk.segments)
            clean = render_clean_text(merged_text)
            merged_est = _estimate_duration_local(clean)
            if max_est_seconds > 0 and merged_est > max_est_seconds:
                merged.append(current)
                idx += 1
                continue
            merged.append(
                ChunkInfo(
                    segments=[SpeechSegment("text", merged_text)],
                    sentence_count=current.sentence_count + next_chunk.sentence_count,
                    char_count=len(merged_text),
                    word_count=count_words(clean),
                    comma_count=clean.count(","),
                    estimated_duration=merged_est,
                    reason="min-words-merge",
                    boundary_kind=next_chunk.boundary_kind,
                    pivot=current.pivot or next_chunk.pivot,
                    ends_with_suspended=False,
                    oversize_sentence=current.oversize_sentence or next_chunk.oversize_sentence,
                )
            )
            idx += 2
            continue
        if merged:
            prev = merged.pop()
            merged_text = render_clean_text_from_segments(prev.segments) + " " + clean_current
            clean = render_clean_text(merged_text)
            merged_est = _estimate_duration_local(clean)
            if max_est_seconds > 0 and merged_est > max_est_seconds:
                merged.append(prev)
                merged.append(current)
                idx += 1
                continue
            merged.append(
                ChunkInfo(
                    segments=[SpeechSegment("text", merged_text)],
                    sentence_count=prev.sentence_count + current.sentence_count,
                    char_count=len(merged_text),
                    word_count=count_words(clean),
                    comma_count=clean.count(","),
                    estimated_duration=merged_est,
                    reason="min-words-merge",
                    boundary_kind=prev.boundary_kind,
                    pivot=prev.pivot or current.pivot,
                    ends_with_suspended=False,
                    oversize_sentence=prev.oversize_sentence or current.oversize_sentence,
                )
            )
        idx += 1
    return merged


def _apply_pivot_splits(
    chunks: List[ChunkInfo],
    max_est_seconds: float,
    min_words: int,
) -> List[ChunkInfo]:
    refined: List[ChunkInfo] = []
    for chunk in chunks:
        text = render_clean_text_from_segments(chunk.segments).strip()
        first = _first_word(text)
        comma_count = text.count(",")
        est = chunk.estimated_duration
        if first in PIVOT_WORDS and (comma_count > 2 or est > max_est_seconds):
            split_idx = -1
            if comma_count > 0:
                comma_positions = [idx for idx, ch in enumerate(text) if ch == ","]
                for idx in comma_positions:
                    left = text[: idx + 1]
                    left_clean = render_clean_text(left)
                    if count_words(left_clean) >= max(min_words, 2) and _estimate_duration_local(left_clean) >= 2.0:
                        split_idx = idx
                        break
            if split_idx == -1 and "." in text:
                dot_idx = text.find(".")
                left = text[: dot_idx + 1]
                left_clean = render_clean_text(left)
                if count_words(left_clean) >= max(min_words, 2) and _estimate_duration_local(left_clean) >= 2.0:
                    split_idx = dot_idx
            if split_idx != -1:
                left = text[: split_idx + 1]
                right = text[split_idx + 1 :].lstrip()
                parts = [part for part in (left, right) if part.strip()]
                if len(parts) < 2:
                    refined.append(chunk)
                    continue
                for part in parts:
                    clean = render_clean_text(part)
                    refined.append(
                        ChunkInfo(
                            segments=[SpeechSegment("text", part)],
                            sentence_count=1,
                            char_count=len(part),
                            word_count=count_words(clean),
                            comma_count=clean.count(","),
                            estimated_duration=_estimate_duration_local(clean),
                            reason="pivot-split",
                            boundary_kind=None,
                            pivot=True,
                            ends_with_suspended=False,
                            oversize_sentence=chunk.oversize_sentence,
                        )
                    )
                continue
        refined.append(chunk)
    return refined


# Local import to avoid a cycle: text_chunk depends on text_duration
# transitively for the estimate_duration helper, but we re-define it
# locally to keep the dep graph shallow. text_duration.py remains the
# canonical source of truth.
def _estimate_duration_local(text: str) -> float:
    """Local wrapper that re-uses text_duration.estimate_duration."""
    from backend.shared.text_duration import estimate_duration
    return estimate_duration(text)


__all__ = [
    "chunk_script",
    "parse_manual_chunks",
    "_split_by_length",
    "_split_text_by_punctuation",
    "_split_by_word_count",
    "_split_oversize_chunks",
    "_merge_short_chunks",
    "_apply_pivot_splits",
]
