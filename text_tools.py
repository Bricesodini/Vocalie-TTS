"""Utilities for preparing text prior to passing it to the TTS engine."""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


AVERAGE_WPS = 2.6  # empiric speaking speed for French promo VO
DEFAULT_MAX_CHARS_PER_CHUNK = 380
DEFAULT_MAX_PHRASES_PER_CHUNK = 3
FINAL_MERGE_EST_SECONDS = 3.5

LEGACY_TOKEN_PATTERN = re.compile(r"\{(?P<token>pause:\s*\d+|breath|beat)\}", re.IGNORECASE)
MAX_PAUSE_MS = 4000
DEFAULT_INTER_CHUNK_PAUSE_MS = 500
DEFAULT_COMMA_PAUSE_MS = 250
DEFAULT_PERIOD_PAUSE_MS = 400
DEFAULT_SEMICOLON_PAUSE_MS = 300
DEFAULT_COLON_PAUSE_MS = 300
DEFAULT_DASH_PAUSE_MS = 250
DEFAULT_NEWLINE_PAUSE_MS = 300
DEFAULT_MAX_COMMA_SUBSEGMENTS = 8
DEFAULT_MIN_WORDS_PER_CHUNK = 16
DEFAULT_MAX_EST_SECONDS_PER_CHUNK = 10.0
DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR = 35
PIVOT_WORDS = {"Cependant", "Pourtant", "Or", "Alors", "Néanmoins", "Toutefois"}
TERMINATOR_CHARS = (".", "!", "?")
FALLBACK_PUNCTUATION = (":", ";", "—", "-", ",")
LEXIQUE_CACHE: Dict[str, Dict] = {}
DETERMINERS = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "au",
    "aux",
    "ce",
    "cet",
    "cette",
    "ces",
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "son",
    "sa",
    "ses",
    "notre",
    "nos",
    "votre",
    "vos",
    "leur",
    "leurs",
}


def normalize_whitespace(text: str) -> str:
    """Collapse trailing spaces while keeping manual line breaks."""

    cleaned = []
    for block in text.splitlines():
        cleaned.append(re.sub(r"\s+", " ", block).strip())
    return "\n".join(filter(None, cleaned)).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def stabilize_trailing_punct(text: str) -> tuple[str, str | None]:
    stripped = text.rstrip()
    if not stripped:
        return text, None
    if stripped.endswith(("...", "…")):
        return text, None
    if stripped.endswith((",", ";", ":")):
        return f"{stripped[:-1]}.", f"trailing '{stripped[-1]}' -> '.'"
    return text, None


def _first_word(text: str) -> str:
    if not text:
        return ""
    word = text.split(maxsplit=1)[0]
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    content = text.replace("\r\n", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = strip_legacy_tokens(content)
    content = re.sub(r'(^|[.!?\n;:])\s*II\b', r"\1 Il", content)
    content = re.sub(r"([.!?;:,])(?!\s|$)", r"\1 ", content)
    lines = []
    for line in content.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    normalized = "\n".join(lines)
    return normalized.strip()


def load_lexique_json(path: str | Path) -> Dict:
    cache_key = str(path)
    if cache_key in LEXIQUE_CACHE:
        return LEXIQUE_CACHE[cache_key]
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    except json.JSONDecodeError:
        data = {}
    LEXIQUE_CACHE[cache_key] = data
    return data


def normalize_paste_fr(text: str) -> Tuple[str, List[str]]:
    if text is None:
        return "", ["paste_norm_applied: false"]
    original = text
    content = text.replace("\r\n", "\n").replace("\r", "\n")
    content, nbsp_count = re.subn(r"[\u00A0\u202F\u2007]", " ", content)
    content, space_count = re.subn(r"[ \t]+", " ", content)
    content, ellipsis_count = re.subn(r"\.{3,}", "…", content)
    content, double_dot_count = re.subn(r"(?<!\.)\.\.(?!\.)", ".", content)
    content, space_before_count = re.subn(r"\s+([,.;:!?])", r"\1", content)
    content, space_after_count = re.subn(r'([,.;:!?])(?=[^\s»”"])', r"\1 ", content)
    content, newline_count = re.subn(r"\n{3,}", "\n\n", content)
    content = content.strip()

    changes: List[str] = []
    paste_changed = content != original
    changes.append(f"paste_norm_applied: {str(paste_changed).lower()}")
    if paste_changed:
        changes.append(
            "paste_norm_counts: "
            f"nbsp={nbsp_count}, spaces={space_count}, "
            f"ellipsis={ellipsis_count}, double_dot={double_dot_count}, "
            f"space_before_punct={space_before_count}, space_after_punct={space_after_count}, "
            f"newlines={newline_count}"
        )
    return content, changes


def normalize_for_chatterbox(text: str, lex: Dict) -> Tuple[str, List[str]]:
    if not text:
        return "", []
    exceptions = lex.get("exceptions", {}) if lex else {}
    letters = lex.get("letters", {}) if lex else {}
    changes: List[str] = []

    def undot(match: re.Match) -> str:
        original = match.group(0)
        compact = re.sub(r"[.\s]+", "", original)
        if compact != original:
            changes.append(f"sigle_undot: {original} -> {compact}")
        return compact

    undot_pattern = re.compile(r"(?:[A-Z]\.\s*){2,10}")
    content = undot_pattern.sub(undot, text)

    for key, replacement in exceptions.items():
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        content, count = pattern.subn(replacement, content)
        if count:
            changes.append(f"lexicon_hit: {key} -> {replacement}")

    auto_hits: Dict[str, int] = {}

    def replace_sigle(match: re.Match) -> str:
        token = match.group(0)
        if token in exceptions:
            return token
        if any(ch.isdigit() for ch in token):
            return token
        pieces = []
        for ch in token:
            rep = letters.get(ch)
            if rep is None:
                return token
            pieces.append(rep)
        replacement = "".join(pieces)
        auto_hits[token] = auto_hits.get(token, 0) + 1
        return replacement

    auto_pattern = re.compile(r"\b[A-Z]{2,6}\b")
    content = auto_pattern.sub(replace_sigle, content)
    for token, _count in auto_hits.items():
        assembled = "".join(letters.get(ch, "") for ch in token)
        changes.append(f"sigle_auto: {token} -> {assembled}")
    return content, changes


def prepare_adjusted_text(user_text: str, lex_path: str | Path) -> Tuple[str, List[str]]:
    text1, changes1 = normalize_paste_fr(user_text)
    lex = load_lexique_json(lex_path)
    text2, changes2 = normalize_for_chatterbox(text1, lex)
    return text2, changes1 + changes2


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


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
class PauseEvent:
    symbol: str
    duration_ms: int


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
    word_count: int
    comma_count: int
    estimated_duration: float
    reason: str
    boundary_kind: str | None = None
    pivot: bool = False
    ends_with_suspended: bool = False
    oversize_sentence: bool = False
    warnings: List[str] = field(default_factory=list)


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


def strip_legacy_tokens(text: str) -> str:
    if not text:
        return ""
    return LEGACY_TOKEN_PATTERN.sub("", text)


def get_trailing_silence_ms(segments: Iterable[SpeechSegment]) -> int:
    total = 0
    for segment in reversed(list(segments)):
        if segment.kind != "silence":
            break
        total += segment.duration_ms
    return total


def compute_inter_chunk_pause_ms(trailing_ms: int, target_ms: int) -> int:
    if trailing_ms >= target_ms:
        return 0
    return max(target_ms - max(trailing_ms, 0), 0)


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


def _is_dash_separator(text: str, idx: int) -> bool:
    if idx < 0 or idx >= len(text):
        return False
    if text[idx] == "—":
        return True
    if text[idx] != "-":
        return False
    before = text[idx - 1] if idx > 0 else ""
    after = text[idx + 1] if idx + 1 < len(text) else ""
    return bool(before.isspace() and after.isspace())


@dataclass
class PunctUnit:
    kind: str  # text, comma, period, semicolon, colon, dash, newline, paragraph
    value: str


def tokenize_punctuation(
    text: str,
    *,
    sentence_endings: Sequence[str] = (".", "!", "?", "…"),
) -> List[PunctUnit]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    units: List[PunctUnit] = []
    buffer: List[str] = []
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if ch == "\n" and cleaned[i : i + 2] == "\n\n":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("paragraph", "\n\n"))
            i += 2
            continue
        if ch == "\n":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("newline", "\n"))
            i += 1
            continue
        if ch == ",":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("comma", ch))
            i += 1
            continue
        if ch == ";":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("semicolon", ch))
            i += 1
            continue
        if ch == ":":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("colon", ch))
            i += 1
            continue
        if ch in ("—", "-") and _is_dash_separator(cleaned, i):
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("dash", ch))
            i += 1
            continue
        if ch == "." and cleaned[i : i + 3] == "...":
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("period", "..."))
            i += 3
            continue
        if ch in sentence_endings:
            if buffer:
                units.append(PunctUnit("text", "".join(buffer)))
                buffer = []
            units.append(PunctUnit("period", ch))
            i += 1
            continue
        buffer.append(ch)
        i += 1
    if buffer:
        units.append(PunctUnit("text", "".join(buffer)))
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
    return normalize_text(cleaned)


def split_text_and_pauses(
    text: str,
    *,
    comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
    period_pause_ms: int = DEFAULT_PERIOD_PAUSE_MS,
    semicolon_pause_ms: int = DEFAULT_SEMICOLON_PAUSE_MS,
    colon_pause_ms: int = DEFAULT_COLON_PAUSE_MS,
    dash_pause_ms: int = DEFAULT_DASH_PAUSE_MS,
    newline_pause_ms: int = DEFAULT_NEWLINE_PAUSE_MS,
    suppress_final_pause: bool = False,
    return_events: bool = False,
) -> List[SpeechSegment] | tuple[List[SpeechSegment], List[PauseEvent]]:
    if not text:
        return [] if not return_events else ([], [])
    cleaned = normalize_text(text)
    if not cleaned:
        return [] if not return_events else ([], [])
    pause_map = {
        ",": int(comma_pause_ms),
        ".": int(period_pause_ms),
        "!": int(period_pause_ms),
        "?": int(period_pause_ms),
        ";": int(semicolon_pause_ms),
        ":": int(colon_pause_ms),
        "—": int(dash_pause_ms),
        "-": int(dash_pause_ms),
    }
    segments: List[SpeechSegment] = []
    events: List[PauseEvent] = []
    start = 0
    for idx, ch in enumerate(cleaned):
        if ch == "\n":
            if idx > start:
                segments.append(SpeechSegment("text", cleaned[start:idx]))
            pause_ms = max(int(newline_pause_ms), 0)
            if pause_ms > 0:
                segments.append(SpeechSegment("silence", "", pause_ms))
                events.append(PauseEvent("\\n", pause_ms))
            start = idx + 1
            continue
        if ch in pause_map:
            if ch == "-" and not _is_dash_separator(cleaned, idx):
                continue
            if idx + 1 > start:
                segments.append(SpeechSegment("text", cleaned[start : idx + 1]))
            pause_ms = max(pause_map[ch], 0)
            if pause_ms > 0:
                segments.append(SpeechSegment("silence", "", pause_ms))
                events.append(PauseEvent(ch, pause_ms))
            start = idx + 1
    if start < len(cleaned):
        segments.append(SpeechSegment("text", cleaned[start:]))

    if suppress_final_pause and segments and segments[-1].kind == "silence":
        segments.pop()
        if events:
            events.pop()

    if return_events:
        return segments, events
    return segments


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


def _recalculate_chunk_state(tokens: Sequence[str]) -> tuple[int, int, int | None, dict[str, int | None]]:
    word_count = 0
    words_since_terminator = 0
    last_terminator_idx: int | None = None
    fallback_indices: dict[str, int | None] = {punct: None for punct in FALLBACK_PUNCTUATION}
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
    fallback_indices: dict[str, int | None],
) -> tuple[str | None, int | None]:
    for punct in FALLBACK_PUNCTUATION:
        idx = fallback_indices.get(punct)
        if idx is not None:
            return punct, idx
    return None, None


def _find_word_split_index(
    tokens: Sequence[str],
    target_words: int,
    min_words: int,
) -> int | None:
    word_positions: List[tuple[int, int, str]] = []
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


def chunk_script(
    script: str,
    *,
    min_words_per_chunk: int = DEFAULT_MIN_WORDS_PER_CHUNK,
    max_words_without_terminator: int = DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR,
    max_est_seconds_per_chunk: float = DEFAULT_MAX_EST_SECONDS_PER_CHUNK,
    split_on_newline: bool = True,
) -> List[ChunkInfo]:
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
    last_terminator_idx: int | None = None
    fallback_indices = {punct: None for punct in FALLBACK_PUNCTUATION}
    warnings_current: List[str] = []
    chunks: List[ChunkInfo] = []
    tail_from_forced_split = False

    def finalize_chunk(split_idx: int, reason: str, boundary_kind: str | None, warnings: List[str]) -> None:
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
                    estimated_duration=estimate_duration(clean),
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
                estimated_duration=estimate_duration(clean),
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

    return normalize_text(text or "")


def estimate_duration_with_pauses(
    text: str,
    *,
    comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
    period_pause_ms: int = DEFAULT_PERIOD_PAUSE_MS,
    semicolon_pause_ms: int = DEFAULT_SEMICOLON_PAUSE_MS,
    colon_pause_ms: int = DEFAULT_COLON_PAUSE_MS,
    dash_pause_ms: int = DEFAULT_DASH_PAUSE_MS,
    newline_pause_ms: int = DEFAULT_NEWLINE_PAUSE_MS,
    sentence_endings: Sequence[str] = (".", "!", "?", "…"),
) -> float:
    cleaned = normalize_text(text)
    units = tokenize_punctuation(cleaned, sentence_endings=sentence_endings)
    comma_count = sum(1 for unit in units if unit.kind == "comma")
    period_count = sum(1 for unit in units if unit.kind == "period")
    semicolon_count = sum(1 for unit in units if unit.kind == "semicolon")
    colon_count = sum(1 for unit in units if unit.kind == "colon")
    dash_count = sum(1 for unit in units if unit.kind == "dash")
    newline_count = sum(1 for unit in units if unit.kind in ("newline", "paragraph"))
    base = estimate_duration(cleaned)
    pause_seconds = (
        comma_count * max(comma_pause_ms, 0)
        + period_count * max(period_pause_ms, 0)
        + semicolon_count * max(semicolon_pause_ms, 0)
        + colon_count * max(colon_pause_ms, 0)
        + dash_count * max(dash_pause_ms, 0)
        + newline_count * max(newline_pause_ms, 0)
    ) / 1000.0
    return base + pause_seconds


def split_on_commas(text: str, max_subsegments: int = DEFAULT_MAX_COMMA_SUBSEGMENTS) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    segments: List[str] = []
    buffer: List[str] = []
    comma_count = 0
    for ch in cleaned:
        buffer.append(ch)
        if ch == ",":
            comma_count += 1
            segments.append("".join(buffer))
            buffer = []
    if buffer:
        segments.append("".join(buffer))
    if comma_count >= max_subsegments:
        return [cleaned]
    return segments


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
                boundary_kind = None
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
                        estimated_duration=estimate_duration(clean_sub),
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
        est = estimate_duration(clean_current)
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
            merged_est = estimate_duration(clean)
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
            merged_est = estimate_duration(clean)
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
                    if count_words(left_clean) >= max(min_words, 2) and estimate_duration(left_clean) >= 2.0:
                        split_idx = idx
                        break
            if split_idx == -1 and "." in text:
                dot_idx = text.find(".")
                left = text[: dot_idx + 1]
                left_clean = render_clean_text(left)
                if count_words(left_clean) >= max(min_words, 2) and estimate_duration(left_clean) >= 2.0:
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
                            estimated_duration=estimate_duration(clean),
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


__all__ = [
    "AVERAGE_WPS",
    "DEFAULT_MAX_CHARS_PER_CHUNK",
    "DEFAULT_MAX_PHRASES_PER_CHUNK",
    "DEFAULT_INTER_CHUNK_PAUSE_MS",
    "DEFAULT_COMMA_PAUSE_MS",
    "DEFAULT_PERIOD_PAUSE_MS",
    "DEFAULT_SEMICOLON_PAUSE_MS",
    "DEFAULT_COLON_PAUSE_MS",
    "DEFAULT_DASH_PAUSE_MS",
    "DEFAULT_NEWLINE_PAUSE_MS",
    "DEFAULT_MAX_COMMA_SUBSEGMENTS",
    "DEFAULT_MIN_WORDS_PER_CHUNK",
    "DEFAULT_MAX_EST_SECONDS_PER_CHUNK",
    "DEFAULT_MAX_WORDS_WITHOUT_TERMINATOR",
    "FINAL_MERGE_EST_SECONDS",
    "DurationAdjustment",
    "SpeechSegment",
    "PauseEvent",
    "ChunkInfo",
    "TextUnit",
    "PunctUnit",
    "MAX_PAUSE_MS",
    "adjust_text_to_duration",
    "chunk_script",
    "estimate_duration",
    "estimate_duration_with_pauses",
    "ensure_strong_ending",
    "compute_inter_chunk_pause_ms",
    "get_trailing_silence_ms",
    "load_lexique_json",
    "normalize_paste_fr",
    "normalize_for_chatterbox",
    "prepare_adjusted_text",
    "normalize_whitespace",
    "count_words",
    "normalize_text",
    "render_clean_text",
    "render_clean_text_from_segments",
    "split_on_commas",
    "split_text_and_pauses",
    "stabilize_trailing_punct",
    "strip_legacy_tokens",
    "stitch_segments",
    "tokenize_punctuation",
]
