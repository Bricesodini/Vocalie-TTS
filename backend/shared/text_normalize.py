"""Text normalization helpers — pure, no external deps beyond stdlib + numpy-free.

Each function is small enough to unit-test in isolation. Side-effect
free (no logging, no global mutation) so they compose safely.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from backend.shared.text_constants import (
    LEGACY_TOKEN_PATTERN,
)


def normalize_whitespace(text: str) -> str:
    """Collapse trailing spaces while keeping manual line breaks."""

    cleaned = []
    for block in text.splitlines():
        cleaned.append(re.sub(r"\s+", " ", block).strip())
    return "\n".join(filter(None, cleaned)).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _first_word(text: str) -> str:
    if not text:
        return ""
    word = text.split(maxsplit=1)[0]
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)


def strip_legacy_tokens(text: str) -> str:
    if not text:
        return ""
    return LEGACY_TOKEN_PATTERN.sub("", text)


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


def normalize_paste_fr(text: str) -> Tuple[str, List[str]]:
    """Normalize text freshly pasted from a word processor / webpage.

    Returns the cleaned text and a list of change tags so the UI can
    explain to the user what was modified.
    """
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


__all__ = [
    "normalize_whitespace",
    "count_words",
    "strip_legacy_tokens",
    "normalize_text",
    "normalize_paste_fr",
]
