"""Lexique (custom dictionary) loading and chatterbox-specific substitutions.

The lexique is a JSON file that maps acronyms to their spelled-out
form and provides exception replacements. We use it to make a
voice-over script read naturally (e.g. "RATP" → "R A T P" or "A I"
instead of "AI" pronounced as a single word).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from backend.shared.text_normalize import normalize_paste_fr


def load_lexique_json(path: str | Path) -> Dict:
    """Load a lexique JSON file with a per-process cache."""
    from backend.shared.text_constants import LEXIQUE_CACHE

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


def normalize_for_chatterbox(text: str, lex: Dict) -> Tuple[str, List[str]]:
    """Apply the lexique (and auto-expand known acronyms) for chatterbox."""
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
    """Full preparation pipeline: paste-normalize then lexique-expand."""
    text1, changes1 = normalize_paste_fr(user_text)
    lex = load_lexique_json(lex_path)
    text2, changes2 = normalize_for_chatterbox(text1, lex)
    return text2, changes1 + changes2


__all__ = [
    "load_lexique_json",
    "normalize_for_chatterbox",
    "prepare_adjusted_text",
]
