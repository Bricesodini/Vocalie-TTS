import pytest

pytest.importorskip("chatterbox")

from tts_engine import _is_too_short_text, _prepare_segments_for_synthesis
from text_tools import SpeechSegment


def test_prepare_segments_filters_short_texts():
    segments = [
        SpeechSegment("text", ""),
        SpeechSegment("text", "\n"),
        SpeechSegment("text", "."),
        SpeechSegment("text", "—"),
        SpeechSegment("text", "a"),
        SpeechSegment("text", "OK."),
        SpeechSegment("text", "Bonjour."),
    ]
    prepared = _prepare_segments_for_synthesis(segments)
    assert all(seg.kind != "text" or not _is_too_short_text(seg.content) for seg in prepared)


def test_prepare_segments_preserves_poetry_tokens():
    segments = [
        SpeechSegment("text", "où le soleil,"),
        SpeechSegment("text", ",Luit"),
        SpeechSegment("text", "tête nue"),
        SpeechSegment("text", "Dort ;"),
        SpeechSegment("text", "."),
        SpeechSegment("text", "\n"),
        SpeechSegment("text", "—"),
    ]
    prepared = _prepare_segments_for_synthesis(segments)
    combined = " ".join(seg.content for seg in prepared if seg.kind == "text")
    tokens = ["où", "soleil", "Luit", "tête", "nue", "Dort"]
    positions = [combined.find(token) for token in tokens]
    assert all(pos >= 0 for pos in positions)
    assert positions == sorted(positions)
