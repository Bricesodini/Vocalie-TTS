import numpy as np

from text_tools import (
    FINAL_MERGE_EST_SECONDS,
    MAX_PAUSE_MS,
    chunk_script,
    ensure_strong_ending,
    render_clean_text_from_segments,
    stitch_segments,
)


def test_chunker_sentence_priority():
    text = "Premiere phrase. Deuxieme phrase, avec virgule. Troisieme phrase!"
    chunks = chunk_script(text, max_chars=40, max_sentences=1)
    assert len(chunks) >= 3
    assert chunks[0].reason == "phrase-limit"
    assert chunks[1].reason == "phrase-limit"


def test_chunker_fallback_length():
    text = "A" * 200
    chunks = chunk_script(text, max_chars=50, max_sentences=3)
    assert any(chunk.reason == "char-fallback" for chunk in chunks)


def test_last_chunk_merge():
    text = "Phrase une.\n\nOk."
    chunks = chunk_script(text, max_chars=200, max_sentences=3)
    assert chunks[-1].reason == "merged-final"
    assert chunks[-1].estimated_duration < FINAL_MERGE_EST_SECONDS


def test_tokens_preserved_across_chunks():
    text = f"{'A' * 150} {{pause:4000}} {'B' * 150}"
    chunks = chunk_script(text, max_chars=100, max_sentences=2)
    durations = [
        seg.duration_ms
        for chunk in chunks
        for seg in chunk.segments
        if seg.kind == "silence"
    ]
    assert MAX_PAUSE_MS in durations


def test_concat_length_consistency():
    sr = 1000
    text = "Bonjour {pause:500} monde. Encore une phrase."
    chunks = chunk_script(text, max_chars=50, max_sentences=2)

    def synth_fn(t: str) -> np.ndarray:
        return np.ones(len(t), dtype=np.float32)

    total_len = 0
    expected_len = 0
    for chunk_info in chunks:
        audio = stitch_segments(chunk_info.segments, sr, synth_fn)
        total_len += len(audio)
        for seg in chunk_info.segments:
            if seg.kind == "silence":
                expected_len += int(sr * (seg.duration_ms / 1000.0))
            else:
                expected_len += len(seg.content.strip())
    assert total_len == expected_len


def test_chunk_preview_matches_real():
    text = "Une phrase. Une autre phrase. Derniere."
    chunks = chunk_script(text, max_chars=80, max_sentences=2)
    previews = [render_clean_text_from_segments(chunk.segments) for chunk in chunks]
    assert all(preview for preview in previews)


def test_preview_does_not_include_injected_punctuation():
    text = "Bonjour sans point"
    chunks = chunk_script(text, max_chars=120, max_sentences=2)
    preview = render_clean_text_from_segments(chunks[0].segments)
    assert not preview.endswith(".")
    segments = list(chunks[0].segments)
    ensure_strong_ending(segments)
    injected = render_clean_text_from_segments(segments)
    assert injected.endswith(".")
