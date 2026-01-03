import app
from text_tools import MANUAL_CHUNK_MARKER, parse_manual_chunks, render_clean_text_from_segments


def test_parse_manual_chunks_basic():
    chunks, count = parse_manual_chunks(f"Bonjour{MANUAL_CHUNK_MARKER}Salut")
    assert count == 1
    assert len(chunks) == 2
    texts = [render_clean_text_from_segments(chunk.segments) for chunk in chunks]
    assert texts == ["Bonjour", "Salut"]


def test_parse_manual_chunks_trims_empties():
    chunks, count = parse_manual_chunks(f"{MANUAL_CHUNK_MARKER}  Salut  {MANUAL_CHUNK_MARKER}")
    assert count == 2
    assert len(chunks) == 1
    assert render_clean_text_from_segments(chunks[0].segments) == "Salut"


def test_direction_no_markers_single_chunk():
    chunks, mode, meta, _log = app._apply_direction_chunking(
        direction_enabled=True,
        direction_source="final",
        direction_snapshot_text="Bonjour",
        tts_ready_text="Bonjour",
        log_text=None,
    )
    assert mode == "manual_single"
    assert meta["markers_count"] == 0
    assert len(chunks) == 1
    assert render_clean_text_from_segments(chunks[0].segments) == "Bonjour"
