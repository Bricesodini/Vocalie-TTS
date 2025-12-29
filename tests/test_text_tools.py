from text_tools import (
    estimate_duration_with_pauses,
    normalize_text,
    render_clean_text,
    split_on_commas,
    strip_legacy_tokens,
)


def test_strip_legacy_tokens():
    text = "Salut {breath} toi {pause:500} ok {beat}."
    assert strip_legacy_tokens(text) == "Salut  toi  ok ."


def test_estimate_duration_with_pauses_counts():
    text = "A, B."
    est = estimate_duration_with_pauses(text, comma_pause_ms=300, period_pause_ms=500, newline_pause_ms=1000)
    assert est > 0.7


def test_estimate_duration_with_pauses_newline():
    text = "A\nB"
    est = estimate_duration_with_pauses(text, comma_pause_ms=300, period_pause_ms=500, newline_pause_ms=1000)
    assert est >= 1.0


def test_render_clean_text_removes_tokens():
    text = "Texte {pause:200} inspiré"
    assert render_clean_text(text) == "Texte inspiré"


def test_split_on_commas():
    text = "Bonjour, le monde, encore."
    parts = split_on_commas(text, max_subsegments=8)
    assert len(parts) == 3


def test_normalize_text_fixes_ii():
    text = "II me manquait."
    assert normalize_text(text).startswith("Il ")
