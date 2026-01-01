from text_tools import (
    estimate_duration,
    normalize_text,
    render_clean_text,
    strip_legacy_tokens,
)


def test_strip_legacy_tokens():
    text = "Salut {breath} toi {pause:500} ok {beat}."
    assert strip_legacy_tokens(text) == "Salut  toi  ok ."


def test_estimate_duration_basic():
    text = "A, B."
    est = estimate_duration(text)
    assert est > 0.0


def test_render_clean_text_removes_tokens():
    text = "Texte {pause:200} inspiré"
    assert render_clean_text(text) == "Texte inspiré"




def test_normalize_text_fixes_ii():
    text = "II me manquait."
    assert normalize_text(text).startswith("Il ")
