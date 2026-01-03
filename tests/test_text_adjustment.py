from pathlib import Path

import app
from text_tools import normalize_paste_fr, prepare_adjusted_text


LEX_PATH = Path(__file__).resolve().parents[1] / "lexique_tts_fr.json"


def test_punctuation_spacing_after_comma():
    text, _changes = normalize_paste_fr(",Luit")
    assert text == ", Luit"
    text, _changes = normalize_paste_fr("fière,Luit")
    assert text == "fière, Luit"
    text, _changes = normalize_paste_fr("fière ,Luit")
    assert text == "fière, Luit"
    text, _changes = normalize_paste_fr("fière,\u00A0Luit")
    assert text == "fière, Luit"


def test_space_before_comma_removed():
    text, _changes = normalize_paste_fr("mot ,")
    assert text == "mot,"


def test_sigle_undot_then_exception():
    text, _changes = prepare_adjusted_text("C.N.C.", LEX_PATH)
    assert text == "céainecé"


def test_exception_replacement():
    text, _changes = prepare_adjusted_text("MJC", LEX_PATH)
    assert text == "èmjicé"


def test_sigle_auto_fallback():
    text, _changes = prepare_adjusted_text("DRAC", LEX_PATH)
    assert text == "déèracé"


def test_sigles_with_digits_ignored():
    text, _changes = prepare_adjusted_text("USB4 et 4K", LEX_PATH)
    assert "uèssbéquatre" in text
    assert "4K" in text


def test_auto_sigles_ignore_long_tokens():
    text, _changes = prepare_adjusted_text("BONJOUR", LEX_PATH)
    assert text == "BONJOUR"


def test_auto_adjustment_off_returns_raw():
    adjusted, _clean, _duration, log_md = app.handle_text_adjustment(
        text="C.N.C.",
        auto_adjust=False,
        show_adjust_log=True,
    )
    assert adjusted == "C.N.C."
    assert log_md == ""
