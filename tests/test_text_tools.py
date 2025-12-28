import pytest

from text_tools import MAX_PAUSE_MS, render_clean_text, split_text_and_pauses


def test_split_text_with_pauses():
    text = "Bonjour {breath} monde {pause:500}! {beat}Finale."
    segments = split_text_and_pauses(text)
    kinds = [seg.kind for seg in segments]
    assert kinds == ["text", "silence", "text", "silence", "text", "silence", "text"]
    assert segments[1].duration_ms == 180
    assert segments[3].duration_ms == 500
    assert segments[5].duration_ms == 250


def test_pause_clamped():
    text = "Salut {pause:9999} tout le monde"
    segments = split_text_and_pauses(text)
    silence = [seg for seg in segments if seg.kind == "silence"][-1]
    assert silence.duration_ms == MAX_PAUSE_MS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Texte {breath} inspiré", "Texte inspiré"),
        ("Première ligne\n{pause:200}Deuxième ligne", "Première ligne\nDeuxième ligne"),
        ("Sans balise", "Sans balise"),
    ],
)
def test_render_clean_text(raw, expected):
    assert render_clean_text(raw) == expected
