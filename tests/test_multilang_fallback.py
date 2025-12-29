import torch
import pytest

pytest.importorskip("chatterbox")

from tts_engine import TTSEngine


class DummyBackend:
    def __init__(self):
        self.calls = []

    def generate(self, text, language_id=None, language=None, **kwargs):
        self.calls.append({"language_id": language_id, "language": language})
        if language_id is not None and language is None:
            raise TypeError("unexpected keyword 'language_id'")
        return torch.zeros((1, 1))


def test_language_id_fallback_to_language():
    engine = TTSEngine()
    backend = DummyBackend()
    wav = engine._synthesize_text(
        backend,
        "Bonjour",
        None,
        0.5,
        0.6,
        0.5,
        1.35,
        "fr",
        "language_id",
        False,
    )
    assert wav.shape[0] == 1
    assert len(backend.calls) == 2
    assert backend.calls[0]["language_id"] == "fr"
    assert backend.calls[1]["language"] == "fr"
