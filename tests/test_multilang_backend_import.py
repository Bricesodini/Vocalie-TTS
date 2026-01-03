import importlib

import pytest

pytest.importorskip("chatterbox")

import tts_engine


def test_multilang_import_uses_mtl_tts(monkeypatch):
    class DummyMultilang:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device):
            return cls()

        def generate(self, *args, **kwargs):
            raise RuntimeError("not used in this test")

    monkeypatch.setattr(tts_engine, "ChatterboxMultilingualTTS", DummyMultilang)
    engine = tts_engine.TTSEngine()
    backend = engine._load_multilang_backend()
    assert isinstance(backend, DummyMultilang)
