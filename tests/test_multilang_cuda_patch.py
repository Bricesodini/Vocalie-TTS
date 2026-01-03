import pytest
torch = pytest.importorskip("torch")

pytest.importorskip("chatterbox")

import tts_engine


def test_multilang_cuda_checkpoint_patch(monkeypatch):
    calls = {"count": 0, "map_location": []}

    class DummyMultilang:
        sr = 24000

        def to(self, device):
            return self

        @classmethod
        def from_pretrained(cls, device):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError(
                    "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
                )
            torch.load("dummy.pt")
            return cls()

    def spy_load(*args, **kwargs):
        calls["map_location"].append(kwargs.get("map_location"))
        return {}

    monkeypatch.setattr(tts_engine, "ChatterboxMultilingualTTS", DummyMultilang)
    monkeypatch.setattr(torch, "load", spy_load)
    engine = tts_engine.TTSEngine()
    backend = engine._load_multilang_backend()
    assert isinstance(backend, DummyMultilang)
    assert calls["count"] == 2
    assert any(loc == torch.device("cpu") for loc in calls["map_location"])
