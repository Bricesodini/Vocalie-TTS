import numpy as np

import tts_pipeline
from text_tools import ChunkInfo, SpeechSegment


class DummyBackend:
    id = "dummy"

    def is_available(self):
        return True

    def unavailable_reason(self):
        return None

    def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
        sr = params.get("sr", 24000)
        length = params.get("length", sr // 2)
        audio = np.ones(length, dtype=np.float32)
        return audio, sr, {}


def _make_chunks():
    return [
        ChunkInfo(
            segments=[SpeechSegment("text", "Hello")],
            sentence_count=1,
            char_count=5,
            word_count=1,
            comma_count=0,
            estimated_duration=0.5,
            reason="hard",
            boundary_kind=",",
            pivot=False,
            ends_with_suspended=False,
            oversize_sentence=False,
            warnings=[],
        ),
        ChunkInfo(
            segments=[SpeechSegment("text", "World")],
            sentence_count=1,
            char_count=5,
            word_count=1,
            comma_count=0,
            estimated_duration=0.5,
            reason="end",
            boundary_kind=None,
            pivot=False,
            ends_with_suspended=False,
            oversize_sentence=False,
            warnings=[],
        ),
    ]


def test_pipeline_applies_postprocessing_all_engines(monkeypatch, tmp_path):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    called = {"count": 0}

    def fake_post(audio, sr, zero_cross_radius_ms, fade_ms, silence_threshold, silence_min_ms):
        called["count"] += 1
        return audio

    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", fake_post)
    request = {
        "tts_backend": "dummy",
        "script": "Hello World",
        "chunks": _make_chunks(),
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 0},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {},
        "target_sr": 24000,
    }
    tts_pipeline.run_tts_pipeline(request)
    assert called["count"] == 1


def test_pipeline_inserts_pauses(monkeypatch, tmp_path):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", lambda audio, *_args, **_kwargs: audio)
    chunks = _make_chunks()
    request = {
        "tts_backend": "dummy",
        "script": "Hello World",
        "chunks": chunks,
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 500},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {"length": 24000, "sr": 24000},
        "target_sr": 24000,
    }
    result = tts_pipeline.run_tts_pipeline(request)
    audio, sr = tts_pipeline.sf.read(result.out_path, dtype="float32")
    assert sr == 24000
    expected_len = 24000 + int(0.5 * 24000) + 24000
    assert abs(len(audio) - expected_len) < 100


def test_pipeline_calls_backend_per_chunk(monkeypatch, tmp_path):
    calls = {"count": 0}

    class CountingBackend(DummyBackend):
        def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
            calls["count"] += 1
            return super().synthesize_chunk(text, voice_ref_path=voice_ref_path, lang=lang, **params)

    backend = CountingBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", lambda audio, *_args, **_kwargs: audio)
    request = {
        "tts_backend": "dummy",
        "script": "Hello World",
        "chunks": _make_chunks(),
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 0},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {},
        "target_sr": 24000,
    }
    tts_pipeline.run_tts_pipeline(request)
    assert calls["count"] == 2


def test_pipeline_resample_if_needed(monkeypatch, tmp_path):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    called = {"count": 0}

    def fake_resample(audio, orig_sr, target_sr):
        called["count"] += 1
        return np.zeros(int(len(audio) * target_sr / orig_sr), dtype=np.float32)

    monkeypatch.setattr(tts_pipeline, "_resample_audio", fake_resample)
    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", lambda audio, *_args, **_kwargs: audio)
    request = {
        "tts_backend": "dummy",
        "script": "Hello",
        "chunks": _make_chunks()[:1],
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 0},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {"sr": 16000, "length": 16000},
        "target_sr": 24000,
    }
    tts_pipeline.run_tts_pipeline(request)
    assert called["count"] == 1


def test_pipeline_coerce_audio_dict(monkeypatch, tmp_path):
    class DictBackend(DummyBackend):
        def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
            sr = params.get("sr", 24000)
            length = params.get("length", sr // 2)
            audio = np.ones(length, dtype=np.float32)
            return {"audio": audio, "sr": sr}

    backend = DictBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", lambda audio, *_args, **_kwargs: audio)
    request = {
        "tts_backend": "dummy",
        "script": "Hello",
        "chunks": _make_chunks()[:1],
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 0},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {"sr": 24000, "length": 12000},
        "target_sr": 24000,
    }
    result = tts_pipeline.run_tts_pipeline(request)
    audio, sr = tts_pipeline.sf.read(result.out_path, dtype="float32")
    assert sr == 24000
    assert len(audio) == 12000


def test_pipeline_inserts_comma_pause_inside_chunk(monkeypatch, tmp_path):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    monkeypatch.setattr(tts_pipeline, "_apply_post_processing", lambda audio, *_args, **_kwargs: audio)
    chunks = [
        ChunkInfo(
            segments=[SpeechSegment("text", "Bonjour, toi")],
            sentence_count=1,
            char_count=12,
            word_count=2,
            comma_count=1,
            estimated_duration=1.0,
            reason="hard",
            boundary_kind=None,
            pivot=False,
            ends_with_suspended=False,
            oversize_sentence=False,
            warnings=[],
        )
    ]
    request = {
        "tts_backend": "dummy",
        "script": "Bonjour, toi",
        "chunks": chunks,
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 500},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.0, "silence_min_ms": 0},
        "engine_params": {"length": 24000, "sr": 24000},
        "target_sr": 24000,
    }
    result = tts_pipeline.run_tts_pipeline(request)
    audio, sr = tts_pipeline.sf.read(result.out_path, dtype="float32")
    assert sr == 24000
    expected_len = 24000 + int(0.5 * 24000) + 24000
    assert abs(len(audio) - expected_len) < 100


def test_postprocessing_does_not_remove_internal_pauses(monkeypatch, tmp_path):
    backend = DummyBackend()
    monkeypatch.setattr(tts_pipeline, "get_backend", lambda _backend_id: backend)
    chunks = [
        ChunkInfo(
            segments=[SpeechSegment("text", "Bonjour, toi")],
            sentence_count=1,
            char_count=12,
            word_count=2,
            comma_count=1,
            estimated_duration=1.0,
            reason="hard",
            boundary_kind=None,
            pivot=False,
            ends_with_suspended=False,
            oversize_sentence=False,
            warnings=[],
        )
    ]
    request = {
        "tts_backend": "dummy",
        "script": "Bonjour, toi",
        "chunks": chunks,
        "out_path": str(tmp_path / "out.wav"),
        "pause_settings": {"comma_pause_ms": 500},
        "post_settings": {"zero_cross_radius_ms": 0, "fade_ms": 0, "silence_threshold": 0.001, "silence_min_ms": 0},
        "engine_params": {"length": 24000, "sr": 24000},
        "target_sr": 24000,
    }
    result = tts_pipeline.run_tts_pipeline(request)
    audio, _sr = tts_pipeline.sf.read(result.out_path, dtype="float32")
    pause_len = int(0.5 * 24000)
    pause_slice = audio[24000 : 24000 + pause_len]
    assert np.max(np.abs(pause_slice)) < 1e-6
