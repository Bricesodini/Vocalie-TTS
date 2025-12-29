from pathlib import Path

import queue
import pytest

import app


def test_auto_apply_before_generate(monkeypatch, tmp_path):
    def fake_worker(payload, result_queue):
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"")
        chunk_count = len(payload.get("chunks") or [])
        meta = {
            "chunks": chunk_count,
            "durations": [0.5 for _ in range(chunk_count)],
            "retries": [False for _ in range(chunk_count)],
            "boundary_kinds": [None for _ in range(chunk_count)],
            "boundary_pauses": [0 for _ in range(chunk_count)],
            "punct_fixes": [None for _ in range(chunk_count)],
            "pause_events": [[] for _ in range(chunk_count)],
            "total_duration": 0.5,
        }
        result_queue.put({"status": "ok", "meta": meta})

    class DummyProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self._alive = False

        def start(self):
            self._alive = True
            self._target(*self._args)
            self._alive = False

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

    class DummyContext:
        def Queue(self):
            return queue.Queue()

        def Process(self, target, args):
            return DummyProcess(target, args)

    monkeypatch.setattr(app, "_generate_longform_worker", fake_worker)
    monkeypatch.setattr(app.mp, "get_context", lambda *_args, **_kwargs: DummyContext())
    chunk_state = {"applied": False, "chunks": [], "signature": None}
    result = app.handle_generate(
        text="Bonjour\nMerci beaucoup",
        ref_name=None,
        out_dir=str(tmp_path),
        user_filename="test",
        add_timestamp=False,
        tts_model_mode="fr_finetune",
        tts_language="fr-FR",
        multilang_cfg_weight=0.5,
        comma_pause_ms=200,
        period_pause_ms=350,
        semicolon_pause_ms=300,
        colon_pause_ms=300,
        dash_pause_ms=250,
        newline_pause_ms=300,
        min_words_per_chunk=2,
        max_words_without_terminator=10,
        max_est_seconds_per_chunk=10.0,
        verbose_logs=False,
        exaggeration=0.5,
        cfg_weight=0.6,
        temperature=0.5,
        repetition_penalty=1.35,
        fade_ms=50,
        zero_cross_radius_ms=10,
        silence_threshold=0.002,
        silence_min_ms=20,
        chunk_state=chunk_state,
        log_text=None,
    )
    _, _, chunk_preview, chunk_status, updated_state, log_text = result
    assert "auto_apply_before_generate" in log_text
    assert chunk_status == "Etat: appliqué"
    assert updated_state["applied"] is True
    assert chunk_preview
