import json

from tts_backends import chatterbox_backend as cb


def test_chatterbox_runner_returns_ok_json(monkeypatch, tmp_path):
    py_path = tmp_path / "python"
    runner_path = tmp_path / "chatterbox_runner.py"
    py_path.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")

    class DummyProc:
        returncode = 0
        stdout = json.dumps(
            {
                "ok": True,
                "out_path": "/tmp/out.wav",
                "duration_s": 1.23,
                "logs": ["ok"],
            }
        )
        stderr = ""

    monkeypatch.setattr(cb, "python_path", lambda _eid: py_path)
    monkeypatch.setattr(cb, "_runner_path", lambda: runner_path)
    monkeypatch.setattr(cb.subprocess, "run", lambda *args, **kwargs: DummyProc())

    result = cb._run_chatterbox_runner({"text": "hello", "out_wav_path": "/tmp/out.wav"})
    assert result["ok"] is True
    assert result["out_path"] == "/tmp/out.wav"
