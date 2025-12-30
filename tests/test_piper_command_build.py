from pathlib import Path

from tts_backends.piper_backend import PiperBackend


def test_piper_command_build():
    py = Path("/tmp/venv/bin/python")
    runner = Path("/repo/tts_backends/piper_runner.py")
    info = {"voice_id": "voice", "model_dir": "/models/voice"}
    cmd = PiperBackend.build_command(py, runner, "Hello", "/tmp/out.wav", info, "fr", 1.1)
    assert cmd[0] == str(py)
    assert cmd[1] == str(runner)
    assert "--text" in cmd
    assert "Hello" in cmd
    assert "--out_wav" in cmd
    assert "/tmp/out.wav" in cmd
    assert "--voice" in cmd
    assert "voice" in cmd
    assert "--model_dir" in cmd
    assert "/models/voice" in cmd
    assert "--lang" in cmd
    assert "--length_scale" in cmd
