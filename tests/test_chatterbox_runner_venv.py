import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_CHATTERBOX_RUNNER_TEST") != "1",
    reason="Set RUN_CHATTERBOX_RUNNER_TEST=1 to enable the chatterbox runner integration test.",
)
def test_chatterbox_runner_in_venv():
    root = Path(__file__).resolve().parents[1]
    py = root / ".venvs" / "chatterbox" / "bin" / "python"
    runner = root / "tts_backends" / "chatterbox_runner.py"
    if not py.exists():
        pytest.skip(".venvs/chatterbox/bin/python missing")
    payload = {"text": "Bonjour", "out_wav_path": str(root / "output" / "smoke_runner.wav")}
    proc = subprocess.run(
        [str(py), str(runner)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(root),
        timeout=180,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data.get("ok") is True
