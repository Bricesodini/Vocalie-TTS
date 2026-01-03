from backend_install import status


def test_backend_status_missing_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(status, "venv_dir", lambda _eid: tmp_path / "missing")
    result = status.backend_status("xtts")
    assert result["installed"] is False
    assert result["reason"] == "venv manquante"


def test_backend_status_import_ok(monkeypatch, tmp_path):
    venv_root = tmp_path / "xtts"
    venv_root.mkdir(parents=True)
    py_path = venv_root / "bin" / "python"
    py_path.parent.mkdir(parents=True)
    py_path.write_text("", encoding="utf-8")
    model_dir = tmp_path / ".assets" / "xtts" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
    model_dir.mkdir(parents=True)

    class DummyProc:
        returncode = 0

    monkeypatch.setattr(status, "venv_dir", lambda _eid: venv_root)
    monkeypatch.setattr(status, "python_path", lambda _eid: py_path)
    monkeypatch.setattr(status.subprocess, "run", lambda *args, **kwargs: DummyProc())
    monkeypatch.setattr(status, "XTTS_ASSETS_DIR", tmp_path / ".assets" / "xtts")

    monkeypatch.setattr(status, "get_manifest", lambda _eid: type("M", (), {"import_probes": ["TTS"]})())
    result = status.backend_status("xtts")
    assert result["installed"] is True
    assert result["reason"] == "OK"
    assert result.get("model_downloaded") is True


def test_xtts_status_model_missing(monkeypatch, tmp_path):
    venv_root = tmp_path / "xtts"
    venv_root.mkdir(parents=True)
    py_path = venv_root / "bin" / "python"
    py_path.parent.mkdir(parents=True)
    py_path.write_text("", encoding="utf-8")

    class DummyProc:
        returncode = 0

    monkeypatch.setattr(status, "venv_dir", lambda _eid: venv_root)
    monkeypatch.setattr(status, "python_path", lambda _eid: py_path)
    monkeypatch.setattr(status.subprocess, "run", lambda *args, **kwargs: DummyProc())
    monkeypatch.setattr(status, "XTTS_ASSETS_DIR", tmp_path / ".assets" / "xtts")
    monkeypatch.setattr(status, "get_manifest", lambda _eid: type("M", (), {"import_probes": ["TTS"]})())
    result = status.backend_status("xtts")
    assert result["installed"] is True
    assert result.get("model_downloaded") is False


def test_backend_status_chatterbox_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(status, "venv_dir", lambda _eid: tmp_path / "missing")
    monkeypatch.setattr(status, "get_manifest", lambda _eid: type("M", (), {"import_probes": ["chatterbox"]})())
    result = status.backend_status("chatterbox")
    assert result["installed"] is False
    assert result["reason"] == "venv manquante"


def test_backend_status_chatterbox_installed(monkeypatch, tmp_path):
    venv_root = tmp_path / "chatterbox"
    venv_root.mkdir(parents=True)
    py_path = venv_root / "bin" / "python"
    py_path.parent.mkdir(parents=True)
    py_path.write_text("", encoding="utf-8")

    class DummyProc:
        returncode = 0

    monkeypatch.setattr(status, "venv_dir", lambda _eid: venv_root)
    monkeypatch.setattr(status, "python_path", lambda _eid: py_path)
    monkeypatch.setattr(status.subprocess, "run", lambda *args, **kwargs: DummyProc())
    monkeypatch.setattr(status, "get_manifest", lambda _eid: type("M", (), {"import_probes": ["chatterbox"]})())
    result = status.backend_status("chatterbox")
    assert result["installed"] is True
    assert result["reason"] == "OK"
