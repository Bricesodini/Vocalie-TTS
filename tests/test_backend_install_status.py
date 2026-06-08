from backend_install import status


def test_backend_status_unknown_engine(monkeypatch, tmp_path):
    """An engine with no manifest returns 'manifest introuvable' before
    we even look at the venv. Catches the case where someone adds a
    backend to the catalog but forgets the install manifest."""
    monkeypatch.setattr(status, "venv_dir", lambda _eid: tmp_path / "missing")
    result = status.backend_status("nonexistent-engine")
    assert result["installed"] is False
    assert result["reason"] == "manifest introuvable"


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
