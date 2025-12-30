from backend_install import installer


def test_run_install_calls_helpers(monkeypatch):
    calls = []
    subprocess_calls = []

    monkeypatch.setattr(installer, "create_venv", lambda *args, **kwargs: calls.append("venv"))
    monkeypatch.setattr(installer, "pip_install", lambda *args, **kwargs: calls.append("pip"))
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append(args) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(
        installer,
        "get_manifest",
        lambda _eid: type(
            "M",
            (),
            {
                "python": "python3.11",
                "pip_packages": ["pkg"],
                "engine_id": "xtts",
                "post_install_checks": [],
            },
        )(),
    )

    ok, logs = installer.run_install("xtts")
    assert ok is True
    assert "venv" in calls
    assert "pip" in calls
    assert any("Installation terminée." in line for line in logs)


def test_xtts_installer_prefetch_calls_runner(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(installer, "create_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(installer, "pip_install", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        installer,
        "get_manifest",
        lambda _eid: type(
            "M",
            (),
            {
                "python": "python3.11",
                "pip_packages": ["pkg"],
                "engine_id": "xtts",
                "post_install_checks": [],
            },
        )(),
    )
    monkeypatch.setattr(installer, "python_path", lambda _eid: tmp_path / "bin" / "python")
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    (tmp_path / "bin").mkdir(parents=True)
    (tmp_path / "bin" / "python").write_text("", encoding="utf-8")
    (tmp_path / "tts_backends").mkdir(parents=True)
    (tmp_path / "tts_backends" / "xtts_prefetch.py").write_text("print('ok')", encoding="utf-8")

    def fake_run(args, **_kwargs):
        calls.append(args)
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    ok, _logs = installer.run_install("xtts")
    assert ok is True
    assert any("xtts_prefetch.py" in " ".join(call) for call in calls)
