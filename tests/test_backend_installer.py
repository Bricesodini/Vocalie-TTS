from backend_install import installer


def test_run_install_calls_helpers(monkeypatch):
    calls = []

    monkeypatch.setattr(installer, "create_venv", lambda *args, **kwargs: calls.append("venv"))
    monkeypatch.setattr(installer, "pip_install", lambda *args, **kwargs: calls.append("pip"))
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
