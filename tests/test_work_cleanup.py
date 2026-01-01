from pathlib import Path

import pytest

import app


def test_clean_work_dir_removes_sessions(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    work_root = repo_root / "work"
    sessions_dir = work_root / ".sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "abc").mkdir(parents=True, exist_ok=True)
    (sessions_dir / "abc" / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "BASE_DIR", repo_root)
    removed = app.clean_work_dir(work_root)
    assert removed == 1
    assert not (sessions_dir / "abc").exists()


def test_clean_work_dir_requires_repo_root(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    work_root = tmp_path / "outside"
    monkeypatch.setattr(app, "BASE_DIR", repo_root)
    with pytest.raises(ValueError):
        app.clean_work_dir(work_root)
