from __future__ import annotations

import os
import shutil
from pathlib import Path


def clean_work_dir(work_root: Path) -> int:
    if os.environ.get("VOCALIE_KEEP_WORK") == "1":
        return 0
    work_root = Path(work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    sessions_dir = work_root / ".sessions"
    tmp_dir = work_root / ".tmp"
    tmp_dir_alt = work_root / "tmp"
    removed_sessions = 0
    if sessions_dir.exists():
        for entry in sessions_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed_sessions += 1
            elif entry.is_file():
                entry.unlink(missing_ok=True)
                removed_sessions += 1
    for tmp_path in (tmp_dir, tmp_dir_alt):
        if tmp_path.exists():
            for entry in tmp_path.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                elif entry.is_file():
                    entry.unlink(missing_ok=True)
    return removed_sessions
