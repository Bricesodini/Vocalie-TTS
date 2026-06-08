"""Tail the backend log file."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from vocalie_backend.config import BACKEND_LOG_FILE


def tail(follow: bool = False, lines: int = 100) -> int:
    if not BACKEND_LOG_FILE.exists():
        print(f"(no log file at {BACKEND_LOG_FILE})")
        return 0
    with BACKEND_LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        data = fh.readlines()
    for line in data[-lines:]:
        print(line.rstrip())
    if follow:
        try:
            with BACKEND_LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
                # Seek to current end-of-file (already read above).
                fh.seek(0, 2)
                while True:
                    line = fh.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        time.sleep(0.3)
        except KeyboardInterrupt:
            return 0
    return 0
