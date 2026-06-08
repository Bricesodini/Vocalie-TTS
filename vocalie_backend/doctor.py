"""Environment sanity check — replaces scripts/doctor.sh."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DoctorResult:
    ok: int = 0
    warn: int = 0
    fail: int = 0
    issues: List[str] = field(default_factory=list)

    def to_human(self) -> str:
        lines = [f"[ok]   {n}" if s == "ok" else f"[{s}] {n}"
                 for n, s in self._rows()]
        lines.append("")
        lines.append(
            f"summary: {self.ok} ok, {self.warn} warn, {self.fail} fail"
        )
        return "\n".join(lines)

    def _rows(self):
        for n in self.issues:
            yield n, "ok" if n.startswith("ok ") else "warn" if n.startswith("warn ") else "fail"

    def exit_code(self) -> int:
        return 0 if self.fail == 0 else 1


def _check(result: DoctorResult, name: str, present: bool) -> bool:
    if present:
        result.ok += 1
        result.issues.append(f"ok {name}")
        return True
    result.fail += 1
    result.issues.append(f"fail {name}")
    return False


def run() -> DoctorResult:
    r = DoctorResult()
    _check(r, "python3", shutil.which("python3") is not None)
    _check(r, "ffmpeg", shutil.which("ffmpeg") is not None)
    _check(r, "sox", shutil.which("sox") is not None)
    _check(r, "node", shutil.which("node") is not None)
    _check(r, ".venv present", (shutil.which("python3") and __import__("pathlib").Path(".venv").exists()))
    return r
