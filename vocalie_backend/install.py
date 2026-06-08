"""Install / verify the local Python venv and runtime dependencies.

We don't install the heavy per-backend venvs here (chatterbox, qwen3,
audiosr) — those have their own scripts under scripts/ and are
intentionally opt-in. The job of this module is to make sure
``vocalie-backend start`` can find a working ``.venv`` with
``uvicorn`` importable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from vocalie_backend.config import VENV_DIR


REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"


def _venv_python() -> Path:
    return VENV_DIR / "bin" / "python"


def venv_exists() -> bool:
    return _venv_python().exists()


def create_venv() -> None:
    import venv
    if venv_exists():
        return
    print(f"creating venv at {VENV_DIR}")
    venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(str(VENV_DIR))


def pip_install(args: List[str]) -> int:
    cmd = [str(_venv_python()), "-m", "pip", "install", *args]
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def check_imports() -> Tuple[bool, List[str]]:
    """Return (ok, missing_modules)."""
    code = (
        "import importlib, sys\n"
        "missing = [m for m in ('fastapi', 'uvicorn', 'pydantic', 'httpx', 'soundfile', 'librosa') "
        "if not importlib.util.find_spec(m)]\n"
        "if missing: print(' '.join(missing))\n"
        "sys.exit(0 if not missing else 1)\n"
    )
    proc = subprocess.run(
        [str(_venv_python()), "-c", code],
        capture_output=True,
        text=True,
    )
    missing = proc.stdout.strip().split() if proc.stdout.strip() else []
    return (proc.returncode == 0, missing)


def install(*, upgrade: bool = False) -> int:
    """Create venv if missing, install requirements, verify imports."""
    if shutil.which("python3") is None and shutil.which("python") is None:
        print("error: no python3/python on PATH", file=sys.stderr)
        return 4
    create_venv()
    cmd = ["-r", str(REQUIREMENTS_FILE)]
    if upgrade:
        cmd = ["--upgrade", *cmd]
    rc = pip_install(cmd)
    if rc != 0:
        return rc
    ok, missing = check_imports()
    if not ok:
        print(f"missing imports after install: {missing}", file=sys.stderr)
        return 1
    print("install OK")
    return 0
