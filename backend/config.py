from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

work_env = os.environ.get("VOCALIE_WORK_DIR")
WORK_DIR = Path(work_env).expanduser() if work_env else BASE_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

output_env = os.environ.get("VOCALIE_OUTPUT_DIR") or os.environ.get("CHATTERBOX_OUT_DIR")
OUTPUT_DIR = Path(output_env).expanduser() if output_env else BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESETS_DIR = BASE_DIR / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_META_DIR = OUTPUT_DIR / ".assets"
ASSETS_META_DIR.mkdir(parents=True, exist_ok=True)
