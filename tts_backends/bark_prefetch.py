from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / ".assets" / "bark"
    assets_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(assets_dir))
    os.environ.setdefault("HF_HOME", str(assets_dir / ".hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(assets_dir / ".hf" / "hub"))
    os.environ.setdefault("SUNO_ENABLE_MPS", "False")

    if os.environ.get("VOCALIE_BARK_SMALL_MODELS") in {"1", "true", "True", "yes", "YES"}:
        os.environ.setdefault("SUNO_USE_SMALL_MODELS", "True")

    from bark import preload_models

    preload_models()
    print("bark prefetch ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
