from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def _parse_models(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_MODELS)
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch Qwen3-TTS model weights.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated HF model ids (default: CustomVoice + Base).",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Override assets dir used for HF cache.",
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir or os.environ.get("VOCALIE_QWEN3_ASSETS_DIR") or "").expanduser()
    if assets_dir:
        assets_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(assets_dir / ".hf"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(assets_dir / ".hf" / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(assets_dir / ".hf" / "hub"))

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"prefetch_import_failed: {exc}\n")
        return 2

    models = _parse_models(args.models or os.environ.get("VOCALIE_QWEN3_PREFETCH_MODELS"))
    if not models:
        sys.stderr.write("no_models_specified\n")
        return 3

    for model_id in models:
        snapshot_download(repo_id=model_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
