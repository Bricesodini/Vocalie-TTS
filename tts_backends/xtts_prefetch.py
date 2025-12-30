#!/usr/bin/env python3
"""Prefetch XTTS model weights without running inference."""

from __future__ import annotations

import argparse
import sys


DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    try:
        from TTS.api import TTS
    except Exception as exc:
        print(f"xtts_prefetch_import_failed: {exc}", file=sys.stderr)
        return 2

    try:
        _ = TTS(model_name=args.model_id, progress_bar=False, gpu=False)
    except Exception as exc:
        print(f"xtts_prefetch_failed: {exc}", file=sys.stderr)
        return 3

    print("xtts_prefetch_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
