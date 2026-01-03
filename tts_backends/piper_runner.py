#!/usr/bin/env python3
"""Minimal Piper CLI runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--out_wav", required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--length_scale", type=float, default=None)
    args = parser.parse_args()

    if not args.model_dir or not args.voice:
        print("missing --model_dir or --voice", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{args.voice}.onnx"
    config_path = model_dir / f"{args.voice}.onnx.json"
    if not model_path.exists() or not config_path.exists():
        print("model files missing", file=sys.stderr)
        return 3

    piper_bin = Path(sys.executable).parent / "piper"
    if not piper_bin.exists():
        print("piper binary not found in venv", file=sys.stderr)
        return 4

    try:
        subprocess.run(
            [
                str(piper_bin),
                "--model",
                str(model_path),
                "--config",
                str(config_path),
                "--output_file",
                str(Path(args.out_wav)),
            ]
            + (["--length_scale", str(args.length_scale)] if args.length_scale else []),
            input=args.text,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"piper failed: {exc}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
