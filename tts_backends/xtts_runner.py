#!/usr/bin/env python3
"""XTTS v2 runner (invoked inside the XTTS venv)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np


DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


def force_cpu_for_platform(system: str, machine: str) -> bool:
    return system == "Darwin" and machine in {"arm64", "aarch64"}


def _tts_init_kwargs(force_cpu: bool) -> dict:
    if force_cpu:
        return {"gpu": False}
    return {"gpu": False}


def _map_language(lang: str | None) -> str | None:
    if not lang:
        return None
    return lang.split("-")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker_wav", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--meta_json", default=None)
    args = parser.parse_args()

    force_cpu = force_cpu_for_platform(platform.system(), platform.machine())
    device = "cpu"
    try:
        from TTS.api import TTS
    except Exception as exc:
        print(f"xtts_import_failed: {exc}", file=sys.stderr)
        return 2

    try:
        tts = TTS(model_name=args.model_id, progress_bar=False, **_tts_init_kwargs(force_cpu))
        if hasattr(tts, "to"):
            try:
                tts = tts.to(device)
            except Exception:
                pass
    except Exception as exc:
        print(f"xtts_load_failed: {exc}", file=sys.stderr)
        return 3

    language = _map_language(args.language)
    segments = None
    segment_lengths = []
    kwargs = {
        "text": args.text,
        "speaker_wav": args.speaker_wav,
        "language": language,
        "file_path": args.out_path,
    }
    if args.speed is not None:
        kwargs["speed"] = float(args.speed)

    try:
        tts.tts_to_file(**kwargs)
    except TypeError:
        kwargs.pop("speed", None)
        tts.tts_to_file(**kwargs)
    except Exception as exc:
        print(f"xtts_synthesis_failed: {exc}", file=sys.stderr)
        return 4

    sr = None
    try:
        sr = getattr(tts, "synthesizer", None)
        if sr is not None:
            sr = getattr(tts.synthesizer, "output_sample_rate", None)
    except Exception:
        sr = None

    try:
        synthesizer = getattr(tts, "synthesizer", None)
        split_fn = None
        if synthesizer is not None:
            for name in ("split_sentences", "split_text"):
                if hasattr(synthesizer, name):
                    split_fn = getattr(synthesizer, name)
                    break
        if split_fn is not None:
            try:
                segments = split_fn(args.text, language=language)
            except TypeError:
                segments = split_fn(args.text)
            if not isinstance(segments, list):
                segments = None
    except Exception:
        segments = None

    if segments:
        for segment in segments:
            seg_kwargs = {
                "text": segment,
                "speaker_wav": args.speaker_wav,
                "language": language,
            }
            if args.speed is not None:
                seg_kwargs["speed"] = float(args.speed)
            try:
                seg_kwargs["split_sentences"] = False
                audio = tts.tts(**seg_kwargs)
            except TypeError:
                seg_kwargs.pop("split_sentences", None)
                try:
                    audio = tts.tts(**seg_kwargs)
                except TypeError:
                    seg_kwargs.pop("speed", None)
                    audio = tts.tts(**seg_kwargs)
            audio = np.asarray(audio)
            segment_lengths.append(int(audio.shape[0]))

    if args.meta_json:
        meta = {
            "sr": sr,
            "device": device,
            "forced_cpu": force_cpu,
            "model_id": args.model_id,
        }
        if segments:
            boundaries = []
            cursor = 0
            for length in segment_lengths[:-1]:
                cursor += int(length)
                boundaries.append(int(cursor))
            meta["segments"] = segments
            meta["segment_lengths_samples"] = segment_lengths
            meta["segment_boundaries_samples"] = boundaries
        try:
            Path(args.meta_json).write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
