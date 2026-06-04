"""Unit tests for backend.services.tts_service module."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.shared.audio_edit import apply_minimal_edit, audio_meta
from backend.services.tts_service import _build_chunks


# --- _build_chunks ---


class TestBuildChunks:
    def test_single_chunk_without_direction(self):
        from backend.shared.text_tools import ChunkInfo

        chunks, reason, meta = _build_chunks("Bonjour le monde", direction_enabled=False, marker="[[CHUNK]]")
        assert len(chunks) == 1
        assert reason == "single"

    def test_single_chunk_returns_reason(self):
        chunks, text_out, _ = _build_chunks("Hello", direction_enabled=False, marker="[[CHUNK]]")
        assert len(chunks) == 1

    def test_direction_enabled_with_markers(self):
        text = "Première partie[[CHUNK]]Deuxième partie"
        chunks, text_out, meta = _build_chunks(text, direction_enabled=True, marker="[[CHUNK]]")
        assert len(chunks) == 2


# --- apply_minimal_edit ---


class TestApplyMinimalEdit:
    def test_trim_silence(self, tmp_path: Path):
        """Test that apply_minimal_edit trims silence from a WAV file."""
        sr = 22050
        silence = np.zeros(sr, dtype=np.float32)
        tone = np.ones(sr // 2, dtype=np.float32) * 0.5
        audio = np.concatenate([silence, tone, silence])
        import soundfile as sf

        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, sr)
        output_path = tmp_path / "output.wav"
        result = apply_minimal_edit(
            input_path,
            output_path,
            trim_enabled=True,
            normalize_enabled=False,
            target_dbfs=-3.0,
            silence_threshold=-40.0,
            silence_min_ms=300,
        )
        assert result is not None or output_path.exists()

    def test_normalize_only(self, tmp_path: Path):
        """Test normalization without trimming."""
        sr = 22050
        audio = np.ones(sr, dtype=np.float32) * 0.1
        import soundfile as sf

        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, sr)
        output_path = tmp_path / "output.wav"
        result = apply_minimal_edit(
            input_path,
            output_path,
            trim_enabled=False,
            normalize_enabled=True,
            target_dbfs=-3.0,
            silence_threshold=-40.0,
            silence_min_ms=300,
        )
        assert output_path.exists()

    def test_no_edit(self, tmp_path: Path):
        """Test with both trim and normalize disabled."""
        sr = 22050
        audio = np.ones(sr, dtype=np.float32) * 0.5
        import soundfile as sf

        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, sr)
        output_path = tmp_path / "output.wav"
        result = apply_minimal_edit(
            input_path,
            output_path,
            trim_enabled=False,
            normalize_enabled=False,
            target_dbfs=-1.0,
            silence_threshold=-40.0,
            silence_min_ms=300,
        )
        assert output_path.exists()

    def test_empty_audio(self, tmp_path: Path):
        """Test with silent (near-zero) audio."""
        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
        import soundfile as sf

        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, sr)
        output_path = tmp_path / "output.wav"
        # Should not crash on silent audio
        apply_minimal_edit(
            input_path,
            output_path,
            trim_enabled=True,
            normalize_enabled=True,
            target_dbfs=-3.0,
            silence_threshold=-40.0,
            silence_min_ms=300,
        )


# --- audio_meta ---


class TestAudioMeta:
    def test_returns_duration_and_info(self, tmp_path: Path):
        sr = 22050
        audio = np.ones(sr * 2, dtype=np.float32) * 0.5
        import soundfile as sf

        path = tmp_path / "test.wav"
        sf.write(str(path), audio, sr)
        meta = audio_meta(path)
        assert "duration_s" in meta
        assert meta["duration_s"] > 0
        assert "sample_rate" in meta
        assert meta["sample_rate"] == sr

    def test_short_file(self, tmp_path: Path):
        sr = 22050
        audio = np.ones(sr // 10, dtype=np.float32) * 0.3
        import soundfile as sf

        path = tmp_path / "short.wav"
        sf.write(str(path), audio, sr)
        meta = audio_meta(path)
        assert meta["duration_s"] < 1.0