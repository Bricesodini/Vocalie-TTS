from __future__ import annotations

import wave

import backend.config as backend_config


def _write_silence_wav(path, duration_s=0.05, sr=22050):
    frames = int(duration_s * sr)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(b"\x00\x00" * frames)


def test_audio_edit(api_client):
    client = api_client
    wav_path = backend_config.OUTPUT_DIR / "test.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    _write_silence_wav(wav_path)

    resp = client.post(
        "/v1/audio/edit",
        json={"input_wav_path": str(wav_path), "trim_enabled": True, "normalize_enabled": True},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "edited_wav_path" in payload
    assert "metrics" in payload
