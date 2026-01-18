from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf


def _to_numpy(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    if hasattr(waveform, "numpy"):
        return waveform.numpy()
    return np.asarray(waveform)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 48000) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def _prepare_input(path: str, input_cutoff: int, tmp_dir: Path) -> Path:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 48000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    if input_cutoff and input_cutoff > 0:
        from audiosr.lowpass import lowpass

        audio = lowpass(audio.astype(np.float32), highcut=int(input_cutoff), fs=sr, order=8, _type="butter")

    tmp_path = tmp_dir / "input.wav"
    _write_wav(tmp_path, audio, sample_rate=sr)
    return tmp_path


def _run_model(model, input_path: str, *, seed: int, guidance_scale: float, ddim_steps: int, chunk_size: int, overlap: int):
    from audiosr import super_resolution, super_resolution_long_audio

    chunk_duration_s = float(chunk_size) / 48000.0
    overlap_duration_s = float(overlap) / 48000.0
    if chunk_size > 0 and overlap > 0 and chunk_duration_s > overlap_duration_s:
        return super_resolution_long_audio(
            model,
            input_path,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            chunk_duration_s=chunk_duration_s,
            overlap_duration_s=overlap_duration_s,
        )
    return super_resolution(
        model,
        input_path,
        seed=seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--overlap", type=int, default=1024)
    parser.add_argument("--multiband_ensemble", type=int, default=0)
    parser.add_argument("--input_cutoff", type=int, default=8000)
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated*",
        category=UserWarning,
    )

    import torch
    from audiosr import build_model

    torch.set_float32_matmul_precision("high")
    model = build_model(model_name="basic", device="auto")

    with tempfile.TemporaryDirectory(prefix="audiosr_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        base_input = _prepare_input(args.input_path, input_cutoff=0, tmp_dir=tmp_root_path)
        base_waveform = _run_model(
            model,
            str(base_input),
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        base_audio = _to_numpy(base_waveform)

        final_audio = base_audio
        if args.multiband_ensemble and args.input_cutoff > 0:
            filtered_dir = tmp_root_path / "filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            filtered_input = _prepare_input(args.input_path, input_cutoff=args.input_cutoff, tmp_dir=filtered_dir)
            filtered_waveform = _run_model(
                model,
                str(filtered_input),
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                ddim_steps=args.ddim_steps,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            filtered_audio = _to_numpy(filtered_waveform)
            min_len = min(final_audio.shape[-1], filtered_audio.shape[-1])
            final_audio = 0.5 * (final_audio[..., :min_len] + filtered_audio[..., :min_len])

    _write_wav(Path(args.output_path), final_audio, sample_rate=48000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
