cat > tts_test.py <<'PY'
import argparse
from pathlib import Path
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForTextToSpeech

MODEL_ID = "Thomcles/Chatterbox-TTS-French"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Texte à synthétiser (FR).")
    p.add_argument("--out", default=str(Path.home() / "Desktop" / "chatterbox_tts_french.wav"),
                   help="Chemin du WAV de sortie.")
    p.add_argument("--sr", type=int, default=24000, help="Sample rate de sortie.")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTextToSpeech.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    inputs = processor(text=args.text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        speech = model.generate(**inputs)

    wav = speech[0].detach().cpu().numpy()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, args.sr)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
PY