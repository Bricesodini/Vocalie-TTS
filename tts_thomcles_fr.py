from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from chatterbox.tts import ChatterboxTTS
import librosa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

BASE_REPO = "ResembleAI/chatterbox"          # modèle de base (ve/s3gen/tokenizer/conds + t3)
FR_REPO   = "Thomcles/Chatterbox-TTS-French" # fine-tune FR (t3_cfg.safetensors)

REF_AUDIO = "/Users/bricesodini/01_ai-stack/Chatterbox/Ref_audio/essai.m4a"

text = (
    "Nouvelle édition, nouvelles émotions !\n"
    "Laissez-vous émerveiller par le Festival Lumières Sauvages.\n"
    "Une expérience magique de sons et lumières,\n "
    "à travers la nature et les continents."
)

# IMPORTANT: ChatterboxTTS.from_pretrained() utilise un REPO_ID interne.
# On force la variable globale REPO_ID dans le module chatterbox.tts avant d'appeler from_pretrained.
import chatterbox.tts as tts_mod
tts_mod.REPO_ID = BASE_REPO

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# 1) Charger le modèle de base complet (t3 + s3gen + ve + tokenizer + conds si dispo)
tts = ChatterboxTTS.from_pretrained(device)

# 2) Remplacer uniquement le T3 par le fine-tune français
fr_t3_path = hf_hub_download(repo_id=FR_REPO, filename="t3_cfg.safetensors")
fr_t3_state = load_file(fr_t3_path)
if "model" in fr_t3_state:
    fr_t3_state = fr_t3_state["model"][0]

tts.t3.load_state_dict(fr_t3_state)
tts.t3.to(device).eval()

# 3) Générer
wav_t = tts.generate(
    text,
    audio_prompt_path=REF_AUDIO,
    exaggeration=0.5,   # plus haut = plus expressif
    cfg_weight=0.6,      # plus haut = plus "tenu"
    temperature=0.5,      # plus bas = plus stable
    repetition_penalty=1.35
)

# --- Sauvegarde "brute" ---
out_dir = Path.home() / "Desktop"
out_dir.mkdir(parents=True, exist_ok=True)

out_raw = out_dir / "thomcles_chatterbox_fr_raw.wav"
y = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

sf.write(out_raw, y, tts.sr)
raw_dur = len(y) / tts.sr
print(f"Saved raw: {out_raw} ({raw_dur:.2f}s)")
