# 🎙️ Chatterbox TTS FR

## Présentation

Chatterbox TTS FR est une interface locale pour produire des voix off en français avec un pipeline simple, stable et reproductible.
Le cœur du produit est la génération TTS ; l’édition audio avancée a été supprimée pour éviter les comportements implicites.

Objectifs V2 :

- génération fiable (multi-moteurs),
- chunking explicite et déterministe (aucun découpage automatique implicite),
- post‑traitement minimal (optionnel),
- sorties propres dans `./output/`.

## Stack actuelle (résumé)

- UI Gradio locale (macOS friendly)
- moteurs : Chatterbox, XTTS v2, Piper, Bark
- chunking **manuel** via marqueur `[[CHUNK]]` (mode Direction)
- montage inter‑chunk optionnel (silence) pour Chatterbox
- édition minimale **optionnelle** : trim début/fin + normalisation

## Moteurs supportés

- **Chatterbox** (FR + multilangue)
- **XTTS v2** (voice cloning, ref audio obligatoire)
- **Piper** (offline rapide, voix à installer)
- **Bark** (créatif, expérimental)

L’UI est capability‑driven : seuls les paramètres supportés par le backend sont visibles et envoyés.
Par exemple, les paramètres de référence vocale ou de segmentation ne sont affichés que pour les moteurs qui les supportent.

## Pipeline de génération (V2)

1. Texte → normalisation + lexique FR (si auto‑ajustement activé)
2. Chunking **manuel** (Mode Direction) :
   - `[[CHUNK]]` = split explicite
   - si Direction activée sans marqueur → **chunk unique**
   - aucun découpage automatique caché
3. Synthèse chunk‑par‑chunk
4. Assemblage global
   - Chatterbox : option **Blanc entre chunks (ms)** (post-assemblage, non moteur)
   - autres moteurs : gap forcé à 0

Aucune insertion de pause automatique, aucune logique d’édition audio avancée.

## Édition audio minimale (optionnelle)

L’édition est **désactivée par défaut** et doit être explicitement activée par l’utilisateur.

L’édition ne touche **jamais** le RAW. Elle est activée manuellement et ne propose que :

- trim début/fin (détection de faible énergie)
- normalisation (peak vers dBFS cible)

Le résultat est un fichier édité **séparé** dans `./output/` (suffix `_edit_01`, `_edit_02`, etc.).

## Sorties

- RAW immuable stocké dans `work/.sessions/.../takes/...`
- Export RAW et fichier édité exportés dans `./output/`
- Pas d’écriture hors du projet

Le dossier `work/` est nettoyé au démarrage (sauf `VOCALIE_KEEP_WORK=1`).

## Quickstart

```bash
cd /Users/bricesodini/01_ai-stack/Chatterbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Ouvrez ensuite http://127.0.0.1:7860

## Prérequis

- macOS (Apple Silicon recommandé)
- Python 3.11
- **ffmpeg** (recommandé, requis pour XTTS si la référence n’est pas en WAV)

```bash
brew install ffmpeg
```

## Dépendances Python

- chatterbox‑tts
- torch (MPS recommandé)
- gradio
- librosa
- soundfile
- huggingface_hub
- safetensors
- numpy
- pytest

## Structure projet

```
Chatterbox/
├── app.py            # UI Gradio (entrée principale)
├── refs.py           # gestion des Ref_audio/
├── text_tools.py     # outils texte + chunking manuel
├── tts_pipeline.py   # pipeline TTS + assemblage
├── tts_engine.py     # wrappers spécifiques
├── output_paths.py   # nommage fichiers
├── state_manager.py  # state + presets
├── tts_backends/     # backends modulaires (Chatterbox, XTTS, Piper, Bark)
├── Ref_audio/        # références vocales
├── output/           # exports WAV (RAW + édités)
├── work/             # sessions temporaires
├── presets/          # presets JSON
└── tests/            # tests pytest
```

## Variables d’environnement (optionnel)

- `CHATTERBOX_REF_DIR` : dossier de références
- `CHATTERBOX_OUT_DIR` : dossier de sortie par défaut
- `GRADIO_SERVER_PORT` : port Gradio (par défaut 7860)
- `VOCALIE_KEEP_WORK=1` : désactive le nettoyage de `work/` au démarrage (nom historique)

## Workflow recommandé

1. Collez votre texte dans **Préparation**
2. (Optionnel) Ajustez le texte / durée
3. **Direction** : chargez un snapshot, insérez `[[CHUNK]]` si besoin
4. Lancez **Générer**
5. (Optionnel) Activez l’édition minimale et générez un fichier édité

---

Pour toute demande de modification, gardez la règle d’or :
**si le bénéfice n’est pas immédiatement audible, la fonctionnalité n’a pas sa place en V2.**
