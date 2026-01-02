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

## Prérequis

- macOS (Apple Silicon recommandé)
- Python 3.11
- Node.js >= 20
- **ffmpeg** (recommandé, requis pour XTTS si la référence n’est pas en WAV)

```bash
brew install ffmpeg
```

## Structure repo (résumé)

- Backend API + cockpit : à la racine du repo
- Frontend Next.js : `./frontend`

## Ordre de lancement (recommandé)

1. Démarrer l’API backend
2. Démarrer le frontend
3. (Optionnel) Démarrer le cockpit Gradio

## Quickstart (API + Frontend)

### Backend (API) — installation minimale (runtime)

```bash
cd Chatterbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## Installation from scratch

### Quickstart (bootstrap)

```bash
./scripts/bootstrap.sh min   # core + chatterbox
./scripts/bootstrap.sh std   # min + xtts + piper
./scripts/bootstrap.sh clean # supprime .venv et .venvs
./scripts/smoke.sh
```

### Manual install (fallback)

Core (API + cockpit Gradio) :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Chatterbox (venv dédié) :

```bash
python3.11 -m venv .venvs/chatterbox
source .venvs/chatterbox/bin/activate
pip install -U pip setuptools wheel
export PIP_NO_BUILD_ISOLATION=1
pip install "numpy<1.26,>=1.24"
pip install -r requirements-chatterbox.txt
```

XTTS / Piper (via API core) :

```bash
source .venv/bin/activate
python -c "from backend_install.installer import run_install; print(run_install('xtts'))"
python -c "from backend_install.installer import run_install; print(run_install('piper'))"
```

### Installer moteur Chatterbox isolé

Chatterbox s’exécute dans un venv dédié (`.venvs/chatterbox`), séparé du core.

```bash
python3.11 -m venv .venvs/chatterbox
source .venvs/chatterbox/bin/activate
pip install -U pip setuptools wheel
export PIP_NO_BUILD_ISOLATION=1
pip install "numpy<1.26,>=1.24"
pip install -r requirements-chatterbox.txt
```

### Frontend (Next.js)

```bash
cd frontend
npm ci
npm run dev
```

Ouvrez ensuite http://localhost:3000

## Quickstart (Gradio cockpit)

```bash
cd Chatterbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python ui_gradio/cockpit.py
```

Ouvrez ensuite http://127.0.0.1:7860

## UI CSS (base skinnable)

- CSS principal : `ui-base.css` (chargé par `app.py` via `load_ui_css()`).
- Debug : activez les toggles **CSS debug** / **CSS debug colors** dans l’UI.
- Itération Safari : Inspecteur → Styles → éditez en live, puis reportez dans `ui-base.css`.
- Si le CSS ne semble pas appliqué, faites un hard refresh (⌘⇧R).

## Dépendances Python

Runtime : `requirements.txt` (API + cockpit Gradio)

Dev/tests : `requirements-dev.txt`

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
- `NEXT_PUBLIC_API_BASE` : base API pour le frontend (optionnel)

Ports par défaut :

- API : 8000
- Frontend : 3000
- Gradio cockpit : 7860

Changer les ports :

- API : `uvicorn backend.app:app --reload --port 8000`
- Frontend : `PORT=3000 npm run dev`
- Gradio : `GRADIO_SERVER_PORT=7860 python ui_gradio/cockpit.py`

## Frontend: priorité des variables

- Si `NEXT_PUBLIC_API_BASE` est défini, le frontend appelle directement cette URL.
- Sinon, il utilise le proxy `/v1` défini dans `frontend/next.config.ts`.

## Usage LAN (optionnel)

Pour accéder depuis un iPhone / autre machine :

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

macOS peut afficher un prompt firewall au premier lancement.

## Architecture des environnements

- `.venv` (core) : API + cockpit Gradio + deps communes.
- `.venvs/chatterbox` : environnement isolé Chatterbox (invocation via subprocess).
- `.venvs/xtts` : environnement isolé XTTS (invocation via subprocess).
- `.venvs/piper` : environnement isolé Piper (invocation via subprocess).

Le backend appelle les moteurs via le Python de `.venvs/*` :

- Chatterbox : `tts_backends/chatterbox_backend.py` appelle `backend_install.paths.python_path(\"chatterbox\")`
  puis lance `tts_backends/chatterbox_runner.py` via subprocess.
- XTTS : `tts_backends/xtts_backend.py` appelle `backend_install.paths.python_path(\"xtts\")`
  puis lance `tts_backends/xtts_runner.py` via subprocess.
- Piper : `tts_backends/piper_backend.py` appelle `backend_install.paths.python_path(\"piper\")`
  puis lance `tts_backends/piper_runner.py` via subprocess.

## Smoke tests moteurs

- Chatterbox (si venv installé) :
  ```bash
  echo '{"text":"Bonjour","out_wav_path":"./output/chatterbox_smoke.wav"}' \
    | ./.venvs/chatterbox/bin/python tts_backends/chatterbox_runner.py
  ```
- XTTS (si venv installé) :
  ```bash
  ./.venvs/xtts/bin/python tts_backends/xtts_runner.py --help
  ```
- Piper (si venv installé) :
  ```bash
  ./.venvs/piper/bin/python tts_backends/piper_runner.py --help
  ```

## Troubleshooting

- `400 engine_required` sur `/v1/tts/voices` : l’engine n’est pas envoyé. Vérifiez que l’UI passe `engine=<id>`.
- XTTS sur macOS : le runner force le CPU pour éviter les instabilités GPU (comportement attendu).
- `SWC lockfile patched` / `Failed to patch lockfile` :
  ```bash
  cd frontend
  rm -rf node_modules .next
  npm ci
  npm install
  npm run dev
  ```
- Pourquoi `npm install` après `npm ci` ?
  - Next peut patcher le lockfile SWC au premier lancement, `npm install` met le lockfile à jour.
- `Module not found` (lucide-react / class-variance-authority / clsx / @/lib/utils) :
  ```bash
  cd frontend
  npm ci
  ```
- Warning `pkg_resources is deprecated` (perth_net) : warning non bloquant.
- Warning lockfile root : supprimez `~/package-lock.json` s’il existe.
- `pkuseg build isolation / numpy` :
  - Certaines dépendances de Chatterbox échouent si `numpy` n’est pas déjà présent.
  - Utilisez `PIP_NO_BUILD_ISOLATION=1`, puis installez `numpy` avant `requirements-chatterbox.txt`.

## Scripts (optionnel)

- `scripts/dev-backend.sh` : lance l’API (active la venv si présente)
- `scripts/dev-frontend.sh` : lance le frontend
- `scripts/dev.sh` : lance backend + frontend (frontend au premier plan)
- `scripts/doctor.sh` : diagnostic dépendances/venvs (exit non‑zero si manquant)
- `scripts/install-chatterbox-venv.sh` : crée le venv Chatterbox isolé
- `scripts/bootstrap.sh` : installation from scratch (min/std)
- `scripts/smoke.sh` : smoke tests API

## Smoke tests (validation rapide)

Backend :

```bash
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/tts/engines
curl "http://127.0.0.1:8000/v1/tts/voices?engine=chatterbox_native"
```

Frontend :

- Ouvrir http://localhost:3000 et vérifier que moteurs + voix s’affichent.

## Workflow recommandé

1. Collez votre texte dans **Préparation**
2. (Optionnel) Ajustez le texte / durée
3. **Direction** : chargez un snapshot, insérez `[[CHUNK]]` si besoin
4. Lancez **Générer**
5. (Optionnel) Activez l’édition minimale et générez un fichier édité

---

Pour toute demande de modification, gardez la règle d’or :
**si le bénéfice n’est pas immédiatement audible, la fonctionnalité n’a pas sa place en V2.**
