# 🎙️ Vocalie-TTS

## How to read this README

- [Quickstart (try it fast)](#quickstart-api--frontend)
- [Architecture & principles](#présentation)
- [API usage](#api-endpoints-v1)
- [Installation from scratch](#installation-from-scratch)
- [Security model](#sécurité-perso-first)
- [Contributing / extending](#scripts-optionnel)

## Quickstart scripts

1. `./scripts/bootstrap.sh min` – installe l’API + Chatterbox (préfetch auto).
2. `./scripts/bootstrap.sh std` – ajoute XTTS + Piper (à utiliser pour un socle complet).
3. `./scripts/dev.sh` – redémarre le backend + front; Linux utilise `npm ci` sur un lock Linux strict.
4. Sur macOS : `./scripts/dev-macos.sh` (installe les dépendances via `npm install --include=optional`, démarre backend + frontend sans modifier le lock).

Les scripts `scripts/dev.sh` / `scripts/dev-macos.sh` sont tes “Quickstart” pour lancer l’ensemble (backend + frontend + optional cockpit). Passer par `scripts/dev-macos.sh` évite les erreurs `npm ci` sur mac car il utilise un install local compatible macOS.

## Présentation

Vocalie-TTS est une interface locale pour produire des voix off en français avec un pipeline simple, stable et reproductible.
Le cœur du produit est la génération TTS.

**L’API est la source de vérité de l’application.** Le Frontend (Next.js) et Gradio sont tous deux des clients de l’API. Gradio ne constitue pas l’interface de production mais sert de cockpit ou d’outil de debug pour explorer et contrôler le backend.

Objectifs :

- génération fiable (multi-moteurs),
- chunking explicite et déterministe (aucun découpage automatique implicite),
- post‑traitement minimal (optionnel),
- sorties propres dans `./output/`.

## Stack actuelle (résumé)

- UI Gradio locale (macOS friendly)
- Backend API : FastAPI + Pydantic + pytest
- Frontend : Next.js (React) + Tailwind CSS + shadcn/ui (Radix UI)
- moteurs : Chatterbox, XTTS v2, Piper, Bark
- chunking **manuel** via marqueur `[[CHUNK]]` (mode Direction)
- montage inter‑chunk optionnel (silence) pour Chatterbox
- édition minimale **optionnelle** : trim début/fin + normalisation

## Licence

- Code de ce dépôt : MIT (voir `LICENSE`).
- Dépendances (Python/Node) : conservent leurs licences respectives.
- Modèles/poids et contenus téléchargés (ex: Bark / XTTS / Chatterbox / Piper) : soumis aux licences/conditions des projets upstream et/ou des fichiers distribués (ex: Hugging Face). Vous êtes responsable de vérifier ces licences avant redistribution ou usage commercial.

## Principe fondamental

Vocalie‑TTS est pensé pour trouver un **équilibre entre automatisation et approche manuelle** : l’app t’aide à préparer, structurer et fiabiliser la génération, tout en te laissant le contrôle sur les décisions qui impactent réellement le rendu.

Principe : **automatiser ce qui est répétitif, garder explicite ce qui influence le son**.

- Pas de découpage automatique caché (le chunking reste une décision visible)
- Pas de post‑traitement audio non demandé
- Pas de paramètre envoyé à un moteur qui ne le supporte pas

## Moteurs supportés

- **Chatterbox** (FR + multilangue)
- **XTTS v2** (voice cloning, ref audio obligatoire)
- **Piper** (offline rapide, voix à installer)
- **Bark** (créatif, expérimental)

L’UI est capability‑driven : seuls les paramètres supportés par le backend sont visibles et envoyés.
Par exemple, les paramètres de référence vocale ou de segmentation ne sont affichés que pour les moteurs qui les supportent.

## À venir (roadmap)

- **Bark** : stabilisation (presets, perf CPU, prefetch optionnel).
- **Assistant LLM** : aide à structurer le texte (titres, sections, pauses, proposition de chunks) avant génération, sans modifier le texte sans validation explicite de l’utilisateur.

## Bark (installation)

Installation venv isolé :

```bash
./scripts/install-bark-venv.sh
```

Alternative :

```bash
./scripts/bootstrap.sh bark
```

Paramètres exposés via `GET /v1/tts/engine_schema?engine=bark` :

- `voice_preset`
- `text_temp` (0..1)
- `waveform_temp` (0..1)
- `seed` (0 = aléatoire)
- `device` (cpu)

Notes :

- Bark peut télécharger des poids au premier lancement (cache sous `./.assets/bark/`).
- macOS : CPU uniquement (par design).
- Si ça timeoute au premier run : export `VOCALIE_BARK_TIMEOUT_S=600` (ou `VOCALIE_BARK_SMALL_MODELS=1`).
- Les poids sont pré-téléchargés lors de `./scripts/bootstrap.sh std` (ou `./scripts/install-bark-venv.sh`).
- Si tu vois une erreur PyTorch `Weights only load failed` : réinstalle Bark (le venv) après mise à jour des deps (`torch<2.6` dans `requirements-bark.txt`).

## Pipeline de génération

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
- macOS Intel : support best effort
- Python 3.11
- Node.js >= 20
- **ffmpeg** (recommandé, requis pour XTTS si la référence n’est pas en WAV)

Installez ffmpeg via votre gestionnaire système (ex: macOS `brew install ffmpeg`).

## Structure repo (résumé)

- Backend API + cockpit : à la racine du repo
- Frontend Next.js : `./frontend`

## Ordre de lancement (recommandé)

1. Démarrer l’API backend
2. Démarrer le frontend
3. (Optionnel) Démarrer le cockpit Gradio

> ⚠️ **Le cockpit Gradio n’est jamais requis pour l’utilisation normale.**

Le cockpit Gradio est un **outil d’exploration et de contrôle**, il ne fait pas partie du chemin critique de production.

## Quickstart (API + Frontend)

### Backend (API) — installation minimale (runtime)

```bash
cd Vocalie-TTS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

### Sécurité (LAN)

- Par défaut, l’API doit rester en local (`--host 127.0.0.1`).
- Pour une exposition LAN volontaire, définissez `VOCALIE_API_KEY` et envoyez `Authorization: Bearer <token>` (ou `X-API-Key: <token>`).
- Si `VOCALIE_API_KEY` n’est pas défini : toute requête non-locale est refusée (403), même si vous lancez `0.0.0.0`.

### Sécurité (perso-first)

**⚠️ Disclaimer :**
**Ce service n’est pas conçu pour être exposé sur Internet.**
**Il est destiné à un usage local ou sur réseau local (LAN) uniquement, sauf si vous le renforcez spécifiquement.**

- Ne pas exposer sur Internet (HTTP local uniquement).
- CORS strict (whitelist) : configurez `VOCALIE_CORS_ORIGINS` (CSV). `*` n’est pas supporté.
- Rate limit soft (endpoints lourds) :
  - `VOCALIE_RATE_LIMIT_RPS` (défaut 5)
  - `VOCALIE_RATE_LIMIT_BURST` (défaut 10)
  - appliqué à `POST /v1/tts/jobs` et `POST /v1/audio/edit` (pas à `/v1/health`).

## Rôle de Gradio

Gradio existe comme cockpit d’exploration et de debug pour l’API backend. Il permet de tester rapidement les fonctionnalités, d’inspecter les retours de l’API et de contrôler les moteurs TTS sans passer par l’interface utilisateur de production (Frontend Next.js).

Gradio est utile pendant le développement, l’intégration de nouveaux moteurs ou pour du prototypage rapide. Il peut être retiré entièrement en production : tout usage normal passe par l’API et le frontend.

## Reproductibilité (lockfiles)

- Python : `requirements.lock.txt` + `requirements-chatterbox.lock.txt` (générés via `./scripts/lock-requirements.sh`)
- Bootstrap utilise les lockfiles si présents.
- CI : préférer `pip install -r requirements.lock.txt`.
- Node : `npm ci` (lock déjà fourni via `package-lock.json`).

### Node lockfile sur Linux (CI)

Certaines dépendances frontend (ex: `lightningcss`, `@tailwindcss/oxide`) s’appuient sur des binaires natifs.

**Choix de design (strict)** :
- La CI frontend est volontairement stricte et n’exécute que `npm ci`.
- Si les binaires natifs Linux ne sont pas présents après `npm ci`, la CI échoue avec un message explicite.
- Aucun “auto-fix” (pas de `npm install`, pas de suppression de lockfile) : le lockfile est la source de vérité.

- Régénérer le lockfile côté Linux (Docker) :
  - `docker run --rm -v "$PWD/frontend:/app" -w /app node:20-bookworm bash -lc "rm -rf node_modules package-lock.json && npm install --include=optional --no-audit --progress=false"`
  - ou via script (depuis la racine du repo) : `bash ./scripts/gen-lock-linux.sh`

## API endpoints (v1)

### Presets

```bash
curl -s http://localhost:8000/v1/presets

curl -s http://localhost:8000/v1/presets/default

curl -sX POST http://localhost:8000/v1/presets \\
  -H 'Content-Type: application/json' \\
  -d '{\"id\":\"demo\",\"label\":\"Demo\",\"state\":{\"preparation\":{\"text_raw\":\"Bonjour\"},\"engine\":{\"engine_id\":\"piper\",\"params\":{}}}}'

curl -sX PUT http://localhost:8000/v1/presets/demo \\
  -H 'Content-Type: application/json' \\
  -d '{\"label\":\"Demo v2\",\"state\":{\"preparation\":{\"text_raw\":\"Salut\"},\"engine\":{\"engine_id\":\"piper\",\"params\":{}}}}'

curl -sX DELETE http://localhost:8000/v1/presets/demo
```

### Preparation

```bash
curl -sX POST http://localhost:8000/v1/prep/adjust \\
  -H 'Content-Type: application/json' \\
  -d '{\"text_raw\":\"Bonjour  monde\"}'

curl -sX POST http://localhost:8000/v1/prep/interpret \\
  -H 'Content-Type: application/json' \\
  -d '{\"text_adjusted\":\"Bonjour monde\",\"glossary_enabled\":false}'
```

### Direction / Chunking

```bash
curl -sX POST http://localhost:8000/v1/chunks/snapshot \\
  -H 'Content-Type: application/json' \\
  -d '{\"text_interpreted\":\"Bonjour le monde\"}'

curl -sX POST http://localhost:8000/v1/chunks/apply_marker \\
  -H 'Content-Type: application/json' \\
  -d '{\"snapshot_text\":\"Bonjour le monde\",\"action\":\"insert\",\"position\":7}'

curl -sX POST http://localhost:8000/v1/chunks/preview \\
  -H 'Content-Type: application/json' \\
  -d '{\"snapshot_text\":\"Bonjour [[CHUNK]] le monde\"}'
```

### Engine schema

```bash
curl -s \"http://localhost:8000/v1/tts/engine_schema?engine=chatterbox_native\"
```

### Audio edit

```bash
curl -sX POST http://localhost:8000/v1/audio/edit \\
  -H 'Content-Type: application/json' \\
  -d '{\"asset_id\":\"asset_xxx\",\"trim_enabled\":true,\"normalize_enabled\":true,\"target_dbfs\":-1.0}'
```

## Installation from scratch

### Quickstart (bootstrap)

```bash
./scripts/bootstrap.sh min   # core + chatterbox (smoke auto)
./scripts/bootstrap.sh std   # min + xtts + piper (smoke auto)
./scripts/bootstrap.sh clean # supprime .venv et .venvs
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

### Frontend (Next.js)

```bash
cd frontend
npm ci
npm run dev
```

> ⚠️ **Bootstrap strict sur Linux seulement**  
> `npm ci` repose sur le lockfile orienté Linux (celui que la CI et le bootstrap utilisent). Sur macOS cette commande échoue à cause de binaires natifs manquants.

### Frontend sur macOS (dev local)

1. Installe manuellement les dépendances localement :
   ```bash
   cd frontend
   npm install --include=optional --no-audit --progress=false
   ```
2. Lance le frontend :
   ```bash
   npm run dev
   ```
3. Avant tout commit, annule les changements du lockfile générés localement :
   ```bash
   git checkout -- frontend/package-lock.json
   ```

`scripts/dev-frontend.sh` détecte macOS et t’indique ce workflow plutôt que d’essayer `npm ci`.

### Démarrage complet macOS

Sur macOS, tout faire “from scratch” devient :

```bash
./scripts/bootstrap.sh min
./scripts/dev-macos.sh
```

`scripts/dev-macos.sh` installe les dépendances front localement puis lance backend + frontend (équivalent de `scripts/dev.sh` mais en gardant les binaires mac). Tu peux aussi lancer le backend séparément et `cd frontend && npm run dev` si tu préfères plus de contrôle.

Ouvrez ensuite http://localhost:3000

## Quickstart (Gradio cockpit)

```bash
cd Vocalie-TTS
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
Vocalie-TTS/
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

## Schéma conceptuel (simplifié)

```
[ Frontend Next.js ]
         |
         v
      [ API Core ]
         |
  -----------------
  |       |       |
[Chatter] [XTTS] [Piper]
```

Chaque moteur TTS s’exécute dans son **environnement Python isolé** et est invoqué par l’API core via subprocess.
Cette séparation garantit la stabilité, la reproductibilité et l’indépendance des moteurs.

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

> **Rationale** : L’utilisation de subprocess et d’environnements Python isolés (venvs) garantit la stabilité, évite les conflits de dépendances entre moteurs et assure la reproductibilité des exécutions.

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
- Crash Gradio `api_info()` (TypeError bool iterable) : mismatch `gradio`/`gradio_client`. Gardez les versions alignées et laissez `show_api=False`.
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
- `scripts/dev.sh` : lance backend + frontend (+ cockpit si `WITH_COCKPIT=1`)
- `scripts/stop.sh` : stoppe les services lancés par `dev.sh`
- `scripts/status.sh` : affiche le statut des services + ports
- `scripts/doctor.sh` : diagnostic dépendances/venvs (exit non‑zero si manquant)
- `scripts/install-chatterbox-venv.sh` : crée le venv Chatterbox isolé
- `scripts/install-bark-venv.sh` : crée le venv Bark isolé
- `scripts/bootstrap.sh` : installation from scratch (min/std)
- `scripts/lock-requirements.sh` : génère les lockfiles Python
- `scripts/update-openapi.sh` : snapshot OpenAPI (contrat API)
- `scripts/smoke.sh` : smoke tests API

## Contrat API (OpenAPI)

- Snapshot versionné : `openapi.json`
- Mettre à jour après changement d’API : `./scripts/update-openapi.sh`

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
**si le bénéfice n’est pas immédiatement audible, la fonctionnalité n’a pas sa place.**

## Design philosophy

- Explicite plutôt qu’implicite
- API-first
- Aucun ajout de fonctionnalité sans bénéfice audible
