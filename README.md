# 🎙️ Vocalie-TTS

## Presentation

Vocalie-TTS est une interface locale dediee a la production de voix off en francais, concue autour d'un pipeline simple, stable et reproductible.

Le coeur du projet est la generation TTS. L'outil est particulierement optimise pour Chatterbox, notamment pour les usages necessitant un decoupage precis du texte et un controle fin du rythme (chunking manuel).

Vocalie-TTS s'adresse aux createurs audio/video qui recherchent :
- un controle explicite du rendu,
- une generation locale sans dependance cloud,
- un pipeline fiable, oriente production.

---

## Fonctionnalites principales

- Adaptation automatique des anagrammes et termes specifiques en phonetique via un glossaire
- Placement manuel des chunks pour segmenter finement les generations
- Reglages specifiques par moteur TTS
  - pour Chatterbox : controle explicite de la duree des silences entre les chunks
- Post-traitement audio optionnel :
  - nettoyage des silences en debut et fin
  - normalisation du niveau sonore (dBFS)

---

## A venir

- Integration d'un LLM local pour ameliorer le texte et l'adapter a une duree cible, sans modification implicite du contenu.

---

## Architecture

La solution a ete initialement construite autour de Gradio, puis a evolue vers une architecture API-first avec un backend et un frontend separes afin d'ameliorer la robustesse et l'experience utilisateur.

L'API est la source de verite de l'application.
Le frontend (Next.js) et Gradio sont deux clients distincts de cette API.

- Le frontend constitue l'interface utilisateur de production.
- Gradio n'est pas une UI de production : il sert de cockpit et d'outil de debug pour explorer et controler le backend.

---

## Objectifs

- generation fiable (multi-moteurs),
- chunking explicite et deterministe (aucun decoupage automatique implicite),
- post-traitement minimal (optionnel),
- sorties propres et organisees dans `./output/`.

---

## Sommaire

- [Presentation](#presentation)
- [Architecture](#architecture)
- [Objectifs](#objectifs)
- [Quickstart](#quickstart)
- [Installation par plateforme](#installation-par-plateforme)
- [Voix et references](#voix-et-references)
- [Reproductibilite](#reproductibilite-lockfiles)
- [API endpoints (v1)](#api-endpoints-v1)
- [Troubleshooting](#troubleshooting)
- [Scripts](#scripts-optionnel)
- [Licence](#licence)

---

## Quickstart

### Pre-requis

- Python 3.11
- Node.js >= 20
- ffmpeg (recommande, requis pour XTTS si la reference n'est pas en WAV)

### Installation rapide (macOS)

```bash
brew install python@3.11 ffmpeg

git clone https://github.com/Bricesodini/Vocalie-TTS.git
cd Vocalie-TTS

./scripts/bootstrap.sh min

# Chatterbox : precharger les poids dans le cache Hugging Face
./scripts/install-chatterbox-weights.sh ResembleAI/chatterbox
export HUGGINGFACE_TOKEN=<token-avec-acces>
./scripts/install-chatterbox-weights.sh Thomcles/Chatterbox-TTS-French

# Frontend macOS
cd frontend
npm install --include=optional --no-audit --progress=false
cd ..

./scripts/dev-macos.sh
```

Ouvrir ensuite :
- Frontend : http://localhost:3000
- API : http://127.0.0.1:8000

---

## Installation par plateforme

### Linux

```bash
./scripts/bootstrap.sh std
./scripts/dev.sh
```

Le frontend utilise `npm ci` avec un lockfile Linux-only (CI stricte).

---

### macOS

1. Bootstrap backend + moteurs :
```bash
./scripts/bootstrap.sh min
```

2. Precharger les poids Chatterbox (cache HF) :
```bash
./scripts/install-chatterbox-weights.sh ResembleAI/chatterbox
export HUGGINGFACE_TOKEN=<token-avec-acces>
./scripts/install-chatterbox-weights.sh Thomcles/Chatterbox-TTS-French
```

3. Installer les dependances frontend localement :
```bash
cd frontend
npm install --include=optional --no-audit --progress=false
```

4. Lancer l'ensemble :
```bash
./scripts/dev-macos.sh
```

Avant tout commit, annule les changements eventuels du lockfile :
```bash
git checkout -- frontend/package-lock.json
```

---

### Windows

```powershell
pwsh ./scripts/dev-windows.ps1
```

Option GPU Nvidia :
```powershell
setx VOCALIE_ENABLE_CUDA 1
```

---

## Voix et references

### Chatterbox (poids)

Les poids sont telecharges via Hugging Face et stockes dans le cache local :

```
~/.cache/huggingface/hub/models--ResembleAI--chatterbox
~/.cache/huggingface/hub/models--Thomcles--Chatterbox-TTS-French
```

Prechargement manuel :

```bash
export HUGGINGFACE_TOKEN=<token-avec-acces>
./scripts/install-chatterbox-weights.sh ResembleAI/chatterbox
./scripts/install-chatterbox-weights.sh Thomcles/Chatterbox-TTS-French
```

Sans ces deux modeles, Chatterbox echoue au premier run.

### References audio (XTTS / Chatterbox)

- Place tes fichiers `.wav` ou `.m4a` dans `Ref_audio/`
- Chaque fichier devient une voix clonable
- Redemarre le backend pour rafraichir la liste

---

## Philosophie de conception

Vocalie-TTS cherche un equilibre entre automatisation et controle manuel.

Principe fondamental :

Automatiser ce qui est repetitif, garder explicite ce qui influence le son.

- Aucun decoupage automatique cache
- Aucun post-traitement non demande
- Aucun parametre envoye a un moteur qui ne le supporte pas

---

## Reproductibilite (lockfiles)

- Python :
  - `requirements.lock.txt`
  - `requirements-chatterbox.lock.txt`
- Generation :
```bash
./scripts/lock-requirements.sh
```

- Node :
  - `npm ci` strict en CI (Linux)
  - `npm install` local sur macOS

Le lockfile est source de verite. Aucun auto-fix silencieux.

---

## API endpoints (v1)

### Sante

```bash
curl http://127.0.0.1:8000/v1/health
```

### Engines et voix

```bash
curl http://127.0.0.1:8000/v1/tts/engines
curl "http://127.0.0.1:8000/v1/tts/voices?engine=chatterbox_native"
```

### Generation (extrait)

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/jobs \
  -H "Content-Type: application/json" \
  -d '{ "text": "Bonjour", "engine": "piper" }'
```

---

## Troubleshooting

- `npm ci` echoue sur macOS -> utiliser `npm install`
- XTTS sur macOS -> CPU only (comportement attendu)
- Probleme Hugging Face gated model -> verifier `HUGGINGFACE_TOKEN`
- `Module not found` frontend -> `rm -rf node_modules && npm install`

---

## Scripts (optionnel)

- `scripts/bootstrap.sh` : installation from scratch
- `scripts/dev.sh` : backend + frontend (Linux)
- `scripts/dev-macos.sh` : backend + frontend (macOS)
- `scripts/dev-windows.ps1` : Windows
- `scripts/stop.sh` : arret des services
- `scripts/doctor.sh` : diagnostic environnement
- `scripts/smoke.sh` : smoke tests API

---

## Licence

- Code : MIT
- Dependances : licences respectives
- Modeles / poids : soumis aux licences des projets upstream (Hugging Face, Bark, XTTS, Piper, Chatterbox)

---

Si le benefice n'est pas immediatement audible, la fonctionnalite n'a pas sa place.
