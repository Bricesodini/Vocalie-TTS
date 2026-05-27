# Frontieres systeme

Vue synthetique des frontieres de Chatterbox pour auditabilite.

## Entrees

- API HTTP locale: endpoints `GET/POST` sous `/v1/*`.
- Fichiers de reference voix: `Ref_audio/` (`.wav`, `.m4a`).
- Variables d'environnement: configuration securite, limites, paths, mode runtime.
- Payloads utilisateurs: texte TTS, options moteur, fichiers audio a ameliorer.

## Sorties

- Audio genere et artefacts associes dans `output/`.
- Fichiers temporaires et jobs dans `work/`.
- Metadonnees d'assets dans `output/.assets/`.
- Reponses API JSON + headers version.

## Dependances externes

- Modeles/poids via Hugging Face (cache local utilisateur).
- Outils systeme requis (ex: `ffmpeg` selon moteur/flow).
- Ecosysteme Python/Node installe localement (pas de service cloud obligatoire pour le run standard).

## Frontieres reseau et confiance

- Usage cible: local-first.
- Production: API key obligatoire, localhost trust desactive, hosts/proxies explicites.
- Headers de forwarding interpretes seulement si le pair direct est dans `VOCALIE_TRUSTED_PROXIES`.

## Frontieres applicatives internes

- `backend/app.py`: composition middleware, routeurs, garde auth.
- `backend/routes/*`: surface API.
- `backend/services/*`: logique metier et orchestration.
- `backend/workers/*`: traitements asynchrones lourds.
- `backend/shared/*`: modules partages entre backend canonique et surfaces de compatibilite (`refs`, `text_tools`, `audio_defaults`, `output_paths`, `session_manager`, `tts_pipeline`). Les shims de re-export au niveau racine preservent la compatibilite ascendante.
- `frontend/`: client UI de production.

## Hors frontiere explicite (non-goals)

- Exposition publique directe de l'API sans reverse proxy et hardening.
- Gradio comme interface de production.
- Comportements implicites non demandes (rewrite texte, post-traitement auto).
