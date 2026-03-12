# Invariants critiques

Ce document centralise les invariants fonctionnels et operationnels utilises comme reference d'audit.

## Invariants fonctionnels

1. Auth requise sur les routes protegees
- Regle: hors mode local explicite, les routes protegees exigent une cle API valide.
- Source: `backend/security.py` (`required_api_key`, `require_authorized`), `backend/app.py` (dependencies sur routeurs).
- Verification: `tests/test_api_auth.py`, execution API avec/sans `X-API-Key`.

2. Limite de taille de texte
- Regle: une requete TTS depassant `MAX_TEXT_CHARS` est rejetee.
- Source: `backend/config.py` (`MAX_TEXT_CHARS`), `backend/routes/tts.py`.
- Verification: `tests/test_api_jobs.py`.

3. Format de sortie borne a WAV
- Regle: les jobs TTS via endpoint courant n'acceptent que `wav`.
- Source: `backend/routes/tts.py` (`only_wav_supported`).
- Verification: tests d'API jobs + essais manuels curl.

4. Parametres moteur non supportes non imposes implicitement
- Regle: le systeme evite d'envoyer des parametres non supportes par un moteur.
- Source: principes documentes dans `README.md` (philosophie de conception), mapping options dans `backend/routes/tts.py`.
- Verification: `tests/test_api_engine_schema.py`, `tests/test_api_jobs.py`.

## Invariants operationnels

5. Rate limit applique sur endpoints lourds
- Regle: les endpoints lourds utilisent un token bucket et peuvent retourner `429`.
- Source: `backend/rate_limit.py` (`enforce_heavy`), usage dans `backend/routes/tts.py`.
- Verification: `tests/test_api_rate_limit.py`, `scripts/check-rate-limit-fairness.py`, CI.

6. Baseline securite prod explicite
- Regle: cle API forte, docs API desactivees en prod, hosts/proxies explicites, limites upload configurees.
- Source: `.env.example`, `docs/security-runbook.md`, `scripts/check-security-baseline.sh`.
- Verification: job CI backend dans `.github/workflows/ci.yml`.

7. Frontieres de stockage locales stables
- Regle: repertoire de travail et sorties derives de variables env ou defaults fixes, avec creation controlee.
- Source: `backend/config.py` (`WORK_DIR`, `OUTPUT_DIR`, `PRESETS_DIR`).
- Verification: `tests/test_output_naming.py`, smoke local.

## Notes de gouvernance

- Toute evolution d'invariant doit mettre a jour ce document et la source de verification associee (test/script/CI).
- Ce document est la vue compacte; la source de verite technique reste le code reference ci-dessus.
