# Plan de correction — Audit Vocalie-TTS (Juillet 2025/2026)

## Statut : P0 ✅ P1 ✅ P2 13/15 ✅ + Refactoring app.py EN COURS

---

## P0 — Critique ✅ (10/10)
## P1 — Important ✅ (15/15)

## P2 — Amélioration continue

### P2-01 à P2-16 — ✅ (voir résumé précédent)

### Refactoring app.py — EN COURS

**Objectif** : Découper app.py (2920L, 48 fonctions) en modules dans `ui_gradio/`

| Run | Module | Contenu | Lignes extraites | Statut |
|-----|--------|---------|-------------------|--------|
| 1 | `handlers_state.py` | `_reset/set/get_job_state`, `_JOB_LOCK`, `_JOB_STATE`, `_terminate_proc` | 53L | ✅ |
| 2 | `handlers_audio_edit.py` | `_resolve_raw_take_path`, `_apply_minimal_edit`, `update_edit_panel_state`, `handle_generate_edited_audio`, `handle_export_raw_to_output` | 218L | ✅ |
| 3 | `handlers_direction.py` | `handle_direction_*`, `_resolve_direction_source_text`, `_build_chunk_preview`, `_apply_direction_chunking`, etc. | 124L | ✅ |
| 4 | `handlers_presets.py` | `handle_load/save/delete_preset`, `handle_*_confirm`, `_confirm_action`, `refresh_dropdown`, `put` | ~364L | ⬜ TODO |
| 5 | `handlers_engine.py` | `handle_engine/voice/language/change`, `refresh_piper_voices`, `install_default_piper_voice` | ~312L | ⬜ TODO |
| 6 | `handlers_generate.py` | `handle_generate` (480L!), `handle_stop`, `_generate_longform_worker`, etc. | ~640L | ⬜ TODO |
| 7 | `ui_builder.py` | `build_ui` (989L) + helpers divers | ~1064L | ⬜ TODO |

**Progression** : app.py 2920 → 2616 lignes (304L extraites, ~10%)

**Conventions** :
- Chaque module extrait est dans `ui_gradio/handlers_<domain>.py`
- `app.py` importe les fonctions et les re-exporte avec `noqa: F401`
- Les anciennes définitions sont remplacées par un commentaire `# moved to ui_gradio.handlers_<domain>`
- Tests verts après chaque run

### Corrections additionnelles (post-audit)

- **Bug critique** : `status` → `status_code` dans audio.py (NameError potentiel en production)
- **CONV-01** : print() justifiés subprocess §5 dans tts_backends
- **CONV-02** : console.* absent de frontend (error.tsx excepté)
- **CONV-04** : Docstrings ajoutées aux 7 endpoints publics
- **CONV-05** : Fix from __future__ ordering dans 6 fichiers de test
- **CONV-06** : Duplicate import logging dans app.py
- **OBS-01** : `/v1/metrics` endpoint créé (jobs, backends, dirs)
- **OBS-02** : `/v1/health` enrichi (backends, writable dirs, degraded status)
- **HYGIENE-02** : NumPy 1.23.5 → 1.26.4, librosa, scipy mis à jour
- **HYGIENE-03** : pytest-cov + seuil 40% dans CI
- **HYGIENE-04** : 6 env vars documentées dans .env.example
- **HYGIENE-05** : Ruff lint (duplicate import supprimé)
- **DOC-01** : CONVENTIONS.md drift checklist mis à jour
- **DOC-02** : tts_pipeline.py shim marqué DEPRECATED