# TD-001 Audit — app.py Business Logic & Dependency Analysis for Extraction

## Objective

Map every function in `app.py` (2965 lines, 67 functions) to its business domain, external dependencies, coupling score, and extraction viability. Identify which groups can be extracted into separate `ui_gradio/` modules with minimal cross-dependency disruption.

---

## Current state

| Metric | Value |
|--------|-------|
| Total lines | 2965 |
| Handler functions | 25 `handle_*` |
| Internal helpers | 18 `_*` |
| UI layout (`build_ui`) | 991 lines (33% of total) |
| Top-level globals | 7 (`_JOB_LOCK`, `_JOB_STATE`, `_LOAD_PRESET_OUTPUT_COUNT`, `TMP_DIR`, `DEFAULT_OUTPUT_DIR`, `BASE_DIR`, `LEXIQUE_PATH`) |
| Gradio state components | 2 (`session_state`, `confirm_state`) |
| Gradio UI components declared | 73 |
| Handlers wired in `build_ui` | 32 |
| External modules imported | 21 |

---

## Domain decomposition (by line count, largest first)

| Domain | Lines | # Funcs | Coupled to other domains | Extraction risk |
|--------|-------|---------|--------------------------|----------------|
| **UI layout (`build_ui`)** | 1013 | 2 | ALL (wires 32 handlers) | **Critical** — must be done last |
| **Generation/Job management** | 558 | 7 | 8/9 other domains | **High** — most coupled domain |
| **Session/Direction/Edit pipeline** | 323 | 13 | 7/9 other domains | **Medium** — cohesive internally |
| **Preset management** | 315 | 5 | 7/9 other domains | **Medium** — depends on state_manager |
| **Engine/Voice/Language selection** | 223 | 5 | 7/9 other domains | **Medium** — depends on engine_ui_helpers |
| **UI utilities** | 159 | 7 | 6/9 other domains | **Low** — thin wrappers |
| **Upload/Ref** | 107 | 4 | 6/9 other domains | **Low** — narrow scope |
| **Text adjustment** | 49 | 3 | 6/9 other domains | **Low** — self-contained |
| **Backend install** | 21 | 2 | 3/9 other domains | **Low** — minimal coupling |
| **Session** | 18 | 1 | 2/9 other domains | **Low** — minimal coupling |

---

## Function-by-function dependency matrix

### Generation/Job management (558 lines)

| Function | Lines | External modules | Calls other handlers | Uses `_JOB_STATE` | Uses `gr.*` |
|----------|-------|------------------|---------------------|--------------------|-------------|
| `handle_generate` | 482 | backend.config, output_paths, refs, session_manager, text_tools, tts_backends.*, ui_gradio.* | `_apply_direction_chunking`, `_build_chunk_preview`, `_generate_longform_worker`, `_reset_job_state`, `_set_job_state`, `_get_job_state`, `_terminate_proc` | ✅ | ✅ (returns tuple) |
| `_generate_longform_worker` | 27 | backend.shared.tts_pipeline, backend.shared.session_manager | None | ❌ | ❌ |
| `_reset_job_state` | 13 | threading | None | ✅ | ❌ |
| `_set_job_state` | 6 | threading | None | ✅ | ❌ |
| `_get_job_state` | 8 | threading | None | ✅ | ❌ |
| `_terminate_proc` | 8 | multiprocessing | None | ❌ | ❌ |
| `handle_stop` | 14 | ui_gradio.gradio_helpers | `_get_job_state`, `_reset_job_state`, `_terminate_proc` | ✅ | ✅ |

**Extraction viability**: The `_JOB_STATE`/`_JOB_LOCK` module-level global is the coupling point. All job state functions could move to a `ui_gradio/job_state.py` module. `handle_generate` is the largest single function (482 lines) and the highest-risk extraction.

### Session/Direction/Edit pipeline (323 lines)

| Function | Lines | External modules | Uses session_manager | Uses gr.* |
|----------|-------|------------------|---------------------|-----------|
| `_resolve_raw_take_path` | 18 | session_manager | ✅ | ❌ |
| `_apply_minimal_edit` | 48 | audio_defaults, pathlib | ❌ | ❌ |
| `update_edit_panel_state` | 11 | gradio | ❌ | ✅ |
| `_resolve_direction_source_text` | 15 | None (pure) | ❌ | ❌ |
| `handle_direction_load_snapshot` | 17 | ui_gradio.gradio_helpers | ✅ | ✅ |
| `handle_direction_preview` | 18 | None | ✅ (via `_apply_direction_chunking`) | ✅ |
| `update_direction_controls` | 7 | gradio | ❌ | ✅ |
| `append_chunk_marker` | 4 | None (pure) | ❌ | ❌ |
| `handle_generate_edited_audio` | 76 | audio_defaults, output_paths, session_manager, ui_gradio.gradio_helpers | ✅ | ✅ |
| `handle_export_raw_to_output` | 41 | output_paths, pathlib, session_manager, ui_gradio.gradio_helpers | ✅ | ✅ |
| `_build_chunk_preview` | 14 | None (pure) | ❌ | ❌ |
| `_single_chunk` | 21 | text_tools | ❌ | ❌ |
| `_apply_direction_chunking` | 33 | text_tools | ❌ | ❌ |

**Extraction viability**: **HIGH**. This is the best first extraction candidate. The pure text/chunking functions (`_apply_direction_chunking`, `_single_chunk`, `_build_chunk_preview`, `_resolve_direction_source_text`, `append_chunk_marker`) have zero Gradio dependency. The edit/audio functions depend on `session_manager` and `audio_defaults` (both in `backend/shared`). Could become `ui_gradio/direction_handlers.py` and `ui_gradio/audio_edit_handlers.py`.

### Preset management (315 lines)

| Function | Lines | External modules | Uses state_manager |
|----------|-------|------------------|---------------------|
| `handle_load_preset` | 139 | state_manager, refs, backend_install.status, tts_backends.*, ui_gradio.* | ✅ |
| `handle_save_preset` | 55 | state_manager, ui_gradio.* | ✅ |
| `handle_delete_preset` | 24 | state_manager, ui_gradio.gradio_helpers | ✅ |
| `handle_save_preset_confirm` | 60 | ui_gradio.gradio_helpers | ❌ |
| `handle_delete_preset_confirm` | 37 | ui_gradio.gradio_helpers | ❌ |

**Extraction viability**: **MEDIUM**. `handle_load_preset` is the second-largest function (139 lines) with heavy state initialization. Depends on `state_manager`, `engine_ui_helpers`, and engine state. Could become `ui_gradio/preset_handlers.py` but needs access to engine config from `state_manager`.

### Engine/Voice/Language selection (223 lines)

| Function | Lines | External modules |
|----------|-------|------------------|
| `handle_engine_change` | 112 | tts_backends.*, backend_install.status, state_manager, ui_gradio.engine_ui_helpers |
| `handle_voice_change` | 51 | tts_backends.*, tts_backends.piper_assets, ui_gradio.engine_ui_helpers |
| `handle_language_change` | 21 | tts_backends.*, ui_gradio.* |
| `handle_chatterbox_mode_change` | 26 | state_manager, tts_backends.*, ui_gradio.* |
| `engine_status_updates` | 13 | backend_install.status, ui_gradio.* |

**Extraction viability**: **HIGH**. `engine_ui_helpers.py` already exists with 324 lines. These handlers are the natural completion of that module. They depend on `tts_backends`, `state_manager`, and `engine_ui_helpers` — all isolated dependencies. Could become additions to `ui_gradio/engine_ui_helpers.py` or a new `ui_gradio/engine_handlers.py`.

### Text adjustment (49 lines)

| Function | Lines | External modules |
|----------|-------|------------------|
| `handle_adjust` | 12 | text_tools, ui_gradio.gradio_helpers |
| `handle_text_adjustment` | 21 | text_tools, ui_gradio.gradio_helpers |
| `apply_adjusted` | 16 | None (pure passthrough) |

**Extraction viability**: **HIGHEST**. Three functions, one pure passthrough, two thin wrappers. Could become `ui_gradio/text_handlers.py` in 30 minutes.

### Upload/Ref (107 lines)

| Function | Lines | External modules |
|----------|-------|------------------|
| `handle_upload` | 12 | refs, ui_gradio.gradio_helpers |
| `refresh_dropdown` | 6 | refs |
| `refresh_piper_voices` | 42 | tts_backends.*, tts_backends.piper_assets |
| `install_default_piper_voice` | 47 | tts_backends.piper_assets |

**Extraction viability**: **HIGH**. Narrow scope. Could become `ui_gradio/voice_ref_handlers.py`.

### Backend install (21 lines)

| Function | Lines | External modules |
|----------|-------|------------------|
| `handle_install_backend` | 9 | backend_install.installer, ui_gradio.engine_ui_helpers |
| `handle_uninstall_backend` | 12 | backend_install.paths |

**Extraction viability**: **HIGHEST** — two tiny functions. Could merge into `engine_handlers.py`.

---

## Shared mutable state analysis

### `_JOB_STATE` + `_JOB_LOCK` (Generation domain)

**Scope**: Only used by `handle_generate`, `handle_stop`, `_reset_job_state`, `_set_job_state`, `_get_job_state`, `_terminate_proc`.

**Extraction path**: Move all 5 functions + `_JOB_LOCK`/`_JOB_STATE` into `ui_gradio/job_state.py`. This is self-contained.

### `_LOAD_PRESET_OUTPUT_COUNT` (Preset domain)

**Scope**: Only used by `handle_load_preset`.

**Extraction path**: Move with `handle_load_preset` into `ui_gradio/preset_handlers.py`.

### `gr.State` components: `session_state`, `confirm_state`

**Scope**: `session_state` used by `handle_direction_preview`, `handle_generate`, `handle_session_texts`. `confirm_state` used by `_confirm_action` and the 3 confirmation handlers.

**Problem**: These are Gradio state objects — they can only exist within a `gr.Blocks()` context and are passed as event handler arguments/outputs. They **cannot** be extracted; they must remain in `build_ui`.

### Module-level computed globals: `TMP_DIR`, `DEFAULT_OUTPUT_DIR`, `BASE_DIR`, `LEXIQUE_PATH`

**Extraction path**: `DEFAULT_OUTPUT_DIR = OUTPUT_DIR` from `backend.config`. `TMP_DIR = WORK_DIR / ".tmp"` from `backend.config`. Only `LEXIQUE_PATH` and `BASE_DIR` are truly `app.py`-local.

---

## Extraction priority order (safest first)

Based on: (1) internal cohesion, (2) low cross-domain coupling, (3) no shared mutable state conflict, (4) small line count for initial win.

| Priority | Target module | Domain | Lines extracted | Risk | Rationale |
|----------|--------------|--------|-----------------|------|-----------|
| **E1** | `ui_gradio/text_handlers.py` | Text adjustment | ~50 | Minimal | 3 functions, zero shared state, pure wrappers |
| **E2** | `ui_gradio/job_state.py` | Generation job state | ~40 | Low | 5 functions + `_JOB_LOCK`/`_JOB_STATE`, self-contained |
| **E3** | `ui_gradio/direction_handlers.py` | Direction/chunking logic | ~100 | Low | Pure text/chunking functions, no Gradio dependency |
| **E4** | `ui_gradio/voice_ref_handlers.py` | Upload/Ref + install | ~110 | Low | Narrow scope, isolated dependencies |
| **E5** | `ui_gradio/audio_edit_handlers.py` | Edit/export pipeline | ~140 | Medium | Depends on session_manager, audio_defaults |
| **E6** | `ui_gradio/engine_handlers.py` | Engine/voice/language | ~210 | Medium | Depends on state_manager, engine_ui_helpers |
| **E7** | `ui_gradio/preset_handlers.py` | Preset management | ~315 | Medium | Depends heavily on state_manager load/save |
| **E8** | `ui_gradio/generate_handler.py` | handle_generate | ~482 | High | Largest function, most coupled, uses job state |
| **E9** | `ui_gradio/build_ui.py` | UI layout | ~991 | Critical | Must be last — wires all other modules |

---

## What NOT to extract

| Item | Reason |
|------|--------|
| `build_ui()` (991 lines) | Wires all 32 handlers to Gradio components. Must remain in `app.py` until all handlers are extracted. Can be simplified once handlers are in separate modules. |
| `main()` (22 lines) | Entry point, must stay. |
| `refresh_piper_voices()` / `install_default_piper_voice()` | These are Gradio event handlers that return `gr.update()` objects. They depend on the Gradio runtime context — extracting them requires passing `gr` module. Can stay in `app.py` or move to `voice_ref_handlers.py`. |
| `_confirm_action` + confirmation handlers | Tightly coupled to Gradio confirmation flow and `confirm_state`. Low risk but low value. |

---

## Key risk: `handle_generate` (482 lines)

This is the single most complex function in the codebase. It:
- Manages the full TTS generation lifecycle (validation → chunking → job launch → result handling)
- Uses `_JOB_STATE`/`_JOB_LOCK` for concurrent job tracking
- Accesses `LEXIQUE_PATH` for glossary
- Calls `_apply_direction_chunking`, `_single_chunk`, `_build_chunk_preview`
- Spawns multiprocessing workers
- Returns a tuple of ~15 Gradio component updates

**Extraction approach**: 
1. First extract the shared state (`_JOB_STATE`) into `ui_gradio/job_state.py`
2. Then extract the pure chunking/text functions into `ui_gradio/direction_handlers.py`
3. Finally extract `handle_generate` itself into `ui_gradio/generate_handler.py`

Each step should pass the full test suite before proceeding.

---

## Expected `app.py` size after full extraction

| Current | After E1-E4 | After E1-E8 | After E1-E9 |
|---------|------------|-------------|-------------|
| 2965 lines | ~2600 lines | ~1160 lines | ~1013 lines (build_ui + main + imports) |

After all extractions, `app.py` would contain only:
- `build_ui()` — the layout definition that wires everything together
- `main()` — the entry point
- All imports from `ui_gradio.*` handler modules
- Module-level globals (`gr.State` declarations)

This is the target architecture: `app.py` as a thin Gradio assembly that imports handlers from domain modules.