# Fix Plan — Technical Debt P1–P3 — 20260527

## Goal

Execute the three highest-priority technical debt treatments from `docs/reports/TECH_DEBT_AUDIT_20260527.md`:

- **P1**: TD-002 (backend boundary violations) + TD-003 (triple path constants)
- **P2**: TD-005 (3-way engine alias map) + TD-013 (dual preset management)
- **P3**: TD-008 (contract tests for untested backend services)

## Path

**STRUCTURED** — Each priority is a sequential dependency. P1 must complete before P2 is meaningful (P2 touches backend services whose imports will change during P1). P3 can proceed independently but benefits from P1's cleaner module structure.

## Governance status

| File | Status | Relevant to |
|------|--------|------------|
| `docs/CONVENTIONS.md` | ✅ Available | P1 (import/boundary rules), P2 (naming, engine IDs) |
| `docs/DECISIONS.md` | ✅ Available | P1 (DEC-001 API-first), P2 (DEC-005 canonical engine IDs) |
| `docs/invariants.md` | ✅ Available | P1 (invariant #7 storage paths), P3 (invariant verification) |
| `docs/system-boundaries.md` | ✅ Available | P1 (internal boundaries), P2 (Preset domain boundary) |
| `docs/ENV_POLICY.md` | ✅ Available | P1 (path constants reference) |
| `docs/GLOSSARY.md` | ✅ Available | P2 (engine ID definitions) |
| `.github/workflows/ci.yml` | ✅ Available | All (CI must pass after each run) |
| `docs/AUDIT_STATUS.md` | ✅ Available | Status tracking |

Governance is complete. No missing files.

## Primary skill used

`0-vbb-pilotage` — manual implementation (agent not registered)

## Supporting skills

- `0-vbb-scope-freeze` — verified scope freeze guardrails in CI before planning
- `0-vbb-audit-readiness` — audit readiness report serves as baseline (READY verdict)
- `t-vbb-impact-analyzer` — manual implementation (agent not registered); impact analysis performed above

## Fallback justification

`0-vbb-pilotage`, `0-vbb-scope-freeze`, `0-vbb-audit-readiness`, `t-vbb-impact-analyzer` are not in the current executable agent list. Each was implemented manually following the Vibebackbone specification.

## Assumptions

1. **P1 is the highest-impact, lowest-risk change.** Moving shared root modules into `backend/` is mechanical and reversible. All 6 root modules (`refs`, `text_tools`, `audio_defaults`, `output_paths`, `session_manager`, `tts_pipeline`) have no root-level circular dependencies — they import only from stdlib and third-party packages (numpy, soundfile), not from `app.py` or `state_manager.py` (except `session_manager` which imports from `text_tools` and `soundfile`).

2. **The canonical backend is the future surface.** Per DEC-001 and DEC-003, `backend/` is the canonical runtime and Gradio is the compatibility cockpit. P1 strengthens this boundary.

3. **`state_manager.py` is a special case.** It overlaps with `backend/services/preset_service.py` (TD-013). It should NOT be moved into `backend/` because it would create two competing state systems inside the canonical boundary. Instead, P2 addresses this by extracting the shared engine alias map.

4. **Tests will break.** 15+ test files import from root-level modules. After P1 moves modules into `backend/`, test imports must be updated. This is mechanical but high-touch.

5. **The Gradio surface (`app.py`, `ui_gradio/`) must continue to work.** After P1, `app.py` and `ui_gradio/gradio_helpers.py` will import from `backend/` instead of root level. This is the intended direction per DEC-001.

6. **P2's engine alias map consolidation requires P1 to be complete first.** The canonical source for engine IDs is already in `backend/routes/tts.py` (`ENGINE_CATALOG`). Moving it to a shared location (`tts_backends/` or `backend/`) requires the boundary to be clean first.

7. **P3 is independent and can proceed in parallel with P1/P2, but testing is more reliable after P1's module moves are complete.**

8. **Scope is unchanged.** No new features, no API surface changes. All changes are internal restructuring that preserves existing behavior.

---

## Plan

### Run 01 — P1: Move shared root modules into `backend/` (TD-002, TD-003)

**Goal**: Make the canonical backend self-contained by moving the 6 modules it imports from root level into `backend/shared/`. Consolidate path constants in `backend/config.py` only.

**Target files**:

| Current location | New location | Size | Used by backend | Used by app.py/Gradio | Used by tests |
|---|---|---|---|---|---|
| `refs.py` | `backend/shared/refs.py` | 121 lines | `tts_service.py` (1 import) | `app.py` (4 imports) | 1 test file |
| `text_tools.py` | `backend/shared/text_tools.py` | 884 lines | `tts_service.py` (3 imports), `routes/tts.py` (1), `routes/chunks.py` (1), `routes/prep.py` (1) | `app.py` (many) | 8 test files |
| `audio_defaults.py` | `backend/shared/audio_defaults.py` | 4 lines | `tts_service.py` (1), `routes/audio.py` (1) | `app.py` (1) | 0 |
| `output_paths.py` | `backend/shared/output_paths.py` | 137 lines | `tts_service.py` (2), `routes/audio.py` (1), `audiosr_service.py` (2) | `app.py` (3) | 2 test files |
| `session_manager.py` | `backend/shared/session_manager.py` | 398 lines | `tts_service.py` (4) | `app.py` (8) | 4 test files |
| `tts_pipeline.py` | `backend/shared/tts_pipeline.py` | 357 lines | `tts_service.py` (1) | `app.py` (1) | 4 test files |

**Path constants consolidation** (TD-003): Remove duplicate `WORK_DIR`/`DEFAULT_OUTPUT_DIR` from `app.py` and `ui_gradio/gradio_helpers.py`. Import from `backend/config.py` instead.

**Steps**:

1. Create `backend/shared/__init__.py`
2. Move each module to `backend/shared/` ( preserving exact content)
3. Add re-export shims at root level (`refs.py` → `from backend.shared.refs import *`) for backward compatibility with existing `app.py`, `ui_gradio/`, and `tts_backends/` imports
4. Update `backend/` imports from root-level to `backend.shared`
5. Update `backend/config.py` to be the sole source of `WORK_DIR`/`OUTPUT_DIR`
6. Remove path constants from `app.py` and `ui_gradio/gradio_helpers.py`, import from `backend.config`
7. Update all test imports
8. Run `pytest -q`
9. Run `ruff check`
10. Run CI guardrail checks
11. Update `ui_gradio/gradio_helpers.py` imports

**Vigilance points**:
- `session_manager.py` depends on `text_tools.py` (same-package import will change)
- `state_manager.py` is NOT moved (it overlaps with `preset_service.py` — TD-013, addressed in P2)
- Root-level re-export shims must be thin wrappers that preserve the public API
- `tts_backends/` modules also import some of these (verify no breakage)
- `app.py` still imports from `state_manager.py`, `logging_utils.py`, `conftest.py`, `audio_defaults.py` — these remain at root unless moved

**Expected outcome**: Backend imports only from `backend/` package. Root-level re-export shims maintain Gradio compatibility. Path constants computed once in `backend/config.py`.

---

### Run 02 — P2: Consolidate engine alias map (TD-005, TD-013 partial)

**Goal**: Extract the engine alias map to a single source of truth. Reduce `_legacy_to_ui_state` in `preset_service.py` by making it import from the shared catalog.

**Target files**:

| File | Current state | Change |
|---|---|---|
| `backend/routes/tts.py` | `ENGINE_CATALOG` list with `id`, `label`, `backend_id`, `supports_ref` | Extract to `tts_backends/catalog.py` as canonical source |
| `backend/services/preset_service.py` | `_legacy_to_ui_state()` with inline `engine_map` dict | Import alias map from catalog |
| `state_manager.py` | Inline engine name handling in `migrate_state`/`load_engine_config` | Import alias map from catalog |
| `ui_gradio/engine_ui_helpers.py` | `backend_choices()`, engine param helpers | Import from catalog |
| `app.py` | Engine name references via state_manager | No change (imports from state_manager which will use catalog) |

**Steps**:

1. Create `tts_backends/catalog.py` with:
   - `ENGINE_CATALOG` (moved from `backend/routes/tts.py`)
   - `ENGINE_ALIAS_MAP` = `{"chatterbox": "chatterbox_finetune_fr", "xtts": "xtts_v2"}` (canonical)
   - Helper: `canonical_engine_id(raw_id: str) -> str`
   - Helper: `is_legacy_alias(engine_id: str) -> bool`
2. Update `backend/routes/tts.py` to import `ENGINE_CATALOG` from `tts_backends.catalog`
3. Update `backend/services/preset_service.py` to import `ENGINE_ALIAS_MAP` and `canonical_engine_id` from catalog, replace inline map
4. Update `state_manager.py` to import from `tts_backends.catalog` for engine name resolution
5. Update `ui_gradio/engine_ui_helpers.py` to import from catalog
6. Keep `_legacy_to_ui_state()` in `preset_service.py` for now (it does more than just alias mapping — it reshapes the entire preset structure)
7. Run `pytest -q`
8. Run `ruff check`

**Vigilance points**:
- The alias map must remain backward-compatible during transition
- `_legacy_to_ui_state` does structural reshaping, not just ID translation — don't remove it until preset files are migrated
- Tests that assert on engine IDs must use canonical IDs
- `ENGINE_CATALOG` currently defines `backend_id` mapping — ensure this is preserved

**Expected outcome**: Engine alias map defined once in `tts_backends/catalog.py`. All consumers import from it. No behavioral changes.

---

### Run 03 — P3: Contract tests for untested backend services (TD-008)

**Goal**: Add contract-level tests for `asset_service`, `work_service`, and verify `tts_service` orchestrations via route tests.

**Target files** (new):

| Test file | Module under test | Priority |
|---|---|---|
| `tests/test_asset_service.py` | `backend/services/asset_service.py` | HIGH |
| `tests/test_work_service.py` | `backend/services/work_service.py` | HIGH |
| `tests/test_config.py` | `backend/config.py` (env var parsing) | MEDIUM |

**Steps**:

1. Write `tests/test_asset_service.py`:
   - Test `write_meta` / `get_meta` round-trip
   - Test `resolve_asset` with valid and invalid IDs
   - Test metadata structure matches `AssetMetaResponse` schema
   - Test file creation in `OUTPUT_DIR/.assets/`

2. Write `tests/test_work_service.py`:
   - Test `clean_work_dir` with `VOCALIE_KEEP_WORK=1` (skip)
   - Test `clean_work_dir` creates and cleans `.sessions/` and `.tmp/`
   - Test path safety validation (rejects paths outside base dir)

3. Write `tests/test_config.py`:
   - Test env var defaults for `WORK_DIR`, `OUTPUT_DIR`, `PRESETS_DIR`
   - Test `VOCALIE_OUTPUT_DIR` override
   - Test `CHATTERBOX_OUT_DIR` backward compat

4. Run full test suite
5. Verify CI passes

**Vigilance points**:
- These tests verify behavior, not implementation details
- Use `tmp_path` fixture for filesystem tests
- Mock `OUTPUT_DIR`/`WORK_DIR` env vars in config tests
- Asset service tests should clean up after themselves

**Expected outcome**: 3 new test files covering previously untested backend services. CI regression baseline established.

---

## Execution readiness

**Status: waiting for confirmation**

The plan is explicit and complete. No execution has been performed. All three runs are defined with target files, steps, vigilance points, and expected outcomes.

**Planned runs**:

| Run | Priority | Target debt | Target files | Effort |
|-----|----------|-------------|-------------|--------|
| Run 01 | P1 | TD-002, TD-003 | 6 modules to `backend/shared/`, path constants in `config.py` | L | **DONE** c4c5b62 |
| Run 02 | P2 | TD-005, TD-013 | New `tts_backends/catalog.py`, update 4 consumers | M |
| Run 03 | P3 | TD-008 | 3 new test files | M |

**Run order**: Run 01 → Run 02 → Run 03 (sequential, each depends on previous for stable module boundaries)

**First run to execute**: Run 01 (P1: backend boundary cleanup + path constant consolidation)

**Vigilance escalation**: If Run 01 reveals circular dependencies or unexpected coupling not caught during analysis, stop and escalate before proceeding to Run 02.
---

## Execution result

**All three runs completed successfully.**

| Run | Status | Commit | What changed |
|-----|--------|--------|--------------|
| Run 01 (P1) | ✅ DONE | `c4c5b62` | 6 root modules → `backend/shared/`, re-export shims at root, path constants consolidated in `backend/config.py` |
| Run 02 (P2) | ✅ DONE | `8ef3d33` | `tts_backends/catalog.py` created, ENGINE_CATALOG + ENGINE_ALIAS_MAP as single source, 4 consumers updated |
| Run 03 (P3) | ✅ DONE | `03fac63` | 32 new contract tests for asset_service, work_service, config |

**Debt items resolved:**
- TD-002 (backend boundary violations): ✅ — `backend/` no longer imports from root-level modules; all imports go through `backend.shared.*`
- TD-003 (triple path constants): ✅ — WORK_DIR/OUTPUT_DIR computed once in `backend/config.py`; `app.py` and `gradio_helpers.py` import from it
- TD-005 (3-way engine alias map): ✅ — Single source in `tts_backends/catalog.py`; inline maps removed from `preset_service.py`
- TD-013 (dual preset mgmt): ✅ partial — `state_manager.py` now uses `canonical_engine_id()` from catalog; dual validation model remains (full convergence in future P2 iteration)
- TD-008 (untested services): ✅ — 32 new tests covering asset_service, work_service, config
