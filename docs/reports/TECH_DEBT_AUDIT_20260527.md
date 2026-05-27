# Technical Debt Audit — Vocalie-TTS — 20260527

## Goal

Identify and classify technical debt with a practical focus on maintainability, risk, and future cost. Surface noise was cleaned first (janitor pass); this audit diagnoses structural debt the janitor cannot address.

## Scope

- Commit: `04244da` on `main`
- Canonical backend: `backend/` (routes, services, schemas, workers, config, security, rate_limit)
- Compatibility/legacy surface: `app.py`, `state_manager.py`, `session_manager.py`, `tts_engine.py`, `tts_pipeline.py`, `ui_gradio/`
- Frontend: `frontend/src/`
- Cross-cutting: `tts_backends/`, `scripts/`, `tests/`, `.github/workflows/ci.yml`

## Janitor verdict and structural signals (from Phase 1)

**Janitor verdict: READY_WITH_STRUCTURAL_SIGNALS**

7 structural signals carried forward:

1. Dual orchestration surface (app.py + backend/)
2. Triple-computed path constants (config.py + app.py + gradio_helpers.py)
3. Dual `clean_work_dir` with semantic divergence
4. Legacy preset bridge embedded in canonical backend
5. In-memory job store with no persistence + parallel Gradio job state
6. Backend boundary violations (6+ root-level imports from canonical services)
7. Frontend monolith (1675 lines, 34 useState)

## Primary skill used

`1-vbb-tech-debt` — manual implementation (agent not registered in current agent list)

## Supporting skills

- `t-vbb-dependency-mapper` — manual implementation (agent not registered)
- `t-vbb-test-coverage-mapper` — manual implementation (agent not registered)

## Fallback justification

`1-vbb-code-janitor`, `1-vbb-tech-debt`, `t-vbb-dependency-mapper`, `t-vbb-test-coverage-mapper` are not registered as executable agents. Each was implemented manually following the Vibebackbone specification.

---

## Debt items

---

### TD-001 — Dual orchestration surface (Gradio monolith + canonical backend)

**Category**: structural  
**What it is**: Two independent orchestration surfaces coexist: `app.py` (2978 lines, 25 Gradio `handle_` functions) and `backend/` (canonical FastAPI routes/services). Both can create jobs, manage state, and orchestrate TTS generation.  
**Why it matters**: Any feature change must be implemented and tested in two places. Risk of behavioral divergence between surfaces. The canonical backend imports 7 root-level modules that belong to the Gradio surface, creating reverse dependencies.  
**Current impact**: Every TTS/engine change requires coordination across both surfaces. `backend/services/tts_service.py` imports from `session_manager`, `output_paths`, `tts_pipeline`, `refs`, `text_tools`, `audio_defaults` — coupling canonical backend to legacy surface.  
**Likely future cost**: Each new engine or feature doubles integration work. Behavioral drift between surfaces becomes harder to detect.  
**Urgency**: HIGH  
**Debt class**: **active** — actively increasing maintenance cost with each feature addition

---

### TD-002 — Backend boundary violations (root-level imports from canonical services)

**Category**: structural  
**What it is**: The canonical `backend/` imports from 6+ root-level modules: `refs`, `text_tools`, `audio_defaults`, `output_paths`, `session_manager`, `tts_pipeline`. These root modules are documented as "compatibility/legacy surfaces" per `docs/CONVENTIONS.md`, yet the canonical backend depends on them.  
**Why it matters**: Root modules carry Gradio-context assumptions, global state, and lack the interface contracts the canonical backend needs. Any refactoring of root modules risks breaking canonical backend behavior without clear test coverage.  
**Current impact**: The backend cannot be deployed or tested without the full root module tree present. Module boundaries documented in `docs/CONVENTIONS.md` assert `backend/routes/* → backend/services/* → backend/schemas/*` layering, but in practice the services layer reaches outside `backend/`.  
**Likely future cost**: As root modules evolve (or are decommissioned), the backend's dependency on them creates migration blockers. Each root module import is a coupling point that must be untangled.  
**Urgency**: HIGH  
**Debt class**: **active** — violates documented architecture, blocks clean separation

---

### TD-003 — Triple-computed path constants

**Category**: code quality  
**What it is**: `WORK_DIR`, `OUTPUT_DIR`/`DEFAULT_OUTPUT_DIR` are independently computed from the same env vars (`VOCALIE_WORK_DIR`, `VOCALIE_OUTPUT_DIR`, `CHATTERBOX_OUT_DIR`) in three locations: `backend/config.py` (lines 61–67), `app.py` (lines 130–138), and `ui_gradio/gradio_helpers.py` (lines 34–40).  
**Why it matters**: If the env var parsing logic or defaults diverge, different parts of the system will resolve different paths silently. Current implementations are identical but fragile.  
**Current impact**: Any change to path resolution (e.g., adding a new env var alias or changing defaults) must be replicated in 3 places.  
**Likely future cost**: Path divergence bugs are notoriously hard to diagnose.  
**Urgency**: MEDIUM  
**Debt class**: **active** — duplication that must be maintained in lockstep

---

### TD-004 — Dual `clean_work_dir` with semantic divergence

**Category**: code quality  
**What it is**: `app.py` has `clean_work_dir` (line 332) with a BASE_DIR safety check and structured logging. `backend/services/work_service.py` has a different version lacking both the safety check and logging.  
**Why it matters**: The backend version could accept a `work_root` outside the repo without error, while the Gradio version would raise `ValueError`. Different callers get different safety guarantees.  
**Current impact**: Low — backend app.py calls the backend version at startup, and no production path calls the app.py version directly from the canonical API.  
**Likely future cost**: If someone adds a new cleanup caller and picks the wrong version, the safety gap surfaces.  
**Urgency**: LOW  
**Debt class**: **tolerable** — unlikely to cause immediate issues, but should converge

---

### TD-005 — Legacy preset bridge in canonical backend

**Category**: structural  
**What it is**: `backend/services/preset_service.py` contains `_legacy_to_ui_state()` with a hardcoded engine alias map (`{"chatterbox": "chatterbox_finetune_fr", "xtts": "xtts_v2"}`). The canonical backend actively translates legacy preset shapes into `UIState` models at read time.  
**Why it matters**: The canonical API's preset domain is polluted with legacy translation logic. The alias map is a separate copy from the one in `state_manager.py` and `engine_ui_helpers.py`, creating three separate places where `chatterbox` → `chatterbox_finetune_fr` is mapped.  
**Current impact**: New canonical IDs must be added to all three maps. Any canonical ID change requires updating all maps simultaneously.  
**Likely future cost**: Map divergence is inevitable over time. Removing legacy aliases requires auditing all three sites.  
**Urgency**: MEDIUM  
**Debt class**: **active** — three-way duplication of a critical business mapping

---

### TD-006 — In-memory job store with no persistence and dual job state

**Category**: data layer  
**What it is**: `JOB_STORE` in `backend/services/job_service.py` is a process-scoped Python dict. Backend restart loses all job state. Separately, `app.py` has `_JOB_STATE` + `_JOB_LOCK` for Gradio's own job tracking — a completely independent mechanism.  
**Why it matters**: Clients querying job status after a backend restart get 404s. No audit trail of past jobs exists. Two independent job state mechanisms create confusion about which is authoritative.  
**Current impact**: Acceptable for local-first single-user use, but makes debugging failed jobs harder (no history). The dual mechanism means API consumers and Gradio consumers have different job lifecycle semantics.  
**Likely future cost**: Multi-session or production deployment scenarios need persistent job history. Migration from dict to SQLite/JSON file is non-trivial since `JobStore` is referenced directly by routes.  
**Urgency**: MEDIUM  
**Debt class**: **tolerable** for current local-first model; becomes **active** if deployment model changes

---

### TD-007 — Frontend monolith component (`page.tsx`)

**Category**: structural  
**What it is**: `frontend/src/app/page.tsx` is 1675 lines with 34 `useState` hooks, 29+ function definitions, and covers the entire application UI in a single file.  
**Why it matters**: State management is tightly coupled. Any UI change risks unintended side effects across unrelated features. Review and test difficulty scales with size.  
**Current impact**: The frontend has only one page, so functional coupling is somewhat inherent. But 34 independent state variables in one component is a maintainability burden.  
**Likely future cost**: Adding new features (e.g., the LLM text improvement roadmap item) will further bloat this file. Without extraction, it will become increasingly difficult to isolate feature behavior.  
**Urgency**: MEDIUM  
**Debt class**: **active** — growing file with each feature addition

---

### TD-008 — Incomplete test coverage for backend services

**Category**: tests  
**What it is**: Three backend service modules have zero direct or indirect test coverage: `asset_service.py`, `work_service.py`, `audiosr_runner.py`. Two more have only indirect coverage through route tests: `preset_service.py`, `tts_service.py`. `backend/config.py` and `backend/routes/info.py` have no dedicated test files.  
**Why it is debt, not just a gap**: Asset service writes the metadata that downstream jobs depend on. Work service performs file-system cleanup. TTS service orchestrates the core generation pipeline. None have contract-level test coverage.  
**Current impact**: Bug in asset metadata format, work cleanup edge case, or TTS service orchestration will not be caught by CI.  
**Likely future cost**: Each of these services will grow with new features. Without contract tests, refactoring is high-risk.  
**Urgency**: MEDIUM  
**Debt class**: **active** — services that handle critical workflows are unprotected by tests

---

### TD-009 — Orphan and misplaced root-level files

**Category**: code quality  
**What it is**: Several root-level files serve unclear purposes or are misplaced: `tts_test.py` (orphan ad-hoc script), `gilles` (80KB HTML blob), `security_best_practices_report.md` (historical report in root, not `reports/`), `openapi.json` (auto-generated spec checked in without CI regeneration guard).  
**Why it matters**: Adds noise to the repo root. Violates documented convention that reports belong in `reports/`. `openapi.json` can silently go stale.  
**Current impact**: Low — does not affect functionality, but increases first-contact confusion.  
**Likely future cost**: Accumulation of more orphan files. Stale openapi.json leads to client integration errors.  
**Urgency**: LOW  
**Debt class**: **tolerable** — hygiene issue, no functional impact

---

### TD-010 — Runner print() statements in subprocess TTS backends

**Category**: deployment / ops  
**What it is**: `xtts_runner.py`, `bark_runner.py`, `bark_prefetch.py`, `piper_runner.py`, `xtts_prefetch.py` use `print()` for error/status output instead of structured logging.  
**Why it matters**: These runners execute as subprocesses. `print()` to stderr is the only practical mechanism for subprocess→parent communication. However, some print to stdout (`bark_backend.py:30` prints `[]` to stdout, `bark_prefetch.py:23` prints to stdout).  
**Current impact**: Stdout prints could interfere if the parent parses stdout as structured output. Currently no evidence of parsing stdout.  
**Likely future cost**: If a runner's stdout output is ever consumed programmatically, stray prints will corrupt the protocol.  
**Urgency**: LOW  
**Debt class**: **tolerable** — subprocess runners, no current integration risk

---

### TD-011 — Frontend type duplication vs backend schemas

**Category**: documentation  
**What it is**: `frontend/src/lib/types.ts` (189 lines, 27 exported types) mirrors `backend/schemas/models.py` (356 lines, 41 Pydantic models). The types are manually synchronized, not auto-generated from the OpenAPI spec.  
**Why it matters**: API contract changes (new fields, renamed fields, type changes) require manual updates in both places. No CI check verifies frontend/backend type alignment.  
**Current impact**: Moderate — the OpenAPI snapshot test exists but only verifies the spec hasn't changed unexpectedly. It does not verify frontend types match.  
**Likely future cost**: High — each API evolution doubles the chance of frontend/backend type divergence.  
**Urgency**: MEDIUM  
**Debt class**: **active** — manual synchronization of a critical contract

---

### TD-012 — Hardcoded development ports in scripts

**Category**: deployment / ops  
**What it is**: Default ports `8018` (backend), `3018` (frontend), `7860` (Gradio) are hardcoded in multiple scripts (`dev-backend.sh`, `dev-frontend.sh`, `dev-macos.sh`, `dev.sh`, `smoke.sh`, `status.sh`, `stop-km.sh`). Not configurable via env vars for all scripts. `cockpit.py` hardcodes `http://127.0.0.1:8018`.  
**Why it matters**: Port changes require editing multiple files. Not all scripts use `${API_PORT:-8018}` pattern.  
**Current impact**: Low for local-first single-user. Frustrating if deploying behind a proxy.  
**Likely future cost**: If deployment topology changes, manual port hunting across 7+ files.  
**Urgency**: LOW  
**Debt class**: **tolerable** — consistent for current use model

---

### TD-013 — `state_manager.py` shared between legacy and canonical surfaces without interface contract

**Category**: structural  
**What it is**: `state_manager.py` is imported by `app.py` (legacy), `ui_gradio/gradio_helpers.py` (extracted legacy helpers), and `tests/`. It provides `load_state`, `save_state`, `load_preset`, `save_preset`, `delete_preset`, `ensure_default_presets`, `migrate_state` — overlapping responsibilities with `backend/services/preset_service.py`.  
**Why it matters**: Two preset management systems exist with different interfaces. `state_manager.py` uses a flat JSON dict model. `backend/services/preset_service.py` uses Pydantic `UIState` models. They read the same files but apply different validation.  
**Current impact**: Preset data validation differs between Gradio and API consumers.  
**Likely future cost**: Preset format evolution must be implemented in both systems. Schema drift risk.  
**Urgency**: MEDIUM  
**Debt class**: **active** — dual preset management with different validation models

---

## Classification summary

| ID | Category | Debt class | Urgency | Description |
|----|----------|-----------|---------|-------------|
| TD-001 | structural | **active** | HIGH | Dual orchestration surface |
| TD-002 | structural | **active** | HIGH | Backend boundary violations (root imports) |
| TD-003 | code quality | **active** | MEDIUM | Triple-computed path constants |
| TD-004 | code quality | tolerable | LOW | Dual `clean_work_dir` divergence |
| TD-005 | structural | **active** | MEDIUM | Legacy preset bridge, 3-way alias map |
| TD-006 | data layer | tolerable→active | MEDIUM | In-memory job store, no persistence |
| TD-007 | structural | **active** | MEDIUM | Frontend monolith component |
| TD-008 | tests | **active** | MEDIUM | 5 backend services without direct tests |
| TD-009 | code quality | tolerable | LOW | Orphan/misplaced root files |
| TD-010 | deployment/ops | tolerable | LOW | Runner print() in subprocess backends |
| TD-011 | documentation | **active** | MEDIUM | Frontend/backend type sync is manual |
| TD-012 | deployment/ops | tolerable | LOW | Hardcoded dev ports in scripts |
| TD-013 | structural | **active** | MEDIUM | Dual preset management, different validation |

## Blocking debt

None identified. No single debt item blocks current development or deployment.

## Active debt (requires attention on next change to affected area)

1. **TD-001** — Dual orchestration surface. Each new engine/feature costs double.
2. **TD-002** — Backend boundary violations. Canonical backend cannot be self-contained.
3. **TD-003** — Path constant duplication. Must be maintained in lockstep.
4. **TD-005** — Three-way legacy alias map. Must be updated simultaneously.
5. **TD-007** — Frontend monolith. Growing with each feature.
6. **TD-008** — Backend services without contract tests. Refactoring is high-risk.
7. **TD-011** — Frontend/backend type sync has no CI verification.
8. **TD-013** — Dual preset management with different validation.

## Tolerable debt (can defer, but should not forget)

1. **TD-004** — Dual `clean_work_dir`: converge when next modifying cleanup logic.
2. **TD-006** — In-memory job store: acceptable for local-first, becomes active if deployment changes.
3. **TD-009** — Orphan root files: hygiene cleanup, no functional impact.
4. **TD-010** — Runner prints: acceptable for subprocess communication.
5. **TD-012** — Hardcoded ports: acceptable for current single-user local deployment.

## Recommended treatment (priority order)

| Priority | Items | Treatment | Effort |
|----------|-------|-----------|--------|
| P1 | TD-002, TD-003 | Move shared root modules into `backend/` or a new `shared/` package. Canonical backend must not import from root level. Consolidate path constants in `backend/config.py` only; remove duplicates from `app.py` and `gradio_helpers.py`. | L |
| P2 | TD-005, TD-013 | Extract engine alias map to a single source of truth (e.g., `backend/engine_catalog.py` or `tts_backends/catalog.py`). Both preset_service and state_manager import from it. Remove `_legacy_to_ui_state` from preset_service once all preset files are migrated to UIState format. | M |
| P3 | TD-008 | Add contract-level tests for `asset_service`, `work_service`, `tts_service`. These are the most critical untested paths. | M |
| P4 | TD-001 | Define explicit interface between Gradio cockpit and canonical API. Move Gradio cockpit to consume the API (not shared modules) for new features. Existing cockpit→API bridge (`ui_gradio/cockpit.py`) exists. | L |
| P5 | TD-007, TD-011 | Extract frontend state into a Zustand/context provider. Add CI check that generates TypeScript types from OpenAPI spec or validates frontend types against it. | M |
| P6 | TD-004, TD-006, TD-009, TD-010, TD-012 | Converge clean_work_dir. Add job persistence when deployment model changes. Clean orphan files. No action on runner prints. Parametrize ports in scripts. | S–M each |

## Codebase health indicators

| Metric | Value | Assessment |
|--------|-------|------------|
| Largest module | `app.py` 2978 lines | Documented legacy surface, 3.8% reduced from prior extraction |
| Frontend largest component | `page.tsx` 1675 lines | Active debt, growing |
| Backend→root imports | 12 import lines | Violates documented architecture |
| Unused imports | 15 | Surface noise, auto-fixable |
| Backend services without direct tests | 5 | Active debt |
| Duplicate implementations | 3 (path constants, clean_work_dir, alias map) | Active debt |
| TODO/FIXME in critical paths | 0 | Clean |
| Debug prints in production backend/frontend | 0 | Clean |
| Canonical docs | 9 files, CI-guarded | Strong |