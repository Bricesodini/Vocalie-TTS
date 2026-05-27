# Conventions — Chatterbox

## Goals

- Predictability
- Reduced review friction
- Reduced drift / entropy

## Decisions (normative)

### 1) Project structure

- `backend/` is the canonical API runtime surface (`routes`, `services`, `schemas`, `workers`, `shared`).
- `backend/shared/` contains modules shared between canonical backend and compatibility surfaces (`refs`, `text_tools`, `audio_defaults`, `output_paths`, `session_manager`, `tts_pipeline`). Root-level re-export shims preserve backward compatibility.
- Root monolith files (`app.py`, `state_manager.py`, `tts_engine.py`) are compatibility/legacy surfaces and must not absorb new product scope.
- `frontend/src/` hosts production UI concerns only.
- `scripts/` contains operational/developer automation; script behavior must reference canonical env policy.
- `tests/` mirrors risk areas (API/security/rate-limit/state/backends) and remains colocated by domain responsibility.
- `docs/` contains living source-of-truth documentation.
- `reports/` contains immutable timestamped audit outputs.

Examples as plain text paths:
- `backend/routes/tts.py`
- `backend/services/preset_service.py`
- `backend/shared/text_tools.py`
- `frontend/src/lib/api.ts`
- `docs/invariants.md`
- `reports/code-janitor-20260311-2214.md`

### 2) Naming

- Python files/functions/modules: `snake_case`.
- TypeScript components: `kebab-case` files under `components/ui/` with exported PascalCase symbols allowed.
- API paths remain versioned under `/v1/*`.
- Canonical engine IDs must be stable and explicit (`chatterbox_native`, `chatterbox_finetune_fr`, `xtts_v2`, `piper`, `bark`, `qwen3_custom`, `qwen3_clone`).
- Legacy aliases (`chatterbox`, `xtts`) are compatibility-only and must stay at migration boundaries.
- Tests: `tests/test_<domain>.py` naming required.

### 3) Imports & boundaries

- `backend/routes/*` may depend on `backend/services/*`, `backend/schemas/*`, `backend/shared/*`, and shared backend utilities.
- `backend/services/*` must not depend on route modules.
- `backend/schemas/*` must remain domain contract definitions with no route-side effects.
- `backend/shared/*` contains shared domain modules (text processing, path helpers, session management, TTS pipeline). Root-level shims re-export these for backward compatibility.
- Frontend `lib/` is API/types/util boundary; page-level components should not duplicate request contract types.
- No-crossing rule: new backend business logic should not be added into root `app.py`.
- No-crossing rule: `backend/` modules must import from `backend.shared` rather than from root-level modules.

### 4) Configuration

- Canonical runtime defaults live in `backend/config.py`.
- `.env.example` documents supported env variables and safe defaults for local setup.
- Ops/security constraints live in `docs/security-runbook.md`; README links and summarizes, but does not redefine policy semantics.
- CI env checks must align with the same variable names and constraints.
- Secrets policy: no secrets in repository; placeholders only.

### 5) Logging / debug

- Structured logger usage is preferred in backend runtime paths.
- `print`/`console.*` in production paths is disallowed unless explicitly justified for operator-visible diagnostics.
- Temporary debug logs must be time-bounded and removed before merge.
- Temporary flags require explicit owner, purpose, and removal condition in documentation or issue tracking.

### 6) Documentation

- Root `README.md` is entrypoint (purpose, setup, architecture snapshot, scope, run commands).
- `docs/` holds canonical details (`invariants`, `system-boundaries`, conventions, future canonical context docs).
- `reports/` remains immutable evidence and should not be edited after generation.
- Required README sections: scope/non-goals, critical use cases, architecture snapshot, install/run, security references, troubleshooting.

## Drift checklist (to use in reviews)

- Does the change keep business logic out of `app.py` unless compatibility-only?
- Are new APIs under `/v1/*` and consistent with existing route conventions?
- Are engine IDs using canonical names (not introducing new ad-hoc aliases)?
- Are legacy compatibility mappings localized rather than spread?
- Is configuration sourced from `backend/config.py` with documented env keys?
- Are docs updated when scope/invariants/boundaries change?
- Are tests added/updated for changed risk areas?
- Are debug logs absent or clearly justified and temporary?
- Are files placed in the correct responsibility folder?
- Are imports respecting route -> service -> schema -> shared layering (no root-level imports in `backend/`)?
- Is any duplicate helper logic introduced instead of reusing shared utilities?
- Are CI checks still aligned with documented operational policy?

## Migration plan (mechanical)

- Step 1 (DONE): Move shared root modules into `backend/shared/` and re-export via root shims; affected paths: `refs.py`, `text_tools.py`, `audio_defaults.py`, `output_paths.py`, `session_manager.py`, `tts_pipeline.py` now at `backend/shared/`, with shims at root. Path constants consolidated in `backend/config.py` only (TD-002, TD-003).
- Step 2: Build canonical naming matrix for engines/presets and list all alias sites; affected paths: `backend/routes/tts.py`, `backend/services/preset_service.py`, `state_manager.py`, `app.py`.
  - Risk note: alias removal too early can break old payloads.
  - Rollback hint: preserve compatibility mapping table until consumers are migrated.
- Step 2: Mark root monolith files as compatibility-boundary in docs and PR checklist; affected paths: `README.md`, `docs/CONVENTIONS.md`.
  - Risk note: accidental new features may still land there.
  - Rollback hint: block via review gate before merge.
- Step 3: Consolidate duplicated helper conventions (time helper/util ownership); affected paths: `backend/services/*`, `backend/routes/*`.
  - Risk note: inconsistent helper semantics.
  - Rollback hint: retain old helper wrappers temporarily.
- Step 4: Centralize env policy references and remove duplicated normative statements; affected paths: `backend/config.py`, `.env.example`, `README.md`, `docs/security-runbook.md`, `.github/workflows/ci.yml`.
  - Risk note: docs/runtime divergence.
  - Rollback hint: keep previous docs references until CI passes.
- Step 5: Enforce logging/debug hygiene review gate; affected paths: `frontend/src/lib/api.ts`, `frontend/src/app/page.tsx`, backend runtime modules.
  - Risk note: operator diagnostics loss if removed blindly.
  - Rollback hint: reintroduce controlled logger-level diagnostics.
- Step 6: Normalize test naming and placement for new tests only; affected paths: `tests/`.
  - Risk note: moving existing tests may create churn.
  - Rollback hint: limit to forward-only convention.
- Step 7: Re-audit drift monthly using janitor + format/lint report chain; affected paths: `reports/`, `docs/`.
  - Risk note: drift returns without cadence.
  - Rollback hint: pin a recurring audit checklist in team workflow.

## Unknowns / open questions

- UNKNOWN whether root `app.py` is intended for long-term maintenance or sunset.
- UNKNOWN whether all external clients already consume canonical API model IDs only.
- UNKNOWN whether an explicit ADR process will be adopted for convention changes.
