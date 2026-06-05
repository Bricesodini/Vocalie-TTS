# Project Context

## What this repo is

Local-first TTS production stack with:
- FastAPI backend (`backend/`) — canonical runtime
- Next.js production frontend (`frontend/src/`)
- Optional debug cockpit (`ui_gradio/cockpit.py`) — not required for normal operation

## Current scope + non-goals

Canonical scope/non-goals are in `README.md` (`Perimetre d'audit (gele)`, `Out-of-scope / Non-goals`, `Cas d'usage critiques`).

Key non-goals:
- Public Internet exposure of the API without hardening/proxy controls
- Implicit text rewriting or implicit post-processing

## Critical invariants

See `docs/invariants.md`.
Key invariants include auth on protected routes, input/output bounds, rate-limit behavior, and explicit production security baseline.

## Architecture snapshot

- Canonical backend: `backend/` (`routes`, `services`, `schemas`, `workers`)
- TTS backends: `tts_backends/` (auto-registered via `__init_subclass__`)
- Production UI: `frontend/src/` (Next.js)
- Shared modules: `backend/shared/` (canonical location, re-exported via root shims)
- Debug cockpit: `ui_gradio/` (optional, not a production surface)

## Commands

- Tests: `pytest -q`
- Frontend lint/build: `cd frontend && npm run lint && npm run build`
- Security baseline: `bash ./scripts/check-security-baseline.sh --prod`
- Smoke: `bash ./scripts/smoke.sh`
- CI baseline: `.github/workflows/ci.yml`

## Conventions pointer

- `docs/CONVENTIONS.md`

## Top risks + mitigations

- Legacy/canonical drift for presets and engine IDs → glossary + decisions + boundary-only alias policy
- Config drift → centralized env policy (`docs/ENV_POLICY.md`) + CI guardrails

## Where truth lives

- `docs/INDEX.md`
- `docs/CONVENTIONS.md`
- `docs/DECISIONS.md`
- `docs/GLOSSARY.md`
- `docs/architecture.md`