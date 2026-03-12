# Context Compact

## What this repo is

Local-first TTS production stack with:
- FastAPI backend (`backend/`)
- Next.js production frontend (`frontend/src/`)
- Compatibility Gradio surface (`app.py`)

## Current scope + non-goals

Canonical scope/non-goals are in `README.md` (`Perimetre d'audit (gele)`, `Out-of-scope / Non-goals`, `Cas d'usage critiques`).

## Critical invariants

See `docs/invariants.md`.
Key invariants include auth on protected routes, input/output bounds, rate-limit behavior, and explicit production security baseline.

## Commands

- Tests: `pytest -q`
- Frontend lint/build: `cd frontend && npm run lint && npm run build`
- Security baseline: `bash ./scripts/check-security-baseline.sh --prod`
- Smoke: `bash ./scripts/smoke.sh`

## Conventions pointer

- `docs/CONVENTIONS.md`

## Top risks + mitigations

- Monolith hotspots (`app.py`, `frontend/src/app/page.tsx`) -> mechanical extraction and no-new-responsibility rule.
- Legacy/canonical drift for presets and engine IDs -> glossary + decisions + boundary-only alias policy.
- Config drift -> centralized env policy (`docs/ENV_POLICY.md`) and CI guardrails.

## Where truth lives

- `docs/INDEX.md`
