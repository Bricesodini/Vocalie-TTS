# Project Context

## Purpose & scope

Chatterbox provides a local-first TTS production stack (API + frontend) focused on deterministic text-to-audio generation, optional reference voice usage, and optional audio enhancement.

Current product scope and non-goals are defined in `README.md` under:
- `Perimetre d'audit (gele)`
- `Out-of-scope / Non-goals`
- `Cas d'usage critiques`

## Non-goals (current)

- Public Internet exposure of the API without hardening/proxy controls.
- Gradio as production UI.
- Implicit text rewriting or implicit post-processing.

## Architecture snapshot

- Canonical backend runtime: `backend/` (`routes`, `services`, `schemas`, `workers`).
- Production UI: `frontend/src/` (Next.js).
- Compatibility/legacy surfaces still present at root (`app.py`, state/session legacy helpers).

## Key workflows / commands

- Setup/run: see `README.md`.
- Security baseline check: `bash ./scripts/check-security-baseline.sh --prod`
- Rate-limit fairness check: `python ./scripts/check-rate-limit-fairness.py`
- API smoke checks: `bash ./scripts/smoke.sh`
- CI baseline: `.github/workflows/ci.yml`

## Data & security highlights

- Invariants: `docs/invariants.md`
- System boundaries: `docs/system-boundaries.md`
- Security operations: `docs/security-runbook.md`
- Env governance: `docs/ENV_POLICY.md`

## Canonical links

- `docs/INDEX.md`
- `docs/CONVENTIONS.md`
- `docs/DECISIONS.md`
- `docs/GLOSSARY.md`
- `docs/CONTEXT.compact.md`
