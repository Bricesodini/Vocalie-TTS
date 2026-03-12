# Decisions (ADR-light)

## DEC-001 — API-first backend as canonical runtime
- Status: Accepted
- Date: 2026-03-12
- Decision: `backend/` is the canonical backend runtime surface.
- Why: clearer boundaries (`routes -> services -> schemas`) and better auditability.
- Tradeoffs: root legacy surfaces remain during transition.
- Evidence links: `reports/audit-readiness-20260311-2211.md`, `docs/reports/TECH_DEBT_AUDIT_2026-03-11.md`

## DEC-002 — Scope freeze is mandatory and CI-guarded
- Status: Accepted
- Date: 2026-03-12
- Decision: README must keep explicit scope freeze/non-goals sections validated in CI.
- Why: preserves stable audit perimeter.
- Tradeoffs: adds documentation gate overhead.
- Evidence links: `reports/scope-freeze-20260311-2211.md`, `.github/workflows/ci.yml`

## DEC-003 — Production UI is Next.js; Gradio is compatibility cockpit
- Status: Accepted
- Date: 2026-03-12
- Decision: frontend production flows are carried by `frontend/src`; Gradio remains non-production compatibility surface.
- Why: aligns architecture with current operational model.
- Tradeoffs: dual surfaces require migration discipline.
- Evidence links: `README.md`, `docs/reports/TECH_DEBT_AUDIT_2026-03-11.md`

## DEC-004 — Canonical environment policy is centralized
- Status: Accepted
- Date: 2026-03-12
- Decision: `docs/ENV_POLICY.md` references canonical runtime defaults from `backend/config.py`.
- Why: reduces config-sprawl drift.
- Tradeoffs: docs must be maintained with config changes.
- Evidence links: `reports/code-janitor-20260311-2214.md`, `docs/security-runbook.md`

## DEC-005 — Canonical engine IDs with legacy aliases at boundary only
- Status: Proposed
- Date: 2026-03-12
- Decision: keep canonical API engine IDs; confine legacy aliases to compatibility adapters.
- Why: reduce domain naming drift.
- Tradeoffs: requires explicit migration handling.
- Evidence links: `docs/reports/DOMAIN_MODEL_DRIFT_REPORT.md`, `reports/code-janitor-20260311-2214.md`

## DEC-006 — Reports are immutable evidence
- Status: Accepted
- Date: 2026-03-12
- Decision: `reports/` and `docs/reports/` are historical evidence, never rewritten.
- Why: preserve audit traceability.
- Tradeoffs: report volume grows over time.
- Evidence links: `docs/INDEX.md`, `reports/doc-context-20260311-2216.md`
