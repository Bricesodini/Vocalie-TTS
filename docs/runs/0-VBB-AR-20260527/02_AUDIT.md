---
run_id: "0-VBB-AR-20260527"
phase: "02_AUDIT"
route: "audit-readiness"
status: "READY"
agent: "0-vbb-audit-readiness"
started_at: "2025-05-27T14:25:00Z"
ended_at: "2025-05-27T14:27:00Z"
next_phase: null
artifacts_consumed:
  - "README.md"
  - "docs/invariants.md"
  - "docs/system-boundaries.md"
  - "docs/ENV_POLICY.md"
  - "docs/INDEX.md"
  - "docs/CONVENTIONS.md"
  - "docs/DECISIONS.md"
  - "docs/GLOSSARY.md"
  - "docs/CONTEXT.md"
  - "docs/CONTEXT.compact.md"
  - "docs/security-runbook.md"
  - ".env.example"
  - "SECURITY.md"
  - "backend/config.py"
  - ".github/workflows/ci.yml"
  - "reports/audit-readiness-20260312-0635.md"
  - "reports/scope-freeze-20260311-2211.md"
  - "docs/reports/TECH_DEBT_AUDIT_2026-03-11.md"
  - "docs/reports/DOMAIN_MODEL_DRIFT_REPORT.md"
artifacts_produced:
  - "docs/runs/0-VBB-AR-20260527/02_AUDIT.md"
  - "docs/audits/audit-readiness-20260527-1427.md"
  - "docs/AUDIT_STATUS.md"
---

# Audit Readiness Report — Vocalie-TTS — 2025-05-27

## Executive summary

The project is **READY** for audit. Scope is frozen and CI-guarded, 6 canonical documentation files exist and are cross-referenced, critical invariants are explicitly identified with code/test references, system boundaries are documented, and environment policy is centralized. The main residual risk is the continued coexistence of a legacy monolith surface alongside the canonical backend, which is well-documented and tracked via ADRs and tech debt reports.

## Global verdict

**READY**

Rationale: all 6 domains (A→F) produce actionable findings. No domain is in BLOCKED or fully UNKNOWN state. The project has undergone prior audit readiness cycles (20260311, 20260312) that addressed earlier gaps; the current state reflects accumulated hardening.

Prior readiness reports reached the same verdict. This assessment independently confirms readiness against the 6 domains with current evidence.

---

## Findings by domain

### A) Functional stability — READY

**Evidence:**
- Scope freeze is explicit in `README.md` (sections "Perimetre d'audit (gele)", "Out-of-scope / Non-goals", "Cas d'usage critiques").
- Non-goals are product-level and specific (no public Internet exposure, no implicit text rewrite, no auto post-processing, Gradio is not production UI).
- Roadmap items are explicitly tagged as "hors perimetre d'audit courant".
- CI workflow (`.github/workflows/ci.yml`) contains a "Scope freeze docs guardrail" step that verifies the presence of required scope sections.
- No `TODO`, `FIXME`, `WIP`, `TBD`, or `HACK` markers found in `backend/` core modules or `docs/`.

**Residual risk:** Future additions could drift scope if CI guardrails are bypassed locally. Mitigated by PR-based workflow requirement in CI.

### B) Structural readability — READY

**Evidence:**
- `backend/` has clear separation: `routes/` (11 route modules), `services/` (6 service modules), `schemas/` (1 models module), `workers/` (1 runner), `utils/`, `security.py`, `config.py`, `rate_limit.py`.
- `frontend/src/` has `app/`, `components/`, `lib/` — standard Next.js structure.
- `scripts/` contains operational tooling (bootstrap, dev, smoke, security checks, doctor).
- `tests/` mirrors risk areas (API auth, jobs, rate-limit, security baseline, engine schema, backends, chunks, etc.).
- `docs/` contains canonical living documentation.
- `reports/` contains immutable timestamped audit evidence.

**Known gap (documented, not blocking):** Root-level monolith files (`app.py` at 3096 lines, `state_manager.py`, `session_manager.py`, `tts_engine.py`, `tts_pipeline.py`) are explicitly documented as "compatibility/legacy surfaces" per `docs/CONVENTIONS.md` with a no-new-responsibility rule. This dual-surface coexistence is tracked in ADR DEC-001 and tech debt report.

### C) Minimal documentation — READY

**Evidence:**
- README.md: comprehensive (architecture, scope, non-goals, critical use cases, quickstart, troubleshooting, run commands, scripts, API examples, security references).
- `docs/` contains 9 canonical documents: CONVENTIONS.md, CONTEXT.md, CONTEXT.compact.md, DECISIONS.md, GLOSSARY.md, INDEX.md, invariants.md, system-boundaries.md, ENV_POLICY.md, security-runbook.md.
- All canonical docs are cross-referenced from INDEX.md.
- CI "Canonical docs guardrail" step verifies existence of all required docs files and their references in README.
- ADRs (DEC-001 through DEC-006) document architectural decisions with status, tradeoffs, and evidence links.

**No gaps identified.** Documentation goes well beyond minimal; the project has a mature doc hierarchy.

### D) Boundary clarity — READY

**Evidence:**
- `docs/system-boundaries.md` explicitly lists: inputs (API HTTP, reference files, env vars, payloads), outputs (generated audio, temp files, metadata), external dependencies (Hugging Face models, ffmpeg, local Python/Node), network/trust boundaries, internal application boundaries, and explicit non-goals.
- `.env.example` documents all supported environment variables with safe defaults.
- `docs/ENV_POLICY.md` is the canonical reference for env var semantics, aligning README, runbook, config, and CI.
- API inputs/outputs are versioned under `/v1/*` with explicit output format bound to WAV (invariant #3).

### E) Critical invariants visible — READY

**Evidence:**
- `docs/invariants.md` lists 7 invariants (4 functional, 3 operational) with explicit:
  - Rule statement
  - Source code reference (file + function/variable)
  - Verification method (specific test files, scripts, or CI steps)
- Invariants cover: auth on protected routes, text size limits, output format bounds, engine parameter hygiene, rate limiting, production security baseline, and stable storage boundaries.
- Governance note requires any invariant evolution to update both the document and its associated verification source.

**No unknown invariants.** All critical business rules are documented with traceable evidence paths.

### F) Environment clarity — READY

**Evidence:**
- Stack is identifiable without executing code: Python 3.11, Node.js >= 20, FastAPI backend, Next.js frontend, ffmpeg, Hugging Face model caching.
- `.env.example` exists with all 14 environment variables documented with default values.
- `backend/config.py` is the canonical source of runtime defaults and parsers, with clear env var mapping.
- `docs/ENV_POLICY.md` documents the canonical variable set and alignment requirements.
- DEV/PROD differences are explicitly recognized:
  - `VOCALIE_TRUST_LOCALHOST=1` / `VOCALIE_TRUST_LOCALHOST=0`
  - `VOCALIE_ENABLE_API_DOCS` / `VOCALIE_EXPOSE_SYSTEM_INFO` toggles
  - `scripts/check-security-baseline.sh --prod` for production baseline validation
  - CI runs both secure and insecure configuration checks
- `scripts/dev.sh`, `scripts/dev-macos.sh`, `scripts/dev-windows.ps1` provide platform-specific dev entry points.

---

## Recommended corrective actions

| ID | Priority | Type | Description | Effort |
|----|----------|------|-------------|--------|
| AR-01 | LOW | governance-continuity | Add PR checklist item requiring simultaneous update of `README.md`, `docs/invariants.md`, and `docs/system-boundaries.md` for any scope perimeter change. | S |
| AR-02 | LOW | maintainability-watch | Continue bounded mechanical extraction from `app.py` root monolith (currently 3096 lines) per documented conventions. | M |
| AR-03 | LOW | maintainability-watch | Maintain monthly janitor + format/lint + doc-context cadence to prevent regression (per READY-01 of prior readiness report). | S (recurring) |

---

## UNKNOWN / evidence gaps

| Item | Domain | Status |
|------|--------|--------|
| External governance (release checklist / approval gate outside repo) | A | UNKNOWN whether required by organization; not required for READY verdict |
| Long-term target for root compatibility surface (`app.py`) | B | UNKNOWN whether intended for long-term maintenance or eventual full sunset; documented in CONVENTIONS.md as compatibility layer |
| ADR process adoption for future convention changes | B | UNKNOWN whether explicit ADR workflow will be adopted (noted in CONVENTIONS.md open questions) |
| External client consumption of canonical API model IDs only | A | UNKNOWN whether all external consumers use canonical IDs; mitigated by boundary-only alias policy (DEC-005) |

None of these UNKNOWNs block audit readiness. They are documented risks that a deeper audit could investigate.