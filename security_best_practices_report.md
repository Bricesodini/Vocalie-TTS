# Security Best Practices Review Report

## Executive summary

As of 2026-02-09, all previously reported code-level findings (`SBP-001` to `SBP-007`) are implemented in the repository and should be treated as closed at code level.

The remaining risk is primarily deployment drift (proxy limits, host allowlists, environment flags, and production secret hygiene). This report now separates verified fixes from operational follow-up items.

## Scope and evidence

- Backend: FastAPI (`/Users/bricesodini/01_ai-stack/Chatterbox/backend`)
- Frontend: Next.js (`/Users/bricesodini/01_ai-stack/Chatterbox/frontend`)
- Validation date: 2026-02-09
- Repository state checked at commit: `f80fb94`

## Verified remediations (closed)

### SBP-001
- Rule ID: `FASTAPI-AUTH-001`
- Status: Closed (code-level)
- Evidence:
  - `VOCALIE_TRUST_LOCALHOST` is optional and default-off: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/config.py:54`
  - API key required when localhost trust is not enabled: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/security.py:49`
  - Auth is applied via route dependencies on `/v1/*` business routes: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/app.py:67`

### SBP-002
- Rule ID: `FASTAPI-DOS-INPUT-001`
- Status: Closed (code-level)
- Evidence:
  - Upload size cap env var present: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/config.py:47`
  - Streamed upload enforces limit and returns `413`: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/routes/audio.py:38`
  - Limit is used by `/v1/audio/enhance`: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/routes/audio.py:179`
  - Test coverage for oversized upload exists: `/Users/bricesodini/01_ai-stack/Chatterbox/tests/test_api_audio_edit.py:35`

### SBP-003
- Rule ID: `FASTAPI-OPENAPI-001`
- Status: Closed (code-level)
- Evidence:
  - API docs are disabled by default: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/config.py:53`
  - Docs URLs only enabled if explicit flag is set: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/app.py:38`

### SBP-004
- Rule ID: `FASTAPI-HOST-001`
- Status: Closed (code-level)
- Evidence:
  - Host allowlist env var and middleware wiring exist: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/config.py:40`, `/Users/bricesodini/01_ai-stack/Chatterbox/backend/app.py:47`

### SBP-005
- Rule ID: `NEXT-HEADERS-001`
- Status: Closed (code-level)
- Evidence:
  - Production security headers configured in Next.js: `/Users/bricesodini/01_ai-stack/Chatterbox/frontend/next.config.ts:11`

### SBP-006
- Rule ID: `FASTAPI-INFO-LEAK-001`
- Status: Closed (code-level)
- Evidence:
  - System info exposure is gated and defaults to hidden: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/config.py:55`, `/Users/bricesodini/01_ai-stack/Chatterbox/backend/routes/info.py:24`

### SBP-007
- Rule ID: `FASTAPI-AUTH-HARDENING-001`
- Status: Closed (code-level)
- Evidence:
  - API key comparison uses constant-time primitive: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/security.py:57`

## Open operational findings (deployment/runtime)

### OPS-001
- Severity: High
- Topic: Ingress/proxy body size alignment
- Risk: If upstream request-body limit is missing or higher than app expectations, oversized traffic can still pressure backend resources.
- Required control:
  - Set proxy body-size limit at or below `VOCALIE_MAX_UPLOAD_BYTES`.
  - Keep value consistent across environments.
- Validation:
  - Send payload above limit and confirm a rejection before expensive processing.

### OPS-002
- Severity: High
- Topic: Production API key governance
- Risk: Missing or weak `VOCALIE_API_KEY` causes either hard failures (availability risk) or weak auth posture if unsafe local defaults are used.
- Required control:
  - Enforce non-empty strong `VOCALIE_API_KEY` in prod deployment manifests/secrets.
  - Ensure `VOCALIE_TRUST_LOCALHOST=0` in all non-dev environments.
- Validation:
  - Startup/deploy checks verify env values before rollout.
  - Synthetic request without API key returns `403`.

### OPS-003
- Severity: Medium
- Topic: Host allowlist hygiene
- Risk: Broad or incorrect `VOCALIE_ALLOWED_HOSTS` values can reduce host-header protections.
- Required control:
  - Maintain explicit production FQDNs and service hostnames only.
- Validation:
  - Requests with unexpected `Host` header are rejected.

### OPS-004
- Severity: Medium
- Topic: Documentation surface in production
- Risk: Accidental enablement of docs/OpenAPI increases unauthenticated reconnaissance surface.
- Required control:
  - Keep `VOCALIE_ENABLE_API_DOCS=0` in production by policy.
- Validation:
  - `/v1/docs`, `/v1/redoc`, `/v1/openapi.json` return non-200 in prod.

### OPS-005
- Severity: Medium
- Topic: Rate-limit identity behind reverse proxy
- Risk: Current limiter keys on app-visible client host; behind some proxy topologies this can collapse many clients into one identity.
- Required control:
  - Define trusted proxy strategy and stable client identity keying (for example API key + validated forwarded IP).
- Validation:
  - Load test from multiple clients confirms predictable and fair throttling behavior.

## Improvement plan (prioritized)

1. P0 - Enforce deployment guardrails in CI/CD manifests
- Add explicit checks for:
  - `VOCALIE_API_KEY` present in prod
  - `VOCALIE_TRUST_LOCALHOST=0` in prod
  - `VOCALIE_ENABLE_API_DOCS=0` in prod
  - `VOCALIE_ALLOWED_HOSTS` not wildcard and not empty in prod
- Acceptance criteria: deployment pipeline fails on non-compliant configuration.

2. P0 - Align ingress limits with app limits
- Configure reverse proxy/body-size caps to match `VOCALIE_MAX_UPLOAD_BYTES`.
- Acceptance criteria: oversized request is rejected at edge and app behavior remains `413` if edge is bypassed.

3. P1 - Add production security regression tests
- Add tests for:
  - docs disabled by default,
  - auth required on protected routes,
  - host allowlist behavior,
  - upload-limit rejection path.
- Acceptance criteria: tests run in CI and block merge on failure.

4. P1 - Harden rate-limit identity model
- Update limiter design for proxy-aware deployments and document trusted headers policy.
- Acceptance criteria: fairness and abuse-resistance validated in staging load tests.

5. P2 - Operational runbook
- Create a short security runbook in README or dedicated doc for environment variables, expected values, and validation commands.
- Acceptance criteria: new environment can be validated end-to-end from the runbook alone.

## Positive practices observed

- Path safety checks are present for user-controlled file paths: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/routes/audio.py:23`
- CORS wildcard is explicitly ignored: `/Users/bricesodini/01_ai-stack/Chatterbox/backend/app.py:52`
- Subprocess calls use argument arrays (no `shell=True`) in reviewed paths.
