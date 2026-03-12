## Summary

- [ ] Scope of change is explicit (no hidden feature behavior changes).

## Ready Audit-Gate Checklist

- [ ] If scope changed: `README.md` scope/non-goals/use-cases updated.
- [ ] If domain behavior changed: `docs/invariants.md` updated (or explicitly N/A).
- [ ] If system boundaries changed: `docs/system-boundaries.md` updated (or explicitly N/A).
- [ ] `docs/CONVENTIONS.md` still respected by changed files.
- [ ] Compatibility layer impact reviewed (engine IDs / presets aliases remain boundary-only).
- [ ] No uncontrolled debug logs added in production frontend paths (`src/app/page.tsx`, `src/lib/api.ts`).
- [ ] CI checks pass (backend tests/security + frontend lint/build + guardrails).

## Validation

- [ ] `pytest -q`
- [ ] `cd frontend && npm run lint && npm run build`

