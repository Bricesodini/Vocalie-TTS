## Summary

- [ ] Scope of change is explicit (no hidden feature behavior changes).

## Ready Audit-Gate Checklist

### Scope & perimeter alignment

- [ ] If scope changed: `README.md` scope/non-goals/use-cases updated.
- [ ] If domain behavior changed: `docs/invariants.md` updated (or explicitly N/A).
- [ ] If system boundaries changed: `docs/system-boundaries.md` updated (or explicitly N/A).
- [ ] If env vars or config changed: `docs/ENV_POLICY.md` and `.env.example` updated (or explicitly N/A).
- [ ] If naming or module structure changed: `docs/CONVENTIONS.md` and `docs/GLOSSARY.md` updated (or explicitly N/A).

### Compatibility & drift

- [ ] `docs/CONVENTIONS.md` still respected by changed files.
- [ ] Compatibility layer impact reviewed (engine IDs / presets aliases remain boundary-only).
- [ ] No new business logic added to root `app.py` (compatibility surface only).

### Security & hygiene

- [ ] No uncontrolled debug logs added in production frontend paths (`src/app/page.tsx`, `src/lib/api.ts`).
- [ ] No secrets or placeholder API keys committed.
- [ ] If new env vars introduced: added to `.env.example`, `backend/config.py`, and `docs/ENV_POLICY.md`.

### Validation

- [ ] `pytest -q`
- [ ] `cd frontend && npm run lint && npm run build`
- [ ] `bash ./scripts/check-security-baseline.sh --prod` (with prod-like env vars)