# Environment Policy

Canonical source for runtime environment variables and their operational intent.

## Source of truth

- Runtime defaults and parsing: `backend/config.py`
- Path constants (`WORK_DIR`, `OUTPUT_DIR`, `PRESETS_DIR`): `backend/config.py` — the single source of truth; `app.py` and `ui_gradio/gradio_helpers.py` import from `backend.config`.
- Example values for local setup: `.env.example`
- Production validation procedure: `docs/security-runbook.md` + `scripts/check-security-baseline.sh`

## Core variables

- `VOCALIE_API_KEY`: API auth secret (required in production).
- `VOCALIE_TRUST_LOCALHOST`: local trust bypass (`0` in production).
- `VOCALIE_ENABLE_API_DOCS`: docs endpoints toggle (`0` in production).
- `VOCALIE_ALLOWED_HOSTS`: explicit allowed hosts.
- `VOCALIE_TRUSTED_PROXIES`: explicit proxy peers trusted for forwarding headers.
- `VOCALIE_MAX_UPLOAD_BYTES`: upload size cap.
- `VOCALIE_RATE_LIMIT_RPS`, `VOCALIE_RATE_LIMIT_BURST`: rate limit tuning.
- `VOCALIE_MAX_TEXT_CHARS`, `VOCALIE_MAX_CONCURRENT_JOBS`: workload safety bounds.

## Policy

- `backend/config.py` defines canonical defaults and parsers.
- `.env.example` must stay aligned with canonical variable names.
- README and runbooks reference this document instead of redefining variable semantics.
- CI and security baseline scripts must validate the same variable set.
