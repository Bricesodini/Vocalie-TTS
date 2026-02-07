# Security Best Practices Review Report

## Executive summary

This review focused on the Python FastAPI backend and the Next.js frontend. The most important risks are: (1) authentication bypass behavior tied to loopback detection, (2) unrestricted upload size in `/v1/audio/enhance` leading to denial-of-service risk, and (3) unprotected OpenAPI/docs exposure outside the authenticated `/v1/*` middleware gate.  
Overall, the codebase already avoids several high-risk classes (no `shell=True`, safe path checks in key file handlers, strict CORS wildcard handling), but it should be hardened for production-default deployment.

## Remediation status (2026-02-07)

- `SBP-001` Fixed: auth now enforces API key by default and only allows localhost bypass if explicitly enabled (`VOCALIE_TRUST_LOCALHOST=1`), with constant-time compare.
- `SBP-002` Fixed: upload size limit added for `/v1/audio/enhance` via `VOCALIE_MAX_UPLOAD_BYTES`, returning `413` on overflow.
- `SBP-003` Fixed: OpenAPI/docs disabled by default (`VOCALIE_ENABLE_API_DOCS=0`).
- `SBP-004` Fixed: `TrustedHostMiddleware` added with env-driven allowlist (`VOCALIE_ALLOWED_HOSTS`).
- `SBP-005` Fixed: production security headers added in Next.js config.
- `SBP-006` Fixed: `/v1/info` now hides sensitive system/path details by default (`VOCALIE_EXPOSE_SYSTEM_INFO=0`).
- `SBP-007` Fixed: API key compare now uses `hmac.compare_digest`.

## Residual checks to validate in deployment

- Verify reverse proxy/body-size limits align with `VOCALIE_MAX_UPLOAD_BYTES`.
- Verify only intended hosts are present in `VOCALIE_ALLOWED_HOSTS`.
- Verify API key is set in all non-dev environments (absence now blocks protected endpoints).
- Verify whether docs should remain disabled in production (`VOCALIE_ENABLE_API_DOCS=0` recommended).

## Critical findings

### SBP-001
- Rule ID: `FASTAPI-AUTH-001`
- Severity: Critical
- Location: `backend/security.py:13`, `backend/security.py:47`, `backend/app.py:46`
- Evidence:
```python
# backend/security.py
def is_local_request(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", None)
    ...
    return host in LOCAL_HOSTS

def is_authorized(request: Request) -> bool:
    if is_local_request(request):
        return True
```
```python
# backend/app.py
if request.url.path.startswith("/v1") and not is_authorized(request):
    return JSONResponse(status_code=403, content={"detail": "forbidden"})
```
- Impact: If deployed behind a local reverse proxy/load balancer where app-visible client IP is loopback (`127.0.0.1`/`::1`), remote callers may be treated as trusted and bypass API key checks.
- Fix: Remove implicit trust by client host; require explicit API key for all non-health endpoints, or gate local bypass behind an explicit env flag defaulting to disabled.
- Mitigation: Restrict network exposure to loopback-only until this is fixed.
- False positive notes: Safe only if deployment guarantees no proxy path that presents external traffic as loopback.

## High findings

### SBP-002
- Rule ID: `FASTAPI-DOS-INPUT-001` (size/abuse hardening)
- Severity: High
- Location: `backend/routes/audio.py:42`, `backend/routes/audio.py:49`, `backend/routes/audio.py:139`
- Evidence:
```python
async def _save_upload(upload: UploadFile) -> Path:
    ...
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        handle.write(chunk)
```
- Impact: Unbounded multipart upload can exhaust disk/memory/CPU (especially with ffmpeg + AudioSR processing), resulting in service degradation or outage.
- Fix: Enforce strict max upload size (server + app level), reject excessive `Content-Length`, and stop stream once limit is exceeded.
- Mitigation: Add reverse-proxy request body limits and tighter route-level rate limiting for enhancement endpoints.
- False positive notes: Lower risk if strict body limits already exist at ingress; not visible in repository.

### SBP-003
- Rule ID: `FASTAPI-OPENAPI-001`
- Severity: High
- Location: `backend/app.py:27`, `backend/app.py:46`
- Evidence:
```python
app = FastAPI(title="Chatterbox TTS API", version="0.1.0", lifespan=lifespan)
...
if request.url.path.startswith("/v1") and not is_authorized(request):
    return JSONResponse(status_code=403, content={"detail": "forbidden"})
```
- Impact: `/docs`, `/redoc`, and `/openapi.json` are enabled by default and are outside the `/v1` auth gate, exposing full API surface and schemas to unauthenticated users.
- Fix: In production, set `docs_url=None`, `redoc_url=None`, `openapi_url=None` or enforce auth for docs routes.
- Mitigation: Restrict docs endpoints via reverse proxy allowlist until app-level change is deployed.
- False positive notes: Acceptable in local-only dev deployments.

## Medium findings

### SBP-004
- Rule ID: `FASTAPI-HOST-001`
- Severity: Medium
- Location: `backend/app.py:35`
- Evidence:
```python
app.add_middleware(
    CORSMiddleware,
    ...
)
```
- Impact: No host header allowlist middleware (`TrustedHostMiddleware`) is configured; this can increase risk around host-header abuse and cache/proxy confusion in some deployments.
- Fix: Add `TrustedHostMiddleware` with explicit allowed hosts per environment.
- Mitigation: Enforce host validation at ingress/proxy layer.
- False positive notes: If upstream strictly validates `Host`, impact is reduced.

### SBP-005
- Rule ID: `NEXT-HEADERS-001`
- Severity: Medium
- Location: `frontend/next.config.ts:5`
- Evidence:
```ts
const nextConfig: NextConfig = {
  turbopack: { root: __dirname },
  async rewrites() { ... }
};
```
- Impact: No baseline security headers are configured in-app (CSP, `X-Content-Type-Options`, frame protections), increasing impact if a frontend injection vector appears later.
- Fix: Add `headers()` in `next.config.ts` (or equivalent edge config) with a strict baseline, especially CSP and anti-mime-sniffing headers.
- Mitigation: Apply headers at CDN/reverse-proxy if that is your deployment control plane.
- False positive notes: Header controls may already exist at hosting edge; not visible in repository.

## Low findings

### SBP-006
- Rule ID: `FASTAPI-INFO-LEAK-001`
- Severity: Low
- Location: `backend/routes/info.py:28`, `backend/routes/info.py:30`
- Evidence:
```python
python=platform.python_version(),
os=platform.platform(),
work_dir=str(WORK_DIR),
output_dir=str(OUTPUT_DIR),
presets_dir=str(PRESETS_DIR),
```
- Impact: Environment fingerprinting and local filesystem path disclosure can aid attacker recon.
- Fix: Return a minimal production-safe info payload (service name/version/capabilities only).
- Mitigation: Restrict endpoint access to trusted callers.
- False positive notes: Less concerning for isolated local-only deployments.

### SBP-007
- Rule ID: `FASTAPI-AUTH-HARDENING-001`
- Severity: Low
- Location: `backend/security.py:53`
- Evidence:
```python
return extract_api_key(request) == required
```
- Impact: Direct string comparison is not constant-time; this is usually low-risk over network but not ideal for secret comparison primitives.
- Fix: Use `hmac.compare_digest(extract_api_key(request) or "", required)`.
- Mitigation: Keep long random API keys and TLS everywhere.
- False positive notes: Practical exploitability is typically low in HTTP API contexts.

## Positive practices observed

- Path safety checks are present for user-provided file paths in `backend/routes/audio.py:24`.
- CORS wildcard with credentials is avoided and wildcard input is explicitly rejected in `backend/app.py:29`.
- Subprocess invocations use argument lists rather than shell interpolation (`shell=True` not found in reviewed paths).

## Recommended secure-by-default improvement plan

1. Remove implicit loopback auth bypass and require explicit auth by default.
2. Add strict upload and request body size limits (app + proxy), plus per-endpoint abuse controls.
3. Disable or protect FastAPI docs/OpenAPI in production.
4. Add `TrustedHostMiddleware` and production security headers (backend and frontend edge/app).
5. Reduce diagnostic info returned by `/v1/info`.
6. Add a deployment hardening checklist to README (`prod mode`, headers, host allowlist, API key required, body limits).
