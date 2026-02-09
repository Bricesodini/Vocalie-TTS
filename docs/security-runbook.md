# Security Runbook

## Scope

This runbook defines the mandatory security checks for production-like deployments.

## Required environment baseline (prod)

- `VOCALIE_API_KEY`: required, non-placeholder, strong secret.
- `VOCALIE_TRUST_LOCALHOST=0`
- `VOCALIE_ENABLE_API_DOCS=0`
- `VOCALIE_ALLOWED_HOSTS`: explicit host list, no wildcard.
- `VOCALIE_TRUSTED_PROXIES`: explicit proxy peers only, no wildcard.
- `VOCALIE_MAX_UPLOAD_BYTES`: positive integer, aligned with ingress body limit.

## Pre-deploy checklist

1. Validate baseline values.
```bash
VOCALIE_API_KEY='<long-random-secret>' \
VOCALIE_TRUST_LOCALHOST=0 \
VOCALIE_ENABLE_API_DOCS=0 \
VOCALIE_ALLOWED_HOSTS='api.example.com' \
VOCALIE_TRUSTED_PROXIES='10.0.0.10' \
VOCALIE_MAX_UPLOAD_BYTES=26214400 \
bash ./scripts/check-security-baseline.sh --prod
```

2. Validate rate-limit fairness assumptions.
```bash
python ./scripts/check-rate-limit-fairness.py
```

3. Verify docs endpoints are not exposed in prod.
```bash
curl -i https://api.example.com/v1/docs
curl -i https://api.example.com/v1/redoc
curl -i https://api.example.com/v1/openapi.json
```
Expected: no `200 OK`.

4. Verify protected endpoints reject missing API key.
```bash
curl -i https://api.example.com/v1/capabilities
```
Expected: `403`.

5. Verify host-header enforcement.
```bash
curl -i https://api.example.com/v1/health -H 'Host: evil.example.com'
```
Expected: `400`.

## Staging load/fairness validation

1. Use at least 3 distinct client identities (different API keys; optionally different source IPs).
2. Send concurrent burst traffic to heavy endpoints (`/v1/tts/jobs`, `/v1/audio/enhance`).
3. Confirm each identity gets comparable acceptance and independent throttling.
4. Confirm one abusive identity does not throttle other identities sharing the same proxy peer.

## Incident triage quick checks

1. `403` spike:
- Check `VOCALIE_API_KEY` rotation / mismatch.
- Check clients still send `X-API-Key` or `Authorization: Bearer`.

2. Unexpected `429`:
- Check `VOCALIE_RATE_LIMIT_RPS` / `VOCALIE_RATE_LIMIT_BURST`.
- Check `VOCALIE_TRUSTED_PROXIES` is accurate for current ingress path.

3. Unexpected docs exposure:
- Check `VOCALIE_ENABLE_API_DOCS`.
- Check deployment config drift between environments.
