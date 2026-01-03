# Security Policy

## Supported deployments

- This project is **local-first**. Do **not** expose the API directly to the public Internet.
- LAN usage is supported only with explicit configuration (API key + CORS allowlist).

## Reporting a vulnerability

Please open a GitHub issue with the **security** label (or contact the maintainer privately if you prefer).

## In scope

- Authentication / authorization bypass
- Path traversal / arbitrary file read or write
- Remote code execution (RCE)
- SSRF / command injection
- Denial of service (unbounded jobs, unbounded payload sizes)

## Out of scope

- Issues requiring public Internet exposure against recommendations
