#!/usr/bin/env bash
set -euo pipefail

# Validate production security baseline from environment variables.
# Designed for CI/CD guardrails and deployment preflight checks.

mode="prod"
if [[ "${1:-}" == "--dev" ]]; then
  mode="dev"
fi
if [[ "${1:-}" == "--prod" ]]; then
  mode="prod"
fi

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

is_truthy() {
  local v
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

is_falsy_or_empty() {
  local v
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ -z "$v" || "$v" == "0" || "$v" == "false" || "$v" == "no" || "$v" == "off" ]]
}

if [[ "$mode" == "dev" ]]; then
  echo "Security baseline check skipped: --dev mode"
  exit 0
fi

api_key="${VOCALIE_API_KEY:-}"
trust_localhost="${VOCALIE_TRUST_LOCALHOST:-0}"
enable_docs="${VOCALIE_ENABLE_API_DOCS:-0}"
allowed_hosts="${VOCALIE_ALLOWED_HOSTS:-}"
trusted_proxies="${VOCALIE_TRUSTED_PROXIES:-}"
max_upload_bytes="${VOCALIE_MAX_UPLOAD_BYTES:-}"

if [[ -z "$api_key" ]]; then
  fail "VOCALIE_API_KEY is required in production."
fi
if [[ "$api_key" == "change-me" ]]; then
  fail "VOCALIE_API_KEY must not use placeholder value."
fi
if [[ "${#api_key}" -lt 24 ]]; then
  fail "VOCALIE_API_KEY should be at least 24 characters."
fi

if ! is_falsy_or_empty "$trust_localhost"; then
  fail "VOCALIE_TRUST_LOCALHOST must be disabled in production."
fi

if is_truthy "$enable_docs"; then
  fail "VOCALIE_ENABLE_API_DOCS must be disabled in production."
fi

if [[ -z "${allowed_hosts// }" ]]; then
  fail "VOCALIE_ALLOWED_HOSTS must not be empty in production."
fi
if [[ "$allowed_hosts" == *"*"* ]]; then
  fail "VOCALIE_ALLOWED_HOSTS must not contain wildcard '*'."
fi

if [[ "$trusted_proxies" == *"*"* ]]; then
  fail "VOCALIE_TRUSTED_PROXIES must not contain wildcard '*'."
fi

if [[ -z "$max_upload_bytes" ]]; then
  fail "VOCALIE_MAX_UPLOAD_BYTES must be set in production."
fi
if ! [[ "$max_upload_bytes" =~ ^[0-9]+$ ]]; then
  fail "VOCALIE_MAX_UPLOAD_BYTES must be a positive integer."
fi
if [[ "$max_upload_bytes" -le 0 ]]; then
  fail "VOCALIE_MAX_UPLOAD_BYTES must be > 0."
fi

echo "OK: production security baseline validated."
