#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/.venvs/chatterbox/lib/python3.11/site-packages/chatterbox/checkpoints"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/install-chatterbox-weights.sh <repo-id|weights-dir|archive-path-or-url>"
  echo "Example (repo): scripts/install-chatterbox-weights.sh ResembleAI/chatterbox"
  echo "Example (repo): scripts/install-chatterbox-weights.sh Thomcles/Chatterbox-TTS-French"
  echo "Example (dir): scripts/install-chatterbox-weights.sh /tmp/chatterbox-vanilla"
  echo "Example (url): scripts/install-chatterbox-weights.sh https://huggingface.co/.../resolve/main/model.tar.gz"
  exit 1
fi

SOURCE="$1"
TMP=""
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
PYTHON="${ROOT_DIR}/.venvs/chatterbox/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

download() {
  TMP="$(mktemp)"
  trap 'rm -f "$TMP"' EXIT
  echo "Downloading $SOURCE ..."
  if [[ -n "$TOKEN" ]]; then
    curl -L -H "Authorization: Bearer $TOKEN" -o "$TMP" "$SOURCE"
  else
    curl -L -o "$TMP" "$SOURCE"
  fi
  SOURCE="$TMP"
}

if [[ ! -e "$SOURCE" && "$SOURCE" != http* && "$SOURCE" == */* ]]; then
  echo "Prefetching Hugging Face repo: $SOURCE"
  "$PYTHON" - <<'PY' "$SOURCE"
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    raise SystemExit(f"Missing huggingface_hub in {sys.executable}: {exc}")

repo_id = sys.argv[1]
token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
snapshot_download(repo_id=repo_id, token=token)
print(f"Cached Hugging Face repo: {repo_id}")
PY
  exit 0
fi

if [[ -d "$SOURCE" ]]; then
  echo "Copying weights from local directory."
  echo "Note: Chatterbox uses the Hugging Face cache; prefer repo ids for prefetch."
  rm -rf "$WEIGHTS_DIR"
  mkdir -p "$WEIGHTS_DIR"
  cp -R "$SOURCE"/. "$WEIGHTS_DIR"/
  echo "Chatterbox weights installed under $WEIGHTS_DIR"
  exit 0
fi

if [[ "$SOURCE" =~ ^https?:// ]]; then
  download
fi

if [[ ! -f "$SOURCE" ]]; then
  echo "File not found: $SOURCE" >&2
  exit 1
fi

MIME=$(file --mime-type -b "$SOURCE")
if [[ ! "$MIME" =~ ^(application/zip|application/x-gzip|application/x-tar)$ ]]; then
  echo "Downloaded file does not look like a ZIP/TAR (mime=$MIME)."
  echo "If Hugging Face requires authentication, export HUGGINGFACE_TOKEN or HF_TOKEN and retry."
  exit 1
fi

rm -rf "$WEIGHTS_DIR"
mkdir -p "$WEIGHTS_DIR"

if file "$SOURCE" | grep -qi 'zip archive'; then
  unzip -q "$SOURCE" -d "$WEIGHTS_DIR"
elif file "$SOURCE" | grep -qi 'tar archive'; then
  tar -xf "$SOURCE" -C "$WEIGHTS_DIR"
else
  cp -R "$SOURCE" "$WEIGHTS_DIR/"
fi

echo "Chatterbox weights installed under $WEIGHTS_DIR"
