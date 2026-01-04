#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/.venvs/chatterbox/lib/python3.11/site-packages/chatterbox/checkpoints"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/install-chatterbox-weights.sh <archive-path-or-url>"
  echo "Example: scripts/install-chatterbox-weights.sh https://huggingface.co/.../resolve/main/model.tar.gz"
  exit 1
fi

SOURCE="$1"
TMP=""
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

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
