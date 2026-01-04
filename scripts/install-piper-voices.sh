#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/.assets/piper/voices"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: scripts/install-piper-voices.sh <archive-or-url>

Args:
- <archive-or-url> : path to a local zip/tar.gz + voices folder, or a URL to download.

Example:
  scripts/install-piper-voices.sh ~/Downloads/piper-voices-fr.zip
  scripts/install-piper-voices.sh https://huggingface.co/example/piper-voices-fr/resolve/main/piper-voices-fr.zip
EOF
  exit 1
fi

SOURCE="$1"
TMP_FILE=""

if [[ "$SOURCE" =~ ^https?:// ]]; then
  TMP_FILE="$(mktemp)"
  trap 'rm -f "$TMP_FILE"' EXIT
  echo "Downloading Piper voice pack from $SOURCE ..."
  curl -L -o "$TMP_FILE" "$SOURCE"
  SOURCE="$TMP_FILE"
fi

if [[ ! -f "$SOURCE" ]]; then
  echo "File not found: $SOURCE" >&2
  exit 1
fi

rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

if file "$SOURCE" | grep -qE 'Zip archive'; then
  unzip -q "$SOURCE" -d "$DEST_DIR"
elif file "$SOURCE" | grep -q 'tar archive'; then
  tar -xf "$SOURCE" -C "$DEST_DIR"
else
  cp -R "$SOURCE" "$DEST_DIR/"
fi

echo "Piper voices installed under $DEST_DIR."
echo "Piper backend should now expose the voices (restart the API if needed)."
