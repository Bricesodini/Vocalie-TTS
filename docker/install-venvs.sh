#!/usr/bin/env bash
# Vocalie-TTS — Backend venv installer
# Idempotent: skips venvs that already exist and pass an import probe.
# Runs on first container start (or whenever the vocalie-venvs volume is empty).
set -uo pipefail

cd /app

VENV_DIR="/app/.venvs"
mkdir -p "${VENV_DIR}"

# Default list of backends to install. Override with VOCALIE_INSTALL_BACKENDS
# (comma-separated) to customize (e.g. "chatterbox" or "chatterbox,qwen3").
DEFAULT_BACKENDS="chatterbox,qwen3"
BACKENDS="${VOCALIE_INSTALL_BACKENDS:-${DEFAULT_BACKENDS}}"

log() { echo "[install-venvs $(date +%H:%M:%S)] $*"; }

install_chatterbox() {
    local venv="${VENV_DIR}/chatterbox"
    local py="${venv}/bin/python"
    if [ -x "${py}" ] && "${py}" -c "import chatterbox" 2>/dev/null; then
        log "chatterbox: venv already installed and importable, skipping."
        return 0
    fi
    log "chatterbox: installing (this takes ~5-10 min, torch + chatterbox-tts + diffusers)..."
    rm -rf "${venv}"
    python -m venv "${venv}"
    "${venv}/bin/pip" install --quiet --upgrade pip setuptools wheel
    "${venv}/bin/pip" install --quiet "numpy<1.26,>=1.24"
    if ! "${venv}/bin/pip" install --quiet -r /app/requirements-chatterbox.txt; then
        log "chatterbox: pip install failed" >&2
        return 1
    fi
    if ! "${py}" -c "import chatterbox" 2>/dev/null; then
        log "chatterbox: import probe failed after install" >&2
        return 1
    fi
    log "chatterbox: venv ready."
}

install_qwen3() {
    local venv="${VENV_DIR}/qwen3"
    local py="${venv}/bin/python"
    if [ -x "${py}" ] && "${py}" -c "import qwen_tts" 2>/dev/null; then
        log "qwen3: venv already installed and importable, skipping."
        return 0
    fi
    log "qwen3: installing (this takes ~5-10 min, qwen-tts + torch)..."
    rm -rf "${venv}"
    python -m venv "${venv}"
    "${venv}/bin/pip" install --quiet --upgrade pip setuptools wheel
    if ! "${venv}/bin/pip" install --quiet "qwen-tts==0.0.4" "torch" "torchaudio"; then
        log "qwen3: pip install failed" >&2
        return 1
    fi
    if ! "${py}" -c "import qwen_tts" 2>/dev/null; then
        log "qwen3: import probe failed after install" >&2
        return 1
    fi
    log "qwen3: venv ready."
}

failed=0
IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"
for backend in "${BACKEND_LIST[@]}"; do
    backend="$(echo "${backend}" | xargs)"  # trim whitespace
    case "${backend}" in
        chatterbox)
            install_chatterbox || failed=1
            ;;
        qwen3)
            install_qwen3 || failed=1
            ;;
        "")
            # skip empty entries (e.g. trailing comma)
            ;;
        *)
            log "unknown backend: ${backend} (skipped)"
            ;;
    esac
done

if [ "${failed}" -ne 0 ]; then
    log "one or more backends failed to install. The container will still start, but TTS jobs for failed backends will return errors." >&2
    return 1
fi

log "all requested backends ready."
