"""Base subprocess runner for TTS backends (JSON stdin/stdout protocol).

Provides the common communication protocol that all subprocess-based
TTS backends share.  Each concrete runner inherits from ``BaseSubprocessRunner``
and only implements ``run_synthesis()`` — all stdin/stdout/JSON/timeout/error
handling is centralized here.

Protocol
-------
1. Parent process serialises a dict payload as JSON on the runner's stdin.
2. Runner reads JSON from stdin, performs synthesis, writes a WAV file.
3. Runner writes a JSON response to stdout:  ``{"ok": true, ...}`` on success
   or ``{"ok": false, "error": "...", "detail?": "..."}`` on failure.
4. Non-zero exit codes also signal failure.

Usage (inside a venv runner script)
------------------------------------
::

    from tts_backends.base_runner import BaseSubprocessRunner

    class MyRunner(BaseSubprocessRunner):
        def run_synthesis(self, payload: dict) -> dict:
            # Load model, synthesise, write WAV, return metadata
            ...

    if __name__ == "__main__":
        raise SystemExit(MyRunner.main())

Usage (inside a backend wrapper)
---------------------------------
::

    from tts_backends.base_runner import SubprocessBackendMixin

    class MyBackend(TTSBackend, SubprocessBackendMixin):
        runner_module = "my_runner"          # filename in tts_backends/
        runner_venv = "my_backend"           # venv dir name
        default_timeout = 300.0

        def synthesize_chunk(self, text, **params):
            return self._run_subprocess_chunk(text, **params)
"""

from __future__ import annotations

import abc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BackendUnavailableError, coerce_bool


# ── Runner side (executed inside venv) ────────────────────────────────────


class BaseSubprocessRunner(abc.ABC):
    """Base class for subprocess runners (runs inside the backend venv).

    Subclasses implement ``run_synthesis()`` which receives the parsed
    JSON payload and must return a dict to be serialised as the response.
    """

    @abc.abstractmethod
    def run_synthesis(self, payload: dict) -> dict:
        """Execute synthesis from *payload* and return a response dict.

        The response **must** include ``"ok": True`` on success.
        Any other keys in the dict are passed back to the caller.
        On failure, raise an exception — ``main()`` catches it and
        writes a standard ``{"ok": false, …}`` response.
        """
        raise NotImplementedError

    # ── Helpers available to subclasses ────────────────────────────

    @staticmethod
    def write_response(payload: dict) -> None:
        """Write a JSON response to stdout and flush."""
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.stdout.flush()

    @staticmethod
    def write_error(message: str, *, detail: str | None = None, trace: str | None = None) -> None:
        """Write a standard error response to stdout and flush."""
        payload: dict[str, Any] = {"ok": False, "error": message}
        if detail:
            payload["detail"] = detail
        if trace:
            payload["trace"] = trace
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.stdout.flush()

    @staticmethod
    def read_payload() -> dict:
        """Read and parse JSON payload from stdin."""
        raw = sys.stdin.read()
        if not raw:
            raise ValueError("missing JSON payload on stdin")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("payload must be a JSON object")
        return data

    @staticmethod
    def log(message: str, *, log_path: str | None = None) -> None:
        """Write a timestamped log line to stderr (and optionally a file)."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}\n"
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception:
                pass

    @staticmethod
    def setup_hf_cache(assets_dir: Path | str | None) -> None:
        """Configure HuggingFace cache directories from *assets_dir*."""
        if not assets_dir:
            return
        assets = Path(assets_dir).expanduser()
        assets.mkdir(parents=True, exist_ok=True)
        hf_home = assets / ".hf"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TORCH_HOME", str(assets / ".torch"))

    @staticmethod
    def resolve_dtype(torch_mod: Any, dtype_value: str | None):
        """Resolve a dtype string to a torch dtype object."""
        if not dtype_value:
            return None
        mapping = {
            "float16": "float16", "fp16": "float16",
            "bfloat16": "bfloat16", "bf16": "bfloat16",
            "float32": "float32", "fp32": "float32",
        }
        name = mapping.get(str(dtype_value).lower())
        if not name:
            return None
        return getattr(torch_mod, name, None)

    # ── Entry point ─────────────────────────────────────────────────

    @classmethod
    def main(cls) -> int:
        """Standard entry point for a subprocess runner.

        Reads JSON from stdin, delegates to ``run_synthesis()``,
        writes JSON response to stdout.  Returns a process exit code.
        """
        runner = cls()
        try:
            payload = runner.read_payload()
            result = runner.run_synthesis(payload)
            if not isinstance(result, dict):
                result = {"ok": True, "data": result}
            result.setdefault("ok", True)
            runner.write_response(result)
            return 0
        except Exception as exc:
            import traceback as _tb
            runner.write_error(
                str(exc),
                trace=_tb.format_exc(),
            )
            return 1


# ── Parent-side (executed in the main API process) ─────────────────────────


class SubprocessBackendMixin:
    """Mixin for TTSBackend subclasses that delegate synthesis to a subprocess.

    Usage::

        class MyBackend(TTSBackend, SubprocessBackendMixin):
            runner_module = "my_runner"       # tts_backends/my_runner.py
            runner_venv  = "my_backend"        # .venvs/my_backend/
            default_timeout = 300.0

    Provides ``_run_subprocess()`` and ``_run_subprocess_chunk()`` helpers.
    """

    runner_module: str = ""     # filename in tts_backends/ (without .py)
    runner_venv: str = ""       # venv directory name under .venvs/
    default_timeout: float = 300.0

    def _runner_path(self) -> Path:
        return Path(__file__).resolve().parent / f"{self.runner_module}.py"

    def _python_path(self) -> Path:
        from backend_install.paths import python_path
        return python_path(self.runner_venv)

    def _run_subprocess(self, payload: dict, *, timeout_s: float | None = None) -> dict:
        """Run the subprocess runner with *payload* and return the parsed response.

        Raises ``BackendUnavailableError`` on any failure.
        """
        from backend_install.paths import python_path

        py = self._python_path()
        if not py.exists():
            raise BackendUnavailableError(f"{self.runner_venv} venv introuvable.")

        runner = self._runner_path()
        if not runner.exists():
            raise BackendUnavailableError(f"Runner {self.runner_module} introuvable.")

        timeout = timeout_s or self.default_timeout

        try:
            result = subprocess.run(
                [str(py), str(runner)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailableError(f"{self.runner_venv} runner timeout.") from exc

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            raise BackendUnavailableError(stderr or stdout or f"{self.runner_venv} runner failed.")

        if not stdout:
            raise BackendUnavailableError(f"{self.runner_venv} runner returned no output.")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Try last line (some runners may emit log lines before JSON)
            lines = [line for line in stdout.splitlines() if line.strip()]
            if not lines:
                raise BackendUnavailableError(
                    f"{self.runner_venv} runner returned invalid JSON."
                ) from None
            try:
                data = json.loads(lines[-1])
            except json.JSONDecodeError as exc:
                raise BackendUnavailableError(
                    f"{self.runner_venv} runner returned invalid JSON."
                ) from exc

        if not data.get("ok"):
            error = data.get("error") or f"{self.runner_venv} runner failed."
            detail = data.get("detail")
            trace = data.get("trace")
            if detail and detail != error:
                error = f"{error}\n{detail}"
            if trace:
                error = f"{error}\n{trace}"
            raise BackendUnavailableError(error)

        if stderr:
            data["stderr"] = stderr
        return data

    def _run_subprocess_chunk(
        self,
        text: str,
        *,
        voice_ref_path: str | None = None,
        lang: str | None = None,
        payload_suffix: dict | None = None,
        timeout_s: float | None = None,
    ) -> tuple:
        """Run the subprocess for a single chunk and return ``(audio, sr, meta)``.

        This is a convenience method that:
        1. Writes to a temp WAV inside ``.assets/{venv}/.tmp/``.
        2. Calls ``_run_subprocess()``.
        3. Reads back the WAV file.
        4. Cleans up the temp file.
        """
        import uuid
        import numpy as np
        import soundfile as sf
        from backend_install.paths import ROOT

        assets_dir = ROOT / ".assets" / self.runner_venv
        tmp_dir = assets_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"{self.runner_venv}_{uuid.uuid4().hex}.wav"

        payload: dict[str, Any] = {
            "text": text,
            "out_path": str(tmp_wav),
        }
        if voice_ref_path:
            payload["voice_ref_path"] = voice_ref_path
        if lang:
            payload["lang"] = lang
        if payload_suffix:
            payload.update(payload_suffix)

        result = self._run_subprocess(payload, timeout_s=timeout_s)

        if not tmp_wav.exists():
            raise BackendUnavailableError(
                f"{self.runner_venv} runner n'a pas produit de WAV."
            )

        audio, sr = sf.read(str(tmp_wav), dtype="float32")
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass

        meta = dict(result)
        meta.pop("ok", None)
        meta.pop("out_path", None)
        return np.asarray(audio, dtype=np.float32), int(sr), meta