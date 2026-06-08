"""Top-level CLI entry point for the vocalie-backend command.

Sub-commands:
    start   Spawn uvicorn in the background, save PID
    stop    SIGTERM the running backend
    status  Print the current state (JSON when --json)
    health  HTTP GET /v1/health
    install Create venv + pip install requirements
    doctor  Environment sanity check
    logs    Tail the backend log (optionally -f)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from vocalie_backend import __version__
from vocalie_backend.config import API_HOST, API_PORT
from vocalie_backend import process, health, logs, install, doctor


# Exit codes (kept small + meaningful for the Swift app to switch on)
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_RUNNING = 2
EXIT_PORT_IN_USE = 3
EXIT_MISSING_DEP = 4


def cmd_start(args: argparse.Namespace) -> int:
    try:
        state = process.start(host=args.host, port=args.port, reload=args.reload)
    except RuntimeError as e:
        msg = str(e)
        if "already in use" in msg:
            print(f"error: {msg}", file=sys.stderr)
            return EXIT_PORT_IN_USE
        if "venv python not found" in msg:
            print(f"error: {msg}", file=sys.stderr)
            return EXIT_MISSING_DEP
        print(f"error: {msg}", file=sys.stderr)
        return EXIT_ERROR
    if args.wait:
        if not process.wait_ready(timeout_s=args.wait_timeout):
            print("warning: backend started but /v1/health not yet ready", file=sys.stderr)
    if args.json:
        print(state.to_json())
    else:
        print(f"backend started (pid={state.pid}, http://{state.host}:{state.port})")
        print(f"log: {state.log_file}")
    return EXIT_OK


def cmd_stop(args: argparse.Namespace) -> int:
    state = process.stop(timeout_s=args.timeout, force=args.force)
    if state.running:
        print(f"warning: backend still running (pid={state.pid})", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(state.to_json())
    else:
        print("backend stopped")
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    state = process.get_state()
    if args.json:
        print(state.to_json())
    else:
        if state.running:
            print(f"running: pid={state.pid} http://{state.host}:{state.port}")
        else:
            print("stopped")
    return EXIT_OK if state.running else EXIT_NOT_RUNNING


def cmd_health(args: argparse.Namespace) -> int:
    result = health.check(timeout_s=args.timeout)
    if args.json:
        print(result.to_json())
    else:
        if result.ok:
            print(f"ok ({result.http_status}, {result.latency_ms}ms)")
        else:
            print(f"fail: {result.error or result.http_status}")
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_install(args: argparse.Namespace) -> int:
    return install.install(upgrade=args.upgrade)


def cmd_doctor(args: argparse.Namespace) -> int:
    r = doctor.run()
    print(r.to_human())
    return r.exit_code()


def cmd_logs(args: argparse.Namespace) -> int:
    return logs.tail(follow=args.follow, lines=args.lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocalie-backend",
        description="Local control CLI for the Vocalie-TTS API",
    )
    p.add_argument("--version", action="version", version=f"vocalie-backend {__version__}")

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("start", help="start the backend in the background")
    sp.add_argument("--host", default=API_HOST)
    sp.add_argument("--port", type=int, default=API_PORT)
    sp.add_argument("--reload", action="store_true", help="enable autoreload (dev)")
    sp.add_argument("--wait", action="store_true", help="block until /v1/health is 200")
    sp.add_argument("--wait-timeout", type=float, default=30.0)
    sp.add_argument("--json", action="store_true", help="emit JSON output")
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("stop", help="stop a running backend")
    sp.add_argument("--timeout", type=float, default=5.0)
    sp.add_argument("--force", action="store_true", help="SIGKILL if SIGTERM times out")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("status", help="report backend state")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("health", help="ping /v1/health")
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_health)

    sp = sub.add_parser("install", help="create venv and pip install requirements")
    sp.add_argument("--upgrade", action="store_true")
    sp.set_defaults(func=cmd_install)

    sp = sub.add_parser("doctor", help="check the local environment")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("logs", help="tail the backend log")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("-n", "--lines", type=int, default=100)
    sp.set_defaults(func=cmd_logs)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
