"""vocalie_backend — local control CLI for the Vocalie-TTS API.

This package is the single entry point the Swift app uses to start,
stop, and query the backend. The previous shell-based flow
(scripts/dev-backend.sh + scripts/stop.sh + scripts/doctor.sh) still
works, but anything that wants to be machine-controllable should go
through the CLI:

    vocalie-backend start
    vocalie-backend stop
    vocalie-backend status        # JSON, parseable from Swift
    vocalie-backend health        # pings /v1/health
    vocalie-backend install       # creates venv + pip installs
    vocalie-backend doctor        # environment sanity check
    vocalie-backend logs [-f]     # tail the backend log

Status output is JSON; everything else prints human-readable text.
Exit codes:
    0  success
    1  generic error
    2  service not running (when querying)
    3  port already in use
    4  dependency missing
"""

__version__ = "0.1.0"
