"""Compatibility shim — canonical location is backend.shared.tts_pipeline."""
from backend.shared.tts_pipeline import (  # noqa: F401
    PipelineResult,
    minimal_post_process,
    generate_raw_wav,
    run_tts_pipeline,
    _find_active_range,
    _fade_in,
    _fade_out,
    _snap_zero_crossing,
)
# Re-export submodule dependencies for test monkeypatching compatibility
from tts_backends import get_backend  # noqa: F401