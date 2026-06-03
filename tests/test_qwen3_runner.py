import numpy as np

from tts_backends import qwen3_runner as qr


def test_coerce_bool_basic():
    from tts_backends.qwen3_runner import coerce_bool
    assert coerce_bool("1", False) is True
    assert coerce_bool("0", True) is False
    assert coerce_bool(None, True) is True
    assert coerce_bool(None, False) is False
    assert coerce_bool(True, False) is True
    assert coerce_bool(False, True) is False
    assert coerce_bool(1, False) is True
    assert coerce_bool(0, True) is False
    assert coerce_bool("yes", False) is True
    assert coerce_bool("no", True) is False
    assert coerce_bool("maybe", True) is True  # unknown → default


def test_pick_dtype_none():
    from tts_backends.qwen3_runner import _pick_dtype
    assert _pick_dtype(None, None) is None
    assert _pick_dtype(None, "fp16") is None


def test_write_error():
    import json
    from tts_backends.qwen3_runner import _write_error
    # _write_error writes to stdout; just ensure it doesn't crash
    import io
    import sys
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        _write_error("test_error", detail="test_detail")
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    data = json.loads(output)
    assert data["ok"] is False
    assert data["error"] == "test_error"
    assert data["detail"] == "test_detail"