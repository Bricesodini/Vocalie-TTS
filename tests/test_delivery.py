import pytest
pytest.skip("Livraison retirée en V2.", allow_module_level=True)

import datetime as dt
import json
from pathlib import Path

from session_manager import (
    build_session_payload,
    create_session_dir,
    deliver_take_to_output,
    get_take_path_global_raw,
    write_session_json,
)


def _make_session(tmp_path: Path) -> Path:
    work_root = tmp_path / "work"
    session_dir = create_session_dir(work_root, dt.datetime(2024, 1, 2, 3, 4, 5), "demo")
    take_path = get_take_path_global_raw(session_dir, "v1")
    take_path.parent.mkdir(parents=True, exist_ok=True)
    take_path.write_bytes(b"dummy")
    payload = build_session_payload(
        engine_id="chatterbox",
        engine_slug="chatterbox",
        ref_name="ref.wav",
        text="Bonjour tout le monde.",
        editorial_text="Bonjour tout le monde.",
        tts_ready_text="Bonjour tout le monde.",
        prep_log_md="",
        created_at=dt.datetime(2024, 1, 2, 3, 4, 5),
        chunks=[],
        artifacts={
            "raw_global": "takes/global/global_v1_raw.wav",
            "processed_global": None,
        },
        artifacts_list=[take_path],
        takes={"global": ["v1"]},
        active_take={"global": "v1"},
    )
    write_session_json(session_dir, payload)
    return session_dir


def test_deliver_take_creates_file_and_meta(tmp_path):
    session_dir = _make_session(tmp_path)
    out_dir = tmp_path / "out"
    exported_path, meta_path = deliver_take_to_output(
        session_dir=session_dir,
        output_dir=out_dir,
        user_filename="delivery",
        add_timestamp=False,
        include_engine_slug=False,
    )
    assert exported_path.exists()
    assert exported_path.name == "delivery.wav"
    assert meta_path.exists()
    assert meta_path.parent == session_dir / "meta"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["active_take"] == "v1"
    assert meta["dest_path"] == str(exported_path)
    data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert data["deliveries"][0]["dest_path"] == str(exported_path)


def test_deliver_take_collision_suffix(tmp_path):
    session_dir = _make_session(tmp_path)
    out_dir = tmp_path / "out"
    first_path, _ = deliver_take_to_output(
        session_dir=session_dir,
        output_dir=out_dir,
        user_filename="delivery",
        add_timestamp=False,
        include_engine_slug=False,
    )
    second_path, _ = deliver_take_to_output(
        session_dir=session_dir,
        output_dir=out_dir,
        user_filename="delivery",
        add_timestamp=False,
        include_engine_slug=False,
    )
    assert first_path.exists()
    assert second_path.exists()
    assert second_path.name == "delivery_01.wav"


def test_deliver_missing_take(tmp_path):
    session_dir = _make_session(tmp_path)
    (session_dir / "takes" / "global" / "global_v1_raw.wav").unlink()
    with pytest.raises(FileNotFoundError):
        deliver_take_to_output(
            session_dir=session_dir,
            output_dir=tmp_path / "out",
            user_filename="delivery",
            add_timestamp=False,
            include_engine_slug=False,
        )


def test_cleanup_on_deliver(tmp_path):
    session_dir = _make_session(tmp_path)
    out_dir = tmp_path / "out"
    exported_path, _ = deliver_take_to_output(
        session_dir=session_dir,
        output_dir=out_dir,
        user_filename="delivery",
        add_timestamp=False,
        include_engine_slug=False,
        cleanup_on_deliver=True,
    )
    assert exported_path.exists()
    assert not session_dir.exists()
