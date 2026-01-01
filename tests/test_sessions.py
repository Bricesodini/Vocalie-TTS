import datetime as dt
import json

from session_manager import (
    build_session_payload,
    create_session_dir,
    extract_session_texts,
    get_take_path_global_raw,
    stage_take_copy,
    write_session_json,
)
from text_tools import chunk_script


def test_session_folder_and_json(tmp_path):
    created_at = dt.datetime(2024, 1, 2, 3, 4, 5)
    work_root = tmp_path / "work"
    session_dir = create_session_dir(work_root, created_at, "hello")
    assert session_dir == work_root / ".sessions" / "20240102_030405_hello"
    assert session_dir.exists()

    chunks = chunk_script(
        "Bonjour tout le monde.",
        min_words_per_chunk=1,
        max_words_without_terminator=10,
        max_est_seconds_per_chunk=10.0,
    )
    source_audio = tmp_path / "preview.wav"
    source_audio.write_bytes(b"dummy")
    take_path = stage_take_copy(session_dir, source_audio, "global_v1_raw.wav")
    assert take_path.exists()
    assert take_path == get_take_path_global_raw(session_dir, "v1")
    assert (session_dir / "takes" / "global").exists()
    assert (session_dir / "takes" / "chunks").exists()
    assert (session_dir / "meta").exists()
    assert (session_dir / "preview").exists()
    payload = build_session_payload(
        engine_id="chatterbox",
        engine_slug="chatterbox",
        ref_name="ref.wav",
        text="Bonjour tout le monde.",
        editorial_text="Bonjour tout le monde.",
        tts_ready_text="Bonjour tout le monde.",
        prep_log_md="",
        created_at=created_at,
        chunks=chunks,
        artifacts={
            "raw_global": "takes/global/global_v1_raw.wav",
            "processed_global": None,
        },
        artifacts_list=[source_audio, take_path],
        takes={"global": ["v1"]},
        active_take={"global": "v1"},
    )
    session_path = write_session_json(session_dir, payload)

    data = json.loads(session_path.read_text(encoding="utf-8"))
    assert data["engine_id"] == "chatterbox"
    assert data["engine_slug"] == "chatterbox"
    assert data["ref_name"] == "ref.wav"
    assert data["text"]["editorial"] == "Bonjour tout le monde."
    assert data["text"]["tts_ready"] == "Bonjour tout le monde."
    assert data["text_legacy"] == "Bonjour tout le monde."
    assert data["created_at"] == "2024-01-02T03:04:05"
    assert data["artifacts"]["raw_global"] == "takes/global/global_v1_raw.wav"
    assert data["artifacts_list"] == [str(source_audio), str(take_path)]
    assert data["chunks"][0]["text"]
    assert data["chunks"][0]["start_word"] == 1
    assert data["chunks"][0]["est_seconds"] == chunks[0].estimated_duration
    assert data["takes"]["global"] == ["v1"]
    assert data["active_take"]["global"] == "v1"


def test_extract_session_texts_legacy():
    editorial, tts_ready, prep_log = extract_session_texts({"text": "Legacy"})
    assert editorial == "Legacy"
    assert tts_ready == "Legacy"
    assert prep_log == ""
