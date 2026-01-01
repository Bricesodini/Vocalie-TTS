from session_manager import (
    build_session_payload,
    create_session_dir,
    write_session_json,
)

import app


def test_handle_session_texts_reads_session(tmp_path):
    session_dir = create_session_dir(tmp_path, app.dt.datetime(2024, 1, 2, 3, 4, 5), "demo")
    payload = build_session_payload(
        engine_id="chatterbox",
        engine_slug="chatterbox",
        ref_name=None,
        text="Bonjour",
        editorial_text="Bonjour",
        tts_ready_text="Bonjour nettoye",
        prep_log_md="log",
        created_at=app.dt.datetime(2024, 1, 2, 3, 4, 5),
        chunks=[],
        artifacts={
            "raw_global": "takes/global/global_v1_raw.wav",
            "processed_global": None,
        },
        artifacts_list=[],
        takes={"global": ["v1"]},
        active_take={"global": "v1"},
    )
    write_session_json(session_dir, payload)
    editorial, tts_ready, prep_log, visibility = app.handle_session_texts(
        {"dir": str(session_dir)}
    )
    assert editorial == "Bonjour"
    assert tts_ready == "Bonjour nettoye"
    assert prep_log == "log"
