import multiprocessing as mp
import time

import app


def test_handle_stop_terminates_process_and_cleans_tmp(tmp_path):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=time.sleep, args=(5,))
    proc.start()
    tmp_file = tmp_path / "audio.wav.tmp"
    tmp_file.write_bytes(b"partial")

    app._set_job_state(
        current_proc=proc,
        current_tmp_path=str(tmp_file),
        current_final_path="unused",
        job_running=True,
    )

    _, _, log_text = app.handle_stop("log")

    proc.join(timeout=1)
    assert not proc.is_alive()
    assert not tmp_file.exists()
    assert "Annulé" in log_text
    assert not app._get_job_state()["job_running"]
