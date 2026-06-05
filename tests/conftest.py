# Legacy tests that depend on removed modules (app.py, tts_engine.py, state_manager.py)
# These are kept for reference but skipped from collection because their
# dependencies have been removed from the project.
collect_ignore = [
    # Gradio UI tests — app.py removed
    "test_build_ui.py",
    "test_stop_button.py",
    "test_capability_driven.py",
    "test_chunk_gap.py",
    "test_direction_chunking.py",
    "test_direction_defaults.py",
    "test_language_policy.py",
    "test_minimal_edit_workflow.py",
    "test_pipeline_options.py",
    "test_piper_assets.py",
    "test_piper_command_build.py",
    "test_piper_speed.py",
    "test_session_texts.py",
    "test_text_adjustment.py",
    "test_voice_modes.py",
    "test_work_cleanup.py",
    "test_xtts_backend.py",
    # tts_engine tests — tts_engine.py removed
    "test_multilang_backend_import.py",
    "test_multilang_cuda_patch.py",
    "test_multilang_fallback.py",
    "test_short_segments.py",
    "test_tts_model_selection.py",
    # state_manager tests — state_manager.py removed
    "test_state_manager.py",
    "test_output_naming.py",
]