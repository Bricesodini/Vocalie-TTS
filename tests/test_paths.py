from output_paths import make_output_filename, prepare_output_paths, sanitize_filename


def test_make_output_filename_with_user_name():
    ts = "2024-01-01_00-00-00"
    filename = make_output_filename(
        text="Texte promo",
        ref_name="voix.wav",
        user_filename="Ma Voix Finale!",
        add_timestamp=True,
        timestamp=ts,
    )
    assert filename.startswith("Ma-Voix-Finale")
    assert filename.endswith(f"__{ts}.wav")


def test_make_output_filename_fallback_slug():
    ts = "2024-01-01_00-00-00"
    filename = make_output_filename(
        text="Texte Très Spécial",
        ref_name="Référence",
        user_filename="",
        add_timestamp=False,
        timestamp=ts,
    )
    assert filename.startswith("texte-tres-special__reference")
    assert filename.endswith(".wav")


def test_prepare_output_paths_handles_collisions(tmp_path):
    preview_dir = tmp_path / "preview"
    user_dir = tmp_path / "user"
    preview_dir.mkdir()
    user_dir.mkdir()
    filename = "demo.wav"

    preview_path, user_path = prepare_output_paths(preview_dir, user_dir, filename)
    assert preview_path.parent == preview_dir
    assert user_path.parent == user_dir
    assert preview_path.name == user_path.name

    # Simulate existing file in user directory to force unique suffix
    (user_dir / preview_path.name).write_text("old")
    _, user_path_2 = prepare_output_paths(preview_dir, user_dir, filename)
    assert user_path_2.name != preview_path.name


def test_sanitize_filename_removes_forbidden():
    cleaned = sanitize_filename('Nom* Invalide:? "test"')
    assert cleaned == "Nom-Invalide-test"
