def test_build_ui_no_crash():
    import app

    ui = app.build_ui()
    assert ui is not None
