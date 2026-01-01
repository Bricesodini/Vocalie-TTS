import app


def test_update_estimated_duration():
    text = "Bonjour tout le monde."
    assert "Durée estimée" in app.update_estimated_duration(text)
