from tts_backends import get_backend, list_backends


def test_backend_registry_has_chatterbox():
    backend = get_backend("chatterbox")
    assert backend is not None
    assert backend.id == "chatterbox"


def test_backend_availability_flags():
    backends = {backend.id: backend for backend in list_backends()}
    assert backends["chatterbox"].is_available() in (True, False)
    assert backends["xtts"].is_available() in (True, False)
    assert backends["piper"].is_available() in (True, False)
    assert backends["bark"].is_available() in (True, False)


def test_backend_language_mapping():
    backend = get_backend("chatterbox")
    assert backend is not None
    assert backend.map_language("fr-FR") == "fr"
    assert backend.map_language("en-US") == "en"


def test_backend_validate_config_returns_list():
    backend = get_backend("chatterbox")
    assert backend is not None
    warnings = backend.validate_config({"voice_ref": None})
    assert isinstance(warnings, list)
