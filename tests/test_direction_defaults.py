import app
from state_manager import migrate_state


def test_state_defaults_direction_enabled_true():
    data = migrate_state({})
    assert data.get("direction_enabled") is True
    assert data.get("direction_source") == "final"


def test_ui_direction_enabled_default_visible():
    ui = app.build_ui()
    components = ui.config.get("components", [])
    matches = [
        component
        for component in components
        if component.get("props", {}).get("label") == "Découpage manuel (recommandé)"
    ]
    assert matches, "Direction toggle not found in UI config."
    props = matches[0].get("props", {})
    assert props.get("value") is True
    assert props.get("visible", True) is True
