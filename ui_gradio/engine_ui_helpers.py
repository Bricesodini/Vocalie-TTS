from __future__ import annotations

import datetime as dt

import gradio as gr

from backend_install.status import backend_status
from tts_backends import list_backends
from tts_backends.base import coerce_language
from tts_backends.catalog import ENGINE_ALIAS_MAP, canonical_engine_id


LANGUAGE_LABELS = {
    "fr-FR": "Français (fr-FR)",
    "en-US": "English US (en-US)",
    "en-GB": "English UK (en-GB)",
    "es-ES": "Español (es-ES)",
    "de-DE": "Deutsch (de-DE)",
    "it-IT": "Italiano (it-IT)",
    "pt-PT": "Português (pt-PT)",
    "nl-NL": "Nederlands (nl-NL)",
}


def backend_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for backend in list_backends():
        label = backend.display_name
        if not backend.is_available():
            label = f"{label} (indisponible)"
        choices.append((label, backend.id))
    return choices


def backend_ref_note(backend) -> str:
    if backend is None:
        return "Moteur indisponible."
    if backend.uses_internal_voices:
        return "Référence vocale désactivée (moteur à voix internes)."
    if not backend.supports_ref_audio:
        return "Référence vocale non supportée par ce moteur."
    return ""


def _param_schema_cache():
    engine_schemas = {}
    all_keys = []
    for backend in list_backends():
        schema = backend.params_schema()
        engine_schemas[backend.id] = schema
        for key in schema:
            if key not in all_keys:
                all_keys.append(key)
    return engine_schemas, all_keys


def param_spec_catalog() -> dict:
    engine_schemas, _ = _param_schema_cache()
    catalog: dict = {}
    for schema in engine_schemas.values():
        for key, spec in schema.items():
            if key not in catalog:
                catalog[key] = spec
    return catalog


def _ordered_languages(supported: list[str]) -> list[str]:
    if not supported:
        return ["fr-FR"]
    if "fr-FR" in supported:
        return ["fr-FR"] + [lang for lang in supported if lang != "fr-FR"]
    return list(supported)


def language_choices(supported: list[str]) -> list[tuple[str, str]]:
    ordered = _ordered_languages(supported)
    return [(LANGUAGE_LABELS.get(code, code), code) for code in ordered]


def coerce_param_value(spec, value):
    if spec.type == "bool":
        return bool(value) if isinstance(value, bool) else bool(spec.default)
    if spec.type == "int":
        try:
            coerced = int(float(value))
        except (TypeError, ValueError):
            coerced = int(spec.default)
        if spec.min is not None:
            coerced = max(int(spec.min), coerced)
        if spec.max is not None:
            coerced = min(int(spec.max), coerced)
        return coerced
    if spec.type == "float":
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            coerced = float(spec.default)
        if spec.min is not None:
            coerced = max(float(spec.min), coerced)
        if spec.max is not None:
            coerced = min(float(spec.max), coerced)
        return coerced
    if spec.type in {"choice", "select"}:
        choices = spec.choices or []
        values = []
        for item in choices:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                values.append(item[1])
            else:
                values.append(item)
        return value if value in values else spec.default
    if spec.type == "str":
        return str(value) if value is not None else str(spec.default or "")
    return value if value is not None else spec.default


def spec_visible(spec, context: dict) -> bool:
    if not spec.visible_if:
        return True
    for key, expected in spec.visible_if.items():
        if key == "voice_count_min":
            if context.get("voice_count", 0) < int(expected):
                return False
            continue
        if context.get(key) != expected:
            return False
    return True


def engine_param_schema(engine_id: str) -> dict:
    schemas, _ = _param_schema_cache()
    return schemas.get(engine_id, {})


def all_param_keys() -> list[str]:
    _, keys = _param_schema_cache()
    return list(keys)


def build_param_updates(engine_id: str, engine_params: dict, context: dict) -> list[dict]:
    schema = engine_param_schema(engine_id)
    updates: list[dict] = []
    for key in all_param_keys():
        spec = schema.get(key)
        if spec is None:
            updates.append(gr.update(visible=False))
            continue
        value = coerce_param_value(spec, engine_params.get(key, spec.default))
        visible = spec_visible(spec, context)
        if spec.type in {"choice", "select"}:
            updates.append(
                gr.update(
                    value=value,
                    visible=visible,
                    interactive=visible,
                    choices=spec.choices or [],
                )
            )
        else:
            updates.append(gr.update(value=value, visible=visible, interactive=visible))
    return updates


def build_param_visibility_updates(engine_id: str, context: dict) -> list[dict]:
    schema = engine_param_schema(engine_id)
    updates: list[dict] = []
    for key in all_param_keys():
        spec = schema.get(key)
        if spec is None:
            updates.append(gr.update(visible=False))
            continue
        visible = spec_visible(spec, context)
        updates.append(gr.update(visible=visible, interactive=visible))
    return updates


def collect_engine_params(engine_id: str, values: dict) -> dict:
    schema = engine_param_schema(engine_id)
    params: dict = {}
    for key, spec in schema.items():
        params[key] = coerce_param_value(spec, values.get(key, spec.default))
    return params


def create_param_widget(spec, value, visible):
    label = spec.label or spec.key
    info = spec.help or None
    if spec.type in {"float", "int"}:
        if spec.min is None or spec.max is None or spec.step is None:
            return gr.Number(label=label, value=value, visible=visible, interactive=visible)
        return gr.Slider(
            spec.min,
            spec.max,
            value=value,
            step=spec.step,
            label=label,
            info=info,
            visible=visible,
            interactive=visible,
        )
    if spec.type in {"choice", "select"}:
        return gr.Dropdown(
            label=label,
            choices=spec.choices or [],
            value=value,
            visible=visible,
            interactive=visible,
        )
    if spec.type == "bool":
        return gr.Checkbox(
            label=label,
            value=bool(value),
            visible=visible,
            interactive=visible,
        )
    return gr.Textbox(
        label=label,
        value=str(value) if value is not None else "",
        visible=visible,
        interactive=visible,
    )


def load_engine_config(container: dict, engine_id: str) -> tuple[str, str | None, dict]:
    engines = container.get("engines", {})
    engine_data = engines.get(engine_id, {}) if isinstance(engines, dict) else {}
    language = engine_data.get("language") or "fr-FR"
    voice_id = engine_data.get("voice_id")
    params = engine_data.get("params") or {}
    return language, voice_id, params


def engine_status_markdown(engine_id: str) -> str:
    status = backend_status(engine_id)
    if status.get("installed"):
        if canonical_engine_id(engine_id) == "xtts_v2" and not status.get("model_downloaded", True):
            return "Statut moteur: ⚠️ Poids XTTS non préchargés (téléchargement au premier usage)"
        return f"Statut moteur: ✅ Installé ({status.get('reason')})"
    return f"Statut moteur: ❌ Non installé ({status.get('reason')})"


def supported_languages_for(engine_id: str, backend, chatterbox_mode: str) -> list[str]:
    if backend is None:
        return ["fr-FR"]
    # Chatterbox FR fine-tune only supports fr-FR
    if getattr(backend, 'id', '') == 'chatterbox' and chatterbox_mode == "fr_finetune":
        return ["fr-FR"]
    return backend.supported_languages() or ["fr-FR"]


def language_ui_updates(
    engine_id: str,
    backend,
    chatterbox_mode: str,
    requested_language: str | None,
    voice_id: str | None = None,
):
    supported = supported_languages_for(engine_id, backend, chatterbox_mode)
    locked_reason = ""
    if backend and backend.uses_internal_voices:
        voices = backend.list_voices()
        voice_langs = None
        for voice in voices:
            if voice.id == voice_id and voice.lang_codes:
                voice_langs = voice.lang_codes
                break
        if voice_langs:
            supported = voice_langs
            locked_reason = " (verrouillée par la voix sélectionnée)"
    final_lang, did_fallback = coerce_language(
        requested_language or "fr-FR",
        supported,
        backend.default_language() if backend else None,
    )
    supports_select = (
        len(supported) > 1
        and bool(backend and backend.supports_multilang)
        and not (backend and backend.uses_internal_voices)
    )
    lang_update = gr.update(
        value=final_lang,
        visible=supports_select,
        interactive=supports_select,
        choices=language_choices(supported) if supports_select else None,
    )
    locked_text = ""
    if not supports_select:
        locked_text = f"Langue : {final_lang} (verrouillée{locked_reason})"
    locked_update = gr.update(value=locked_text, visible=not supports_select)
    return final_lang, did_fallback, lang_update, locked_update


def build_voice_label_update(engine_id: str, backend, requested_voice: str | None):
    if backend is None or not backend.uses_internal_voices:
        return None, gr.update(value="", visible=False), ""
    voices = backend.list_voices()
    voice_ids = [voice.id for voice in voices]
    fallback_note = ""
    final_voice = requested_voice if requested_voice in voice_ids else (voice_ids[0] if voice_ids else None)
    if requested_voice and final_voice != requested_voice:
        fallback_note = f"tts_voice_fallback requested={requested_voice} -> {final_voice} (engine={engine_id})"
    if len(voices) == 0:
        return final_voice, gr.update(value="⚠️ Aucune voix Piper installée", visible=True), fallback_note
    return final_voice, gr.update(value="", visible=False), fallback_note


def piper_speed_note_update(engine_id: str, voice_id: str | None, supports_speed: bool) -> dict:
    if engine_id != "piper" or not voice_id or supports_speed:
        return gr.update(value="", visible=False)
    return gr.update(value="Vitesse non supportée par cette voix", visible=True)


def language_warning(engine_id: str, requested: str, final: str, did_fallback: bool) -> str:
    if not did_fallback:
        return ""
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    return f"[{stamp}] tts_language_fallback requested={requested} -> {final} (engine={engine_id})"


def piper_voice_status_text(voices) -> str:
    if not voices:
        return "⚠️ Aucune voix Piper installée"
    return f"✅ {len(voices)} voix Piper installée(s)"

