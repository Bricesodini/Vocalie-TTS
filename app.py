"""Gradio interface for the local Chatterbox French TTS tool."""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

from output_paths import make_output_filename, prepare_output_paths
from refs import (
    ALLOWED_EXTENSIONS,
    DEFAULT_REF_DIR,
    import_refs,
    list_refs,
    resolve_ref_path,
)
from state_manager import (
    delete_preset,
    list_presets,
    load_preset,
    load_state,
    save_preset,
    save_state,
)
from text_tools import (
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_MAX_PHRASES_PER_CHUNK,
    FINAL_MERGE_EST_SECONDS,
    adjust_text_to_duration,
    chunk_script,
    render_clean_text,
    render_clean_text_from_segments,
)
from tts_engine import TTSEngine


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("chatterbox_app")

BASE_DIR = Path(__file__).resolve().parent

SAFE_PREVIEW_DIR = BASE_DIR / "output"
SAFE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

output_env = os.environ.get("CHATTERBOX_OUT_DIR")
if output_env:
    DEFAULT_OUTPUT_DIR = Path(output_env).expanduser()
else:
    DEFAULT_OUTPUT_DIR = SAFE_PREVIEW_DIR
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENGINE: Optional[TTSEngine] = None


def mac_choose_folder(initial_dir: str | None = None) -> str | None:
    if sys.platform != "darwin":
        return None

    script = 'POSIX path of (choose folder'
    if initial_dir:
        safe_dir = initial_dir.replace('"', '\\"')
        script += f' default location POSIX file "{safe_dir}"'
    script += ')'

    try:
        out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
        return out if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_engine() -> TTSEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = TTSEngine()
    return ENGINE


def ensure_output_dir(path: str | None) -> str:
    target = Path(path).expanduser() if path else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def persist_state(update: dict) -> None:
    state = load_state()
    state.update(update)
    save_state(state)


def append_log(message: str, previous: str | None) -> str:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    new_line = f"[{stamp}] {message}"
    if previous:
        return f"{previous}\n{new_line}"
    return new_line


def append_ui_log(message: str, previous: str | None, verbose: bool = False, enabled: bool = True) -> str:
    if verbose and not enabled:
        return previous or ""
    return append_log(message, previous)


def refresh_dropdown(current: str | None) -> gr.Dropdown:
    refs = list_refs()
    value = current if current in refs else (refs[0] if refs else None)
    return gr.update(choices=refs, value=value)


def handle_upload(files, current: str | None, log_text: str | None):
    saved = import_refs(files)
    refs = list_refs()
    value = current if current in refs else (refs[0] if refs else None)
    if saved:
        value = saved[-1]
        log_text = append_log(f"Import réussi: {', '.join(saved)}", log_text)
    else:
        log_text = append_log("Aucun fichier importé.", log_text)
    return gr.update(choices=refs, value=value), log_text


def handle_adjust(text: str, target_seconds: float | None, log_text: str | None):
    target = float(target_seconds) if target_seconds else 0.0
    result = adjust_text_to_duration(text, target)
    info = f"Durée estimée: {result.estimated_duration:.1f}s"
    if result.target_duration > 0:
        info += f" / cible {result.target_duration:.1f}s"
    if result.warning:
        info += f"\n⚠️ {result.warning}"
    log_text = append_ui_log("Suggestion de durée calculée.", log_text)
    return result.text, info, log_text


def apply_adjusted(preview_text: str) -> str:
    return preview_text


def update_clean_preview(text: str) -> str:
    clean = render_clean_text(text)
    return clean


def handle_load_preset(
    preset_name: str | None,
    log_text: str | None,
):
    outputs = [gr.update() for _ in range(13)]
    name_update = gr.update()
    if not preset_name:
        log_text = append_log("Sélectionnez un preset à charger.", log_text)
        return (*outputs, name_update, log_text)

    data = load_preset(preset_name)
    if not data:
        log_text = append_log(f"Preset introuvable: {preset_name}", log_text)
        return (*outputs, name_update, log_text)

    refs = list_refs()
    ref_value = data.get("ref_name")
    if ref_value not in refs:
        ref_value = None

    updates = [
        gr.update(value=ref_value),
        gr.update(value=data.get("out_dir") or str(DEFAULT_OUTPUT_DIR)),
        gr.update(value=data.get("user_filename", "")),
        gr.update(value=bool(data.get("add_timestamp", True))),
        gr.update(value=bool(data.get("long_form", False))),
        gr.update(value=int(data.get("max_chars", DEFAULT_MAX_CHARS_PER_CHUNK))),
        gr.update(value=int(data.get("max_sentences", DEFAULT_MAX_PHRASES_PER_CHUNK))),
        gr.update(value=bool(data.get("verbose_logs", False))),
        gr.update(value=_coerce_float(data.get("exaggeration"), 0.5)),
        gr.update(value=_coerce_float(data.get("cfg_weight"), 0.6)),
        gr.update(value=_coerce_float(data.get("temperature"), 0.5)),
        gr.update(value=_coerce_float(data.get("repetition_penalty"), 1.35)),
        gr.update(value=preset_name),
    ]
    name_update = gr.update(value=preset_name)
    persist_state({"last_preset": preset_name})
    log_text = append_log(f"Preset chargé: {preset_name}", log_text)
    return (*updates, name_update, log_text)


def handle_save_preset(
    preset_name: str | None,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    long_form: bool,
    max_chars: int,
    max_sentences: int,
    verbose_logs: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    log_text: str | None,
):
    if not preset_name:
        log_text = append_log("Nom de preset requis.", log_text)
        return gr.update(), gr.update(), log_text

    data = {
        "ref_name": ref_name,
        "out_dir": out_dir,
        "user_filename": user_filename or "",
        "add_timestamp": bool(add_timestamp),
        "long_form": bool(long_form),
        "max_chars": int(max_chars),
        "max_sentences": int(max_sentences),
        "verbose_logs": bool(verbose_logs),
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
    }

    try:
        preset_slug = save_preset(preset_name, data)
    except ValueError as exc:
        log_text = append_log(f"Nom de preset invalide: {exc}", log_text)
        return gr.update(), gr.update(), log_text

    choices = list_presets()
    persist_state({"last_preset": preset_slug})
    log_text = append_log(f"Preset sauvegardé: {preset_slug}", log_text)
    return (
        gr.update(choices=choices, value=preset_slug),
        gr.update(value=preset_slug),
        log_text,
    )


def handle_delete_preset(
    preset_name: str | None,
    log_text: str | None,
):
    if not preset_name:
        log_text = append_log("Aucun preset à supprimer.", log_text)
        return gr.update(), gr.update(), log_text

    try:
        delete_preset(preset_name)
    except ValueError as exc:
        log_text = append_log(f"Suppression impossible: {exc}", log_text)
        return gr.update(), gr.update(), log_text

    choices = list_presets()
    log_text = append_log(f"Preset supprimé: {preset_name}", log_text)
    persist_state({"last_preset": None})
    return (
        gr.update(choices=choices, value=None),
        gr.update(value=""),
        log_text,
    )


def handle_choose_output(current_path: str | None, log_text: str | None):
    base = current_path or str(DEFAULT_OUTPUT_DIR)
    chosen = mac_choose_folder(base)
    if not chosen:
        log_text = append_ui_log("Sélection dossier annulée.", log_text)
        return gr.update(value=current_path), log_text
    persist_state({"last_out_dir": chosen})
    log_text = append_ui_log(f"Dossier sortie choisi: {chosen}", log_text)
    return gr.update(value=chosen), log_text


def handle_reset_chunk_defaults(log_text: str | None):
    log_text = append_ui_log("Reset chunking defaults.", log_text)
    return (
        gr.update(value=DEFAULT_MAX_CHARS_PER_CHUNK),
        gr.update(value=DEFAULT_MAX_PHRASES_PER_CHUNK),
        log_text,
    )


def handle_generate(
    text: str,
    ref_name: str | None,
    out_dir: str | None,
    user_filename: str | None,
    add_timestamp: bool,
    long_form: bool,
    max_chars: int,
    max_sentences: int,
    verbose_logs: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    log_text: str | None,
):
    if not text or not text.strip():
        return None, "", "", append_log("Erreur: texte vide.", log_text)

    log_text = append_ui_log("Initialisation de la génération...", log_text)

    audio_prompt = None
    if ref_name:
        try:
            audio_prompt = resolve_ref_path(ref_name)
        except FileNotFoundError:
            log_text = append_ui_log(f"Référence introuvable: {ref_name}", log_text)
            return None, "", "", log_text

    output_dir = ensure_output_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = make_output_filename(
        text=text,
        ref_name=ref_name,
        user_filename=user_filename,
        add_timestamp=bool(add_timestamp),
        timestamp=timestamp,
    )
    preview_path, user_path = prepare_output_paths(SAFE_PREVIEW_DIR, output_dir, filename)
    clean_preview = render_clean_text(text)
    if clean_preview:
        log_text = append_ui_log("Texte interprété prêt.", log_text)

        chunk_preview_text = ""
        if long_form:
            chunks = chunk_script(text, max_chars=int(max_chars), max_sentences=int(max_sentences))
            lines = []
            for idx, chunk_info in enumerate(chunks, start=1):
                chunk_text = render_clean_text_from_segments(chunk_info.segments)
                oversize_flag = " oversize_sentence=True" if chunk_info.oversize_sentence else ""
                lines.append(
                    f"[{idx}] phrases={chunk_info.sentence_count} "
                    f"chars={chunk_info.char_count} "
                    f"est={chunk_info.estimated_duration:.1f}s "
                    f"reason={chunk_info.reason}{oversize_flag}\n{chunk_text}"
                )
            if chunk_info.reason and chunk_info.reason != "end":
                log_text = append_ui_log(
                    f"Split reason: {chunk_info.reason}", log_text, verbose=True, enabled=verbose_logs
                )
        chunk_preview_text = "\n\n".join(lines)
        log_text = append_ui_log(
            "Split reason: phrase-first", log_text, verbose=True, enabled=verbose_logs
        )
        log_text = append_ui_log(
            f"Chunks: {len(chunks)}", log_text, verbose=False, enabled=True
        )

    try:
        engine = get_engine()
        log_text = append_log("Synthèse en cours...", log_text)
        if long_form:
            final_path, _, meta = engine.generate_longform(
                script=text,
                audio_prompt_path=audio_prompt,
                out_path=str(preview_path),
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_chars=int(max_chars),
                max_sentences=int(max_sentences),
            )
            for idx, duration in enumerate(meta.get("durations", []), start=1):
                retry_flag = meta.get("retries", [])[idx - 1] if meta.get("retries") else False
                chunk_info = chunks[idx - 1] if idx - 1 < len(chunks) else None
                reason = chunk_info.reason if chunk_info else "n/a"
                phrases = chunk_info.sentence_count if chunk_info else 0
                chars = chunk_info.char_count if chunk_info else 0
                est = chunk_info.estimated_duration if chunk_info else 0.0
                retry_note = " retry" if retry_flag else ""
                log_text = append_ui_log(
                    f"Chunk {idx}/{meta.get('chunks', len(meta.get('durations', [])))} "
                    f"reason={reason} phrases={phrases} chars={chars} "
                    f"est={est:.1f}s measured={duration:.2f}s{retry_note}",
                    log_text,
                    verbose=True,
                    enabled=verbose_logs,
                )
            total_duration = meta.get("total_duration")
            if total_duration is not None:
                log_text = append_ui_log(f"Durée finale: {total_duration:.2f}s", log_text)
        else:
            final_path, _ = engine.generate(
                text=text,
                audio_prompt_path=audio_prompt,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                out_path=preview_path,
            )
    except Exception as exc:
        LOGGER.exception("TTS generation failed")
        log_text = append_ui_log(f"Erreur TTS: {exc}", log_text)
        return None, "", chunk_preview_text, log_text

    user_path_obj = Path(user_path)
    preview_path_obj = Path(final_path)
    if user_path_obj.resolve() != preview_path_obj.resolve():
        try:
            user_path_obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview_path_obj, user_path_obj)
            log_text = append_ui_log(f"Copie dossier utilisateur: {user_path_obj}", log_text)
        except Exception as exc:
            log_text = append_ui_log(f"Copie échouée: {exc}", log_text)
    persist_state(
        {
            "last_ref": ref_name,
            "last_out_dir": output_dir,
            "last_exaggeration": float(exaggeration),
            "last_cfg_weight": float(cfg_weight),
            "last_temperature": float(temperature),
            "last_repetition_penalty": float(repetition_penalty),
            "last_user_filename": user_filename or "",
            "last_add_timestamp": bool(add_timestamp),
            "last_long_form": bool(long_form),
            "last_max_chars": int(max_chars),
            "last_max_sentences": int(max_sentences),
            "last_verbose_logs": bool(verbose_logs),
        }
    )
    log_text = append_ui_log(f"Fichier pré-écoute: {preview_path_obj}", log_text)
    return str(preview_path_obj), str(user_path_obj), chunk_preview_text, log_text


def build_ui() -> gr.Blocks:
    initial_refs = list_refs()
    state_data = load_state()
    default_ref = state_data.get("last_ref")
    if default_ref not in initial_refs:
        default_ref = initial_refs[0] if initial_refs else None
    default_out_dir_value = state_data.get("last_out_dir") or str(DEFAULT_OUTPUT_DIR)
    default_user_filename = state_data.get("last_user_filename", "")
    default_add_timestamp = _coerce_bool(state_data.get("last_add_timestamp"), True)
    default_long_form = _coerce_bool(state_data.get("last_long_form"), False)
    default_max_chars = int(state_data.get("last_max_chars") or DEFAULT_MAX_CHARS_PER_CHUNK)
    default_max_sentences = int(state_data.get("last_max_sentences") or DEFAULT_MAX_PHRASES_PER_CHUNK)
    default_verbose_logs = _coerce_bool(state_data.get("last_verbose_logs"), False)
    default_exaggeration = _coerce_float(state_data.get("last_exaggeration"), 0.5)
    default_cfg = _coerce_float(state_data.get("last_cfg_weight"), 0.6)
    default_temperature = _coerce_float(state_data.get("last_temperature"), 0.5)
    default_repetition = _coerce_float(state_data.get("last_repetition_penalty"), 1.35)
    preset_choices = list_presets()
    default_preset = state_data.get("last_preset")
    if default_preset not in preset_choices:
        default_preset = None

    with gr.Blocks(title="Chatterbox TTS FR", css=".section-title {font-weight:600;}") as demo:
        gr.Markdown("""# 🎙️ Chatterbox TTS FR\nInterface locale pour générer des voix off expressives.""")

        with gr.Group():
            gr.Markdown("## Références vocales")
            with gr.Row():
                ref_dropdown = gr.Dropdown(
                    label="Choisir une référence",
                    choices=initial_refs,
                    value=default_ref,
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh", size="sm")
            upload = gr.Files(
                label="Importer des fichiers audio",
                file_types=list(ALLOWED_EXTENSIONS),
                file_count="multiple",
            )

        with gr.Group():
            gr.Markdown("## Texte")
            text_input = gr.Textbox(label="Texte", lines=8, placeholder="Collez votre script ici...")
            with gr.Row():
                target_duration = gr.Number(label="Durée cible (s)", value=None, precision=1)
                adjust_btn = gr.Button("Ajuster le texte")
                apply_btn = gr.Button("Utiliser la suggestion")
            adjusted_preview = gr.Textbox(
                label="Suggestion texte",
                lines=6,
                interactive=False,
                placeholder="Le texte ajusté apparaîtra ici...",
            )
            adjust_info = gr.Markdown("Durée estimée: --")
            with gr.Accordion("Voir le texte final", open=False):
                clean_text_box = gr.Textbox(
                    label="Texte interprété (envoyé au TTS)",
                    lines=4,
                    interactive=False,
                    placeholder="Le texte sans balises apparaîtra ici...",
                )
                gr.Markdown(
                    "⚠️ La ponctuation finale peut être renforcée à la synthèse, sans modifier cet aperçu."
                )
            with gr.Accordion("Long-form / découpage", open=False):
                with gr.Row():
                    long_form_toggle = gr.Checkbox(
                        label="Long-form (auto-chunk)",
                        value=default_long_form,
                    )
                    max_chars_slider = gr.Slider(
                        220,
                        600,
                        value=default_max_chars,
                        step=10,
                        label="Max chars/chunk",
                        info="Utilisé uniquement si aucune coupure naturelle n’est possible.",
                    )
                    max_sentences_slider = gr.Slider(
                        1,
                        6,
                        value=default_max_sentences,
                        step=1,
                        label="Max phrases/chunk",
                        info=f"Nombre maximal de phrases par génération. Seuil merge final: {FINAL_MERGE_EST_SECONDS:.1f}s.",
                    )
                    reset_chunk_btn = gr.Button("↺", size="sm")
                    verbose_logs_toggle = gr.Checkbox(
                        label="Logs détaillés",
                        value=default_verbose_logs,
                    )
                chunk_preview_box = gr.Textbox(
                    label="Aperçu des chunks",
                    lines=8,
                    interactive=False,
                    placeholder="Aperçu des chunks (mode long-form).",
                )

        with gr.Group():
            gr.Markdown("## Paramètres")
            with gr.Row():
                exaggeration_slider = gr.Slider(
                    0.0,
                    1.5,
                    value=default_exaggeration,
                    step=0.05,
                    label="Exagération",
                    info="Expressivité globale (0.5 recommandé)",
                )
                cfg_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=default_cfg,
                    step=0.05,
                    label="CFG",
                    info="Plus haut = voix plus rigoureuse",
                )
            with gr.Row():
                temperature_slider = gr.Slider(
                    0.1,
                    1.0,
                    value=default_temperature,
                    step=0.05,
                    label="Température",
                    info="Stabilité vs créativité",
                )
                repetition_slider = gr.Slider(
                    0.8,
                    2.0,
                    value=default_repetition,
                    step=0.05,
                    label="Repetition Penalty",
                    info="Limite les répétitions",
                )

        with gr.Group():
            gr.Markdown("## Presets")
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="Presets",
                    choices=preset_choices,
                    value=default_preset,
                )
                preset_name_box = gr.Textbox(
                    label="Nom preset",
                    value=default_preset or "",
                    placeholder="ex: pub-dynamique",
                )
            with gr.Row():
                load_preset_btn = gr.Button("Charger")
                save_preset_btn = gr.Button("Sauver")
                delete_preset_btn = gr.Button("Supprimer")

        with gr.Group():
            gr.Markdown("## Sortie")
            with gr.Row():
                output_dir_box = gr.Textbox(
                    label="Dossier de sortie",
                    value=default_out_dir_value,
                    lines=1,
                )
                choose_btn = gr.Button("Choisir…", size="sm")
            with gr.Row():
                filename_box = gr.Textbox(
                    label="Nom de fichier (optionnel)",
                    value=default_user_filename,
                    placeholder="Ex: spot_festival_voix_finale",
                )
                timestamp_toggle = gr.Checkbox(
                    label="Ajouter timestamp",
                    value=default_add_timestamp,
                )
            generate_btn = gr.Button("Générer", variant="primary")
            result_audio = gr.Audio(label="Pré-écoute", type="filepath", autoplay=False)
            output_path_box = gr.Textbox(label="Fichier généré", interactive=False)

        logs_box = gr.Textbox(label="Logs", lines=8, interactive=False)

        refresh_btn.click(refresh_dropdown, inputs=ref_dropdown, outputs=ref_dropdown)
        upload.upload(
            fn=handle_upload,
            inputs=[upload, ref_dropdown, logs_box],
            outputs=[ref_dropdown, logs_box],
        )
        adjust_btn.click(
            fn=handle_adjust,
            inputs=[text_input, target_duration, logs_box],
            outputs=[adjusted_preview, adjust_info, logs_box],
        )
        apply_btn.click(fn=apply_adjusted, inputs=adjusted_preview, outputs=text_input)
        text_input.change(fn=update_clean_preview, inputs=text_input, outputs=clean_text_box)

        choose_btn.click(
            fn=handle_choose_output,
            inputs=[output_dir_box, logs_box],
            outputs=[output_dir_box, logs_box],
        )
        reset_chunk_btn.click(
            fn=handle_reset_chunk_defaults,
            inputs=[logs_box],
            outputs=[max_chars_slider, max_sentences_slider, logs_box],
        )
        load_preset_btn.click(
            fn=handle_load_preset,
            inputs=[preset_dropdown, logs_box],
            outputs=[
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                long_form_toggle,
                max_chars_slider,
                max_sentences_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
                preset_dropdown,
                preset_name_box,
                logs_box,
            ],
        )
        save_preset_btn.click(
            fn=handle_save_preset,
            inputs=[
                preset_name_box,
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                long_form_toggle,
                max_chars_slider,
                max_sentences_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
                logs_box,
            ],
            outputs=[preset_dropdown, preset_name_box, logs_box],
        )
        delete_preset_btn.click(
            fn=handle_delete_preset,
            inputs=[preset_dropdown, logs_box],
            outputs=[preset_dropdown, preset_name_box, logs_box],
        )

        generate_btn.click(
            fn=handle_generate,
            inputs=[
                text_input,
                ref_dropdown,
                output_dir_box,
                filename_box,
                timestamp_toggle,
                long_form_toggle,
                max_chars_slider,
                max_sentences_slider,
                verbose_logs_toggle,
                exaggeration_slider,
                cfg_slider,
                temperature_slider,
                repetition_slider,
                logs_box,
            ],
            outputs=[result_audio, output_path_box, chunk_preview_box, logs_box],
        )

    return demo


def main() -> None:
    demo = build_ui()
    port_env = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {}
    if port_env:
        try:
            launch_kwargs["server_port"] = int(port_env)
        except ValueError:
            LOGGER.warning("GRADIO_SERVER_PORT invalide (%s), fallback 7860.", port_env)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
