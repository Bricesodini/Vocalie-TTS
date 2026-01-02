from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Tuple

import gradio as gr
import httpx


API_BASE_URL = os.environ.get("CHATTERBOX_API_URL", "http://127.0.0.1:8000")


def _get(url: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{API_BASE_URL}{url}", params=params)
        resp.raise_for_status()
        return resp.json()


def _post(url: str, payload: dict) -> dict:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{API_BASE_URL}{url}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _download(url: str) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_BASE_URL}{url}")
        resp.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as fh:
            fh.write(resp.content)
        return path


def fetch_engines() -> tuple[List[Tuple[str, str]], Dict[str, bool]]:
    try:
        data = _get("/v1/tts/engines")
        engines = data.get("engines", [])
        choices = [(e["label"], e["id"]) for e in engines]
        supports_ref = {e["id"]: bool(e.get("supports_ref")) for e in engines}
        return choices, supports_ref
    except Exception:
        return [], {}


def fetch_voices(engine_id: str) -> List[Tuple[str, str]]:
    if not engine_id:
        return []
    try:
        data = _get("/v1/tts/voices", params={"engine": engine_id})
        return [(v["label"], v["id"]) for v in data.get("voices", [])]
    except Exception:
        return []


def update_engine(engine_id: str, supports_ref: Dict[str, bool]):
    needs_ref = bool(supports_ref.get(engine_id))
    if not needs_ref:
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=True),
            "",
        )
    voices = fetch_voices(engine_id)
    if voices:
        return (
            gr.update(choices=voices, value=voices[0][1], visible=True),
            gr.update(interactive=True),
            "",
        )
    return (
        gr.update(choices=voices, value=None, visible=True),
        gr.update(interactive=False),
        "Aucune voix de reference disponible.",
    )


def generate_tts(text: str, engine: str, voice: str | None, supports_ref: Dict[str, bool]):
    if not text or not text.strip():
        return None, "Texte vide."
    if supports_ref.get(engine) and not voice:
        return None, "Aucune voix disponible."
    payload = {
        "text": text,
        "engine": engine,
        "voice": voice,
        "direction": {"enabled": False},
    }
    try:
        job = _post("/v1/tts/jobs", payload)
        job_id = job.get("job_id")
        if not job_id:
            return None, "Job invalide."
        asset_id = None
        status = "queued"
        for _ in range(60):
            data = _get(f"/v1/jobs/{job_id}")
            status = data.get("status")
            if status in {"done", "error", "canceled"}:
                asset_id = data.get("asset_id")
                break
            time.sleep(0.5)
        if status != "done" or not asset_id:
            return None, f"Job {status}."
        audio_path = _download(f"/v1/assets/{asset_id}")
        return audio_path, f"OK (asset_id={asset_id})"
    except Exception as exc:
        return None, f"Erreur: {exc}"


def build_cockpit() -> gr.Blocks:
    engines, supports_ref = fetch_engines()
    default_engine = engines[0][1] if engines else None

    with gr.Blocks(title="Chatterbox Cockpit") as demo:
        gr.Markdown("# Chatterbox Cockpit (API)")
        with gr.Row():
            engine = gr.Dropdown(label="Moteur", choices=engines, value=default_engine)
            voice = gr.Dropdown(label="Voix", choices=[], visible=False)
        text = gr.Textbox(label="Texte", lines=4, placeholder="Saisissez votre texte...")
        generate_btn = gr.Button("Générer")
        status = gr.Markdown("")
        audio = gr.Audio(label="Résultat", type="filepath")

        supports_ref_state = gr.State(supports_ref)
        engine.change(
            update_engine,
            inputs=[engine, supports_ref_state],
            outputs=[voice, generate_btn, status],
        )
        demo.load(
            update_engine,
            inputs=[engine, supports_ref_state],
            outputs=[voice, generate_btn, status],
        )
        generate_btn.click(
            generate_tts,
            inputs=[text, engine, voice, supports_ref_state],
            outputs=[audio, status],
        )

    return demo


if __name__ == "__main__":
    build_cockpit().launch()
