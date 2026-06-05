# Plan d'Intégration — Nouveaux Modèles TTS

**Date :** 2026-06-04  
**Philosophie :** « Le plus petit nombre de modèles possible pour les meilleures voix possibles. »  
**Contrainte :** Uniquement modèles installables en local avec licence permissive (Apache 2.0, MIT, etc.)

---

## État actuel

### Backends actifs (2)
| Backend | Engine IDs | Supports ref_audio | Supports inter_chunk_gap | Status |
|---------|-----------|-------------------|--------------------------|--------|
| `chatterbox` | `chatterbox_native`, `chatterbox_finetune_fr` | ✅ | ✅ | Actif, venv isolé |
| `qwen3` | `qwen3_custom`, `qwen3_clone` | ✅ (clone only) | ✅ | Actif, venv isolé |

### Backends legacy (3) — DECISION: supprimer
| Backend | Engine ID | Status |
|---------|-----------|--------|
| `bark` | `bark` | `is_available: False`, fichiers présents |
| `piper` | `piper` | `is_available: False`, fichiers présents |
| `xtts` | `xtts_v2` | `is_available: False`, fichiers présents |

**Décision existante (DEC-007):** Bark, Piper, XTTS v2 sont qualitativement inférieurs et doivent être supprimés.

---

## Candidats — Licence permissive uniquement

| Modèle | Licence | Params | Langues | Clone voix | Streaming | CPU? | Qualité FR |
|--------|---------|--------|---------|-------------|-----------|------|------------|
| 🥇 **CosyVoice 3** | Apache 2.0 ✅ | 0.5B | 9 (FR✅) | ✅ zero-shot | ✅ 150ms | Non | ⭐⭐⭐⭐ |
| 🥈 **Orpheus TTS** | Apache 2.0 ✅ | 3B | EN + 7 multilingue (research) | ✅ zero-shot | ✅ ~200ms | Non | ⭐⭐⭐⭐ |
| 🥉 **MOSS-TTS** | Apache 2.0 ✅ | 8B (flagship) / 0.1B (Nano) | 31 | ✅ zero-shot | ✅ 180ms | ✅ Nano | ⭐⭐⭐⭐ |
| ❌ **Voxtral TTS** | CC BY-NC 4.0 | 4B | 9 (FR✅) | ✅ zero-shot | ✅ 70ms | Non | ⭐⭐⭐⭐⭐ |
| ❌ **Fish Speech** | CC BY-NC-SA 4.0 | 1.5B | Multi | ✅ | — | Non | ⭐⭐⭐⭐ |

**Voxtral et Fish Speech éliminés** : licence NC incompatible avec usage local utilisable.

### 🥇 CosyVoice 3 — Priorité #1

**Pourquoi CosyVoice 3 en priorité ?**
- Apache 2.0 → pas de restriction d'usage
- 0.5B params → léger, GPU modeste suffisant
- 9 langues dont FR, EN, ES, DE, IT, RU, JA, KO, ZH
- 3 modes : zero-shot clone, cross-lingual, instruct (émotions/dialectes/roles)
- Streaming natif (150ms first-packet)
- vLLM support pour déploiement scalable
- Active development : Fun-CosyVoice3-0.5B-2512 + RL model released

| Critère | Évaluation |
|---------|------------|
| Modèle HF | `FunAudioLLM/CosyVoice2-0.5B` / `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| Licence | Apache 2.0 ✅ |
| GPU requis | ≥8GB VRAM (0.5B modèle) |
| Installation | `git clone --recursive` + conda env + `pip install -r requirements.txt` |
| API Python | `cosyvoice.inference_zero_shot()`, `.inference_cross_lingual()`, `.inference_instruct2()` |

**Engine IDs proposés :**
- `cosyvoice_instruct` — Mode instruction (text + émotion/style prompt + ref audio)
- `cosyvoice_clone` — Mode zero-shot clone (ref audio 3s+)
- `cosyvoice_cross` — Mode cross-lingual (voix FR → texte EN, etc.)

### 🥈 Orpheus TTS — Priorité #2

**Pourquoi Orpheus ?**
- Apache 2.0 ✅
- Llama-3b backbone → architecture LLM pur
- 8 voix prédéfinies + zero-shot cloning
- Contrôle émotionnel via tags (`<laugh>`, `<sigh>`, `<giggle>`, etc.)
- ~200ms streaming latency
- Finetuning facile (50-300 exemples suffisent)
- ⚠️ Multilingue en "research preview" — qualité FR à vérifier

| Critère | Évaluation |
|---------|------------|
| Modèle HF | `canopylabs/orpheus-tts-0.1-finetune-prod` |
| Licence | Apache 2.0 ✅ |
| GPU requis | ≥8GB VRAM |
| Installation | `pip install orpheus-speech` (utilise vLLM) |

**Engine IDs proposés :**
- `orpheus_tts` — Mode standard

### 🥉 MOSS-TTS — Priorité #3 (futur)

**Pourquoi MOSS ?**
- Apache 2.0 ✅
- La famille la plus complète : TTS, TTS-Nano, TTSD (dialogue), VoiceGenerator, SoundEffect, Realtime
- **MOSS-TTS-Nano (0.1B) tourne sur CPU** → atout majeur pour déploiement léger
- 31 langues supportées — couverture la plus large
- vLLM-Omni, SGLang, llama.cpp backends — le plus flexible en déploiement
- 48kHz stéréo natif
- ⚠️ 8B pour le flagship → GPU ≥16GB

| Critère | Évaluation |
|---------|------------|
| Modèle HF | `OpenMOSS-Team/MOSS-TTS-v1.5` (8B), `OpenMOSS-Team/MOSS-TTS-Nano` (0.1B) |
| Licence | Apache 2.0 ✅ |
| GPU requis | 8B: ≥16GB / Nano: CPU possible |
| Installation | conda + `pip install -e ".[torch-runtime]"` |

**Engine IDs proposés :**
- `moss_tts` — Flagship (8B, GPU)
- `moss_tts_nano` — Lightweight (CPU/edge)

---

## Phase 0 — Nettoyage Legacy (préalable)

### 0.1 Supprimer les backends legacy

| Fichier | Action |
|---------|--------|
| `tts_backends/bark_backend.py` | Supprimer |
| `tts_backends/bark_runner.py` | Supprimer |
| `tts_backends/bark_prefetch.py` | Supprimer |
| `tts_backends/piper_backend.py` | Supprimer |
| `tts_backends/piper_runner.py` | Supprimer |
| `tts_backends/piper_assets.py` | Supprimer |
| `tts_backends/xtts_backend.py` | Supprimer |
| `tts_backends/xtts_runner.py` | Supprimer |
| `tts_backends/xtts_prefetch.py` | Supprimer |

### 0.2 Nettoyer les imports dans `__init__.py`

Supprimer les lignes :
```python
from .bark_backend import BarkBackend      # noqa: F401
from .piper_backend import PiperBackend     # noqa: F401
from .xtts_backend import XTTSBackend       # noqa: F401
```

### 0.3 Nettoyer le catalogue

- Supprimer les entrées `bark`, `piper`, `xtts_v2` du catalogue dynamique (automatique via `_REGISTRY`)
- Supprimer l'alias `"xtts": "xtts_v2"` de `ENGINE_ALIAS_MAP`
- Nettoyer `backend_install/manifests.py` : supprimer les manifests `bark`, `piper`, `xtts`
- Nettoyer `backend_install/installer.py` si nécessaire

### 0.4 Nettoyer les tests legacy

- Supprimer les tests spécifiques à Bark/Piper/XTTS s'ils existent encore
- Vérifier que le catalogue ne contient que les 4 engine IDs actifs

### 0.5 Nettoyer les scripts et requirements

- Supprimer `requirements-bark.txt`, `requirements-bark.lock.txt`
- Supprimer `scripts/install-bark-venv.sh`, `scripts/install-piper-voices.sh`
- Nettoyer les références dans scripts

**Estimation :** ~2h, 15-20 fichiers touchés

---

## Phase 1 — Intégration CosyVoice 3 (priorité #1)

### 1.1 Backend (`tts_backends/cosyvoice_backend.py`)

```python
class CosyVoiceBackend(TTSBackend, SubprocessBackendMixin):
    id = "cosyvoice"
    display_name = "CosyVoice 3"
    runner_module = "cosyvoice_runner"
    runner_venv = "cosyvoice"
    default_timeout = 300.0

    supports_ref_audio = True
    supports_inter_chunk_gap = True

    COSYVOICE_LANGUAGE_MAP = {
        "fr": "French", "en": "English", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "de": "German",
        "es": "Spanish", "it": "Italian", "ru": "Russian",
    }

    @classmethod
    def engine_variants(cls):
        return [
            {"id": "cosyvoice_instruct", "label": "CosyVoice (Instruct)"},
            {"id": "cosyvoice_clone", "label": "CosyVoice (Voice Clone)"},
            {"id": "cosyvoice_cross", "label": "CosyVoice (Cross-lingual)"},
        ]

    def list_models(self):
        return [
            ModelInfo(id="FunAudioLLM/CosyVoice2-0.5B", label="CosyVoice2 0.5B"),
            ModelInfo(id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", label="CosyVoice3 0.5B"),
        ]

    def params_schema(self):
        return {
            "temperature": ParamSpec(key="temperature", type="float", default=0.7,
                                     min=0.0, max=1.0, step=0.05),
            "streaming": ParamSpec(key="streaming", type="bool", default=False,
                                    label="Streaming mode"),
            "instruct_text": ParamSpec(key="instruct_text", type="str", default="",
                                        label="Instruction (emotion/style/role)",
                                        visible_if={"cosyvoice_instruct": True}),
        }

    @classmethod
    def capabilities(cls, engine_id=None):
        caps = super().capabilities(engine_id)
        caps["supports_voice_design"] = False
        caps["supports_cross_lingual"] = engine_id == "cosyvoice_cross"
        caps["supports_instruct"] = engine_id == "cosyvoice_instruct"
        caps["supports_streaming"] = True
        caps["supports_emotion"] = engine_id == "cosyvoice_instruct"
        return caps

    def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
        engine_id = params.get("engine_id") or "cosyvoice_clone"
        mode = {
            "cosyvoice_instruct": "instruct",
            "cosyvoice_clone": "clone",
            "cosyvoice_cross": "cross_lingual",
        }.get(engine_id, "clone")

        payload_suffix = {
            "mode": mode,
            "model_id": params.get("model_id", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
            "language": self.map_language(lang),
            "instruct_text": params.get("instruct_text", ""),
            "streaming": params.get("streaming", False),
        }
        if voice_ref_path:
            payload_suffix["voice_ref_path"] = voice_ref_path
        return self._run_subprocess_chunk(text, payload_suffix=payload_suffix, lang=lang)
```

### 1.2 Runner (`tts_backends/cosyvoice_runner.py`)

```python
class CosyVoiceRunner(BaseSubprocessRunner):
    def run_synthesis(self, payload):
        text = str(payload.get("text") or "")
        out_path = str(payload.get("out_path") or "")
        mode = str(payload.get("mode") or "clone")
        model_id = payload.get("model_id", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
        language = payload.get("language") or "French"
        instruct_text = payload.get("instruct_text") or ""
        voice_ref_path = payload.get("voice_ref_path")
        streaming = payload.get("streaming", False)

        import sys
        sys.path.append("third_party/Matcha-TTS")  # CosyVoice requirement
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio

        model = AutoModel(model_dir=model_id)

        if mode == "instruct":
            gen = model.inference_instruct2(text, instruct_text, voice_ref_path)
        elif mode == "cross_lingual":
            gen = model.inference_cross_lingual(text, voice_ref_path)
        else:  # clone
            prompt_text = payload.get("prompt_text", "")
            gen = model.inference_zero_shot(text, prompt_text, voice_ref_path,
                                             stream=streaming)

        # Collect all chunks and write final WAV
        all_audio = []
        sr = model.sample_rate
        for chunk in gen:
            all_audio.append(chunk["tts_speech"])

        import torch
        audio = torch.cat(all_audio, dim=1) if len(all_audio) > 1 else all_audio[0]
        torchaudio.save(out_path, audio, sr)
        duration_s = audio.shape[-1] / sr
        return {"ok": True, "sample_rate": sr, "duration_ms": int(duration_s * 1000)}

if __name__ == "__main__":
    raise SystemExit(CosyVoiceRunner.main())
```

### 1.3 Import + Manifest

```python
# __init__.py
from .cosyvoice_backend import CosyVoiceBackend  # noqa: F401

# manifests.py
"cosyvoice": BackendManifest(
    engine_id="cosyvoice",
    python="python3.10",
    pip_packages=["torch", "torchaudio", "soundfile", "conformer", "matcha-tts"],
    system_hints=["NVIDIA GPU ≥ 8GB VRAM", "ffmpeg recommended"],
    import_probes=["cosyvoice"],
    post_install_checks=[["-c", "from cosyvoice.cli.cosyvoice import AutoModel; print('OK')"]],
),
```

---

## Phase 2 — Intégration Orpheus TTS (priorité #2)

### 2.1 Backend (`tts_backends/orpheus_backend.py`)

```python
class OrpheusBackend(TTSBackend, SubprocessBackendMixin):
    id = "orpheus"
    display_name = "Orpheus TTS"
    runner_module = "orpheus_runner"
    runner_venv = "orpheus"
    default_timeout = 300.0

    supports_ref_audio = True
    supports_inter_chunk_gap = True

    @classmethod
    def engine_variants(cls):
        return [{"id": "orpheus_tts", "label": "Orpheus TTS"}]

    def list_models(self):
        return [
            ModelInfo(id="canopylabs/orpheus-tts-0.1-finetune-prod", label="Orpheus 0.1 Finetune"),
        ]

    def params_schema(self):
        return {
            "voice": ParamSpec(key="voice", type="str", default="tara",
                               label="Voice", enum=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]),
            "temperature": ParamSpec(key="temperature", type="float", default=0.7,
                                     min=0.0, max=1.5, step=0.05),
        }

    @classmethod
    def capabilities(cls, engine_id=None):
        caps = super().capabilities(engine_id)
        caps["supports_streaming"] = True
        caps["supports_emotion"] = True  # via tags in text
        return caps

    def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
        payload_suffix = {
            "voice": params.get("voice", "tara"),
            "model_name": "canopylabs/orpheus-tts-0.1-finetune-prod",
        }
        if voice_ref_path:
            payload_suffix["voice_ref_path"] = voice_ref_path
        return self._run_subprocess_chunk(text, payload_suffix=payload_suffix, lang=lang)
```

---

## Phase 3 — Intégration MOSS-TTS (priorité #3, futur)

À planifier quand Phase 1 et 2 sont stables. Points clés :
- `moss_tts` (8B flagship, GPU) → mode premium
- `moss_tts_nano` (0.1B, CPU) → mode léger/edge
- 31 langues — la couverture la plus large
- VoiceGenerator + SoundEffect comme futures capabilities

---

## Ordre d'exécution

| Phase | Action | Fichiers | Durée estimée |
|-------|--------|----------|---------------|
| **0.1** | Supprimer backends legacy | ~9 fichiers | 30 min |
| **0.2** | Nettoyer `__init__.py` | 1 fichier | 5 min |
| **0.3** | Nettoyer catalogue + manifest | 2 fichiers | 15 min |
| **0.4** | Nettoyer tests legacy | 3-5 fichiers | 30 min |
| **0.5** | Nettoyer scripts + requirements | 4-5 fichiers | 15 min |
| **0.V** | Vérifier tests passent | — | 10 min |
| **1.1** | Créer `cosyvoice_backend.py` | 1 fichier | 1.5h |
| **1.2** | Créer `cosyvoice_runner.py` | 1 fichier | 1.5h |
| **1.3** | Ajouter import + manifest + prefetch | 3 fichiers | 30 min |
| **1.4** | Installer venv + tester | — | 2h |
| **1.V** | Valider endpoints + tests | — | 30 min |
| **2.1** | Créer `orpheus_backend.py` | 1 fichier | 1.5h |
| **2.2** | Créer `orpheus_runner.py` | 1 fichier | 1.5h |
| **2.3** | Installer + tester | — | 2h |
| **3** | MOSS-TTS (futur) | — | — |
| **4** | Mettre à jour `architecture.md` + `DECISIONS.md` | 2 fichiers | 30 min |

**Total estimé : Phase 0 ≈ 1.5h, Phase 1 ≈ 6h, Phase 2 ≈ 5h**

---

## Risques et mitigations

| Risque | Mitigation |
|--------|------------|
| CosyVoice API instable/changements | Encapsuler dans runner subprocess, version lock dans manifest |
| Dépendances CUDA conflictuelles | Venv isolé (cosyvoice), `runner_venv = "cosyvoice"` |
| CosyVoice FR qualité inférieure à Voxtral | CosyVoice3 RL model améliore FR (CER 5.44→0.81 sur hard) |
| Orpheus multilingue research preview | Tester qualité FR avant intégration, fallback sur CosyVoice |
| Régression catalogue | Tests vérifient que le catalogue ne contient que les engine IDs attendus |
| Conflit de noms `cosyvoice_mode` | Utiliser `engine_variants()` pour résoudre le mode, pas de if/elif |
| GPU requise pour CosyVoice/Orpheus | MOSS-TTS-Nano comme fallback CPU (Phase 3) |

---

## Critère de validation

### Phase 0
- [ ] Les 3 backends legacy ne sont plus dans le catalogue
- [ ] `pytest tests/test_tts_backends.py` passe
- [ ] Les 4 engine IDs actifs sont toujours présents

### Phase 1 (CosyVoice)
- [ ] `GET /v1/tts/engines` retourne les 7 engine IDs (4 actuels + 3 CosyVoice)
- [ ] `GET /v1/tts/models?engine=cosyvoice_clone` retourne les modèles
- [ ] `GET /v1/tts/engine_schema?engine=cosyvoice_instruct` retourne le schema avec `instruct_text`
- [ ] `GET /v1/tts/engine_schema?engine=cosyvoice_cross` a `supports_cross_lingual: true`
- [ ] Synthèse testée manuellement sur chaque mode (si GPU disponible)

### Phase 2 (Orpheus)
- [ ] `GET /v1/tts/models?engine=orpheus_tts` retourne les modèles
- [ ] Qualité FR testée manuellement avant confirmation

### Documentation
- [ ] `architecture.md` et `CONVENTIONS.md` mis à jour
- [ ] Decision DEC-008 enregistrée (CosyVoice 3 as priority backend)