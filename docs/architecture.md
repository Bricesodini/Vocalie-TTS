# Architecture — Vocalie-TTS Models Engine

**Dernière mise à jour :** 2025-06-03  
**Scope :** `tts_backends/`, `backend/`, `frontend/src/`

---

## Vue d'ensemble

Vocalie-TTS utilise une architecture **backend auto-déclaratif** pour les moteurs TTS. Chaque moteur est une classe Python héritant de `TTSBackend` qui s'enregistre automatiquement dans le registre. Le catalogue d'engines, les endpoints API et l'UI frontend se construisent dynamiquement depuis les backends enregistrés — pas de liste manuelle, pas de conditions hardcodées.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐      │
│  │ Select engine │  │ Dynamic Fields*  │  │ Select model      │      │
│  │  (from /eng.) │  │ (from /schema)   │  │ (from /models)    │      │
│  └──────────────┘  └──────────────────┘  └────────────────────┘      │
│  * schema-driven via visible_if, pas de if/elif sur engine_id       │
└──────────────┬──────────────────────────────────────────────────────┘
               │ /v1/tts/engines · /tts/engine_schema · /tts/models
┌──────────────▼──────────────────────────────────────────────────────┐
│                    Routes FastAPI                                     │
│  routes/tts.py      → discovery + jobs (polymorphique)               │
│  routes/backends.py  → install / uninstall                           │
└──────────────┬──────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────────┐
│              Backend Registry (tts_backends/)                         │
│                                                                      │
│  ┌── TTSBackend (ABC) ──────────────────────────────────────────┐    │
│  │  _REGISTRY: auto-populated via __init_subclass__             │    │
│  │  engine_variants()  → déclare les engine_ids du catalogue    │    │
│  │  list_models()     → déclare les modèles HF disponibles     │    │
│  │  params_schema()   → déclare les params dynamiques (UI)      │    │
│  │  resolve_engine_params() → injecte les defaults par engine  │    │
│  │  capabilities()    → déclare les capacités (ref, voice, …)   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                               │                                      │
│  ┌── SubprocessBackendMixin ───┤── Pour les backends en venv isolé     │
│  │  _run_subprocess()          │                                      │
│  │  _run_subprocess_chunk()    │                                      │
│  └────────────────────────────┘                                      │
│                               │                                      │
│  ┌─── ChatterboxBackend ──┐  ┌─── Qwen3Backend ────────┐  ┌─ … ─┐   │
│  │ runner_module=…_runner │  │ runner_module=…_runner   │  │     │   │
│  │ runner_venv=chatterbox │  │ runner_venv=qwen3       │  │     │   │
│  └────────────────────────┘  └─────────────────────────┘  └─────┘   │
│                               │                                      │
│  ┌── BaseSubprocessRunner ────┘  (dans le venv)                      │
│  │  run_synthesis() → logique spécifique au modèle                  │
│  │  read_payload() / write_response() / write_error() / log()       │
│  └──────────────────────────────────────────────────────────────────┘
│                                                                      │
│  ┌── Catalog (tts_backends/catalog.py) ─────────────────────────┐    │
│  │  rebuild_engine_catalog() ← itère sur _REGISTRY               │    │
│  │  get_engine_catalog() → liste des engines                     │    │
│  │  ENGINE_ALIAS_MAP → compatibilité legacy                      │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────────┐
│              Venvs (.venvs/chatterbox, .venvs/qwen3, …)              │
│  Chaque backend tourne dans son propre venv (dépendances isolées)   │
│  backend_install/manifests.py → pip packages + install hooks       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Ajouter un nouveau backend : la checklist

Un nouveau backend ne touche que **2-3 fichiers** :

### 1. Créer le backend (`tts_backends/mon_modele_backend.py`)

```python
from tts_backends.base import TTSBackend, ModelInfo, ParamSpec, BackendUnavailableError
from tts_backends.base_runner import SubprocessBackendMixin

class MonModeleBackend(TTSBackend, SubprocessBackendMixin):
    # ── Identité ──
    id = "mon_modele"
    display_name = "Mon Modèle TTS"
    runner_module = "mon_modele_runner"    # tts_backends/mon_modele_runner.py
    runner_venv = "mon_modele"            # .venvs/mon_modele/
    default_timeout = 300.0

    # ── Options du backend ──
    supports_ref_audio = False
    supports_inter_chunk_gap = True

    # ── Variants d'engine (ex: mon_modele_v2) ──
    @classmethod
    def engine_variants(cls):
        return [{"id": "mon_modele", "label": "Mon Modèle"}]

    # ── Disponibilité ──
    @classmethod
    def is_available(cls) -> bool:
        from backend_install.status import backend_status
        return backend_status("mon_modele").get("installed", False)

    # ── Modèles ──
    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="org/mon-modele-v1", label="Mon Modèle v1")]

    # ── Paramètres dynamiques (UI schema-driven) ──
    def params_schema(self) -> dict[str, ParamSpec]:
        return {
            "temperature": ParamSpec(key="temperature", type="float", default=0.7,
                                    min=0.0, max=1.0, step=0.05, label="Température"),
        }

    # ── Synthèse ──
    def synthesize_chunk(self, text, *, voice_ref_path=None, lang=None, **params):
        return self._run_subprocess_chunk(text, payload_suffix=params, lang=lang)
```

### 2. Créer le runner (`tts_backends/mon_modele_runner.py`)

```python
#!/usr/bin/env python3
from tts_backends.base_runner import BaseSubprocessRunner

class MonModeleRunner(BaseSubprocessRunner):
    def run_synthesis(self, payload: dict) -> dict:
        # Charger modèle, synthétiser, écrire WAV
        # Return dict avec "ok": True
        ...

if __name__ == "__main__":
    raise SystemExit(MonModeleRunner.main())
```

### 3. Ajouter 1 import dans `tts_backends/__init__.py`

```python
from .mon_modele_backend import MonModeleBackend  # noqa: F401
```

### 4. [Optionnel] Ajouter le manifest d'installation (`backend_install/manifests.py`)

```python
"mon_modele": BackendManifest(
    engine_id="mon_modele",
    python="python3.11",
    pip_packages=["mon-modele-tts", "torch"],
    system_hints=["ffmpeg (optionnel)"],
    import_probes=["mon_modele"],
    post_install_checks=[["-c", "import mon_modele; print('OK')"]],
)
```

**C'est tout.** Le registre, le catalogue, les endpoints `/tts/engines`, `/tts/models`, `/tts/engine_schema`, et l'UI se mettent à jour automatiquement.

---

## Protocole de communication subprocess

### Parent → Runner (stdin)

Le parent sérialise un dict JSON sur le stdin du runner :

```json
{
  "text": "Bonjour le monde",
  "out_path": "/tmp/mon_modele_abc.wav",
  "mode": "custom_voice",
  "model_id": "org/mon-modele-v1",
  "language": "French",
  "speaker": "Alice",
  "params": {"device": "auto", "dtype": "float16"}
}
```

### Runner → Parent (stdout)

**Succès :**
```json
{"ok": true, "sample_rate": 24000, "duration_ms": 1200, "mode": "custom_voice"}
```

**Échec :**
```json
{"ok": false, "error": "model_load_failed", "detail": "CUDA out of memory", "trace": "..."}
```

Le protocole complet (lecture stdin, écriture stdout, gestion timeout, JSON fallback) est centralisé dans `BaseSubprocessRunner` et `SubprocessBackendMixin`.

---

## Auto-registration

Le mécanisme d'auto-registration utilise `__init_subclass__` :

```python
class TTSBackend(ABC):
    _REGISTRY: Dict[str, type[TTSBackend]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "id", None) and not getattr(cls, "__abstractmethods__", None):
            TTSBackend._REGISTRY[cls.id] = cls
```

Dès qu'un module backend est importé, sa classe s'enregistre. L'import est déclenché par `tts_backends/__init__.py`.

---

## Catalogue dynamique

Le catalogue d'engines se construit à partir du registre :

```python
def rebuild_engine_catalog() -> None:
    """Itère sur TTSBackend._REGISTRY et appelle cls.engine_variants()."""
    for cls in TTSBackend._REGISTRY.values():
        for variant in cls.engine_variants():
            catalog.append({
                "id": variant["id"],
                "label": variant.get("label", cls.display_name),
                "backend_id": cls.id,
            })
```

Appelé une fois dans `tts_backends/__init__.py` après les imports des backends.

---

## Capabilities & UI schema-driven

Le frontend ne contient **aucune condition hardcodée** sur les engine_ids. Tout est piloté par l'API :

| Source | Utilisation frontend |
|--------|---------------------|
| `capabilities.supports_voice_reference` | Affiche/masque le sélecteur de voix |
| `capabilities.supports_voice_design` | Affiche/masque les champs VoiceDesign |
| `capabilities.auto_resolved_keys` | Masque les params auto-résolus (ex: `qwen3_mode`) |
| `capabilities.can_refresh_speakers` | Bouton refresh speakers |
| `params_schema[].visible_if` | Visibilité conditionnelle des champs |
| `/tts/models` | Sélecteur de modèle |

---

## Isolation par venv

Chaque backend tourne dans son propre venv Python (`backend_install/`) :

- **Chatterbox** : `.venvs/chatterbox/` → `chatterbox`, numpy<1.26
- **Qwen3** : `.venvs/qwen3/` → `qwen_tts`, torch

Le runner est invoqué via `subprocess.run([python_path, runner_path], input=json, …)`.

Les manifests décrivent l'installation : packages pip, import probes, hooks de pré/post-install, prefetch de poids.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/tts/engines` | Liste les engines avec disponibilité et supports_ref |
| `GET /v1/tts/models?engine=` | Liste les modèles disponibles pour un engine |
| `GET /v1/tts/engine_schema?engine=` | Schema des params dynamiques + capabilities |
| `GET /v1/tts/voices?engine=` | Liste les voix de référence |
| `POST /v1/tts/jobs` | Crée un job de synthèse (résout params via backend) |
| `POST /v1/backends/{id}/install` | Installe un backend |
| `DELETE /v1/backends/{id}/uninstall` | Désinstalle un backend |

---

## Sources de vérité

| Concept | Source unique | Conséquent |
|---------|-------------|-----------|
| `supports_ref` | `backend.supports_ref_for_engine(engine_id)` | N'est PAS dans le catalogue |
| Mode par défaut | `backend.resolve_engine_params(engine_id, params)` | N'est PAS dans les routes |
| Modèles disponibles | `backend.list_models()` | N'est PAS hardcodé |
| Capabilities UI | `backend.capabilities(engine_id)` | N'est PAS dans le frontend |
| Langues supportées | `catalog.CHATTERBOX_LANGUAGE_MAP` / `QWEN3_LANGUAGE_MAP` | N'est PAS dupliqué par backend |
| Modèles HF | `QWEN3_DEFAULT_MODELS` dans `qwen3_backend.py` | `qwen3_prefetch.py` importe |

---

## Décisions d'architecture

1. **Pas de if/elif sur engine_id** dans les routes ou services → tout passe par le backend polymorphique
2. **`supports_ref` toujours résolu depuis le backend** → pas de dual-source-of-truth
3. **Frontend 100% schema-driven** → `visible_if` et `capabilities` contrôlent l'UI
4. **Venv isolation** → chaque backend ases propres dépendances Python
5. **BaseSubprocessRunner centralise le protocole** → chaque runner n'implémente que `run_synthesis()`
6. **Catalogue dynamique** → se reconstruit depuis les backends enregistrés