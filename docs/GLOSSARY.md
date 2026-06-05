# Glossary

- Engine ID: canonical engine identifier exposed by `/v1/tts/engines` (source: `backend/routes/tts.py`).
- Legacy engine alias: historical engine naming used by compatibility mappings (source: `tts_backends/catalog.py`, `backend/services/preset_service.py`).
- Preset: persisted generation configuration (sources: `backend/schemas/models.py`, `backend/services/preset_service.py`).
- UIState: canonical API preset/state model shape (source: `backend/schemas/models.py`).
- Direction chunking: explicit marker/range-based segmentation before synthesis (sources: `backend/routes/chunks.py`, `text_tools.py`).
- Asset ID: identifier for generated/edit/enhanced audio metadata (source: `backend/services/asset_service.py`).
- Job store: in-memory asynchronous job lifecycle manager (source: `backend/services/job_service.py`).
- Scope freeze: explicit in-scope/out-of-scope perimeter for audits (source: `README.md`).
- Non-goals: explicit exclusions from product scope (source: `README.md`).
- Compatibility layer: legacy translation/adaptation path kept for backward compatibility.
