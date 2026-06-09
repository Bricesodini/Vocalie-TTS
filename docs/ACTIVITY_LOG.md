# Activity Log

## 2026-06-09 — Fix Chatterbox freeze
- Fixed hardcoded `/app/.assets/chatterbox` in `chatterbox_runner.py` → local repo path.
- Increased Chatterbox backend timeout 180 s → 600 s.
- Added runner stderr progress logs.
- Killed orphaned runner process consuming 100% CPU.
- See `docs/runs/2026-06-09_fix-chatterbox-freeze/05_PATCH_SUMMARY.md`.
