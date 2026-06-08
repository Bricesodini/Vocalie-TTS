# Vocalie-TTS — macOS Menu Bar App

A lightweight SwiftUI app that lives in the macOS menu bar and
controls the Vocalie-TTS Python backend.

The app is intentionally a thin shell: all the real work (start
uvicorn, hit `/v1/health`, write PID files) is done by the
`vocalie-backend` Python CLI. The Swift app just calls that CLI
as a subprocess and reflects its JSON output in a menu-bar icon.

## Architecture

```
+-----------------------------+
| Vocalie-TTS.app             |    SwiftUI MenuBarExtra
|   BackendManager.swift  <---+---- subprocess ----+
|   HealthMonitor.swift   <---+                    v
|   MenuBarController.swift    |       +-------------------------+
|                             |       | vocalie-backend CLI     |
+-----------------------------+       |   start / stop / status |
                                      |   install / doctor      |
                                      +-----------+-------------+
                                                  |
                                                  v
                                      +-------------------------+
                                      | uvicorn (Python)        |
                                      | backend.app:app         |
                                      | http://127.0.0.1:8018   |
                                      +-------------------------+
```

## Build

```bash
cd apps/macos
./Scripts/build-app.sh    # release build → build/Vocalie-TTS.app
open build/Vocalie-TTS.app
```

The build script:
1. Runs `swift build -c release`
2. Copies the binary into `Vocalie-TTS.app/Contents/MacOS/`
3. Writes a minimal `Info.plist` with `LSUIElement=true` so the
   app stays out of the Dock

No Xcode project, no signing, no notarization — the bundle is a
"developer" build. To distribute to other Macs you'll need to
add code signing (see Future work below).

## Run from source (debug)

```bash
cd apps/macos
swift run                 # debug build, console attached
```

The app shows up in the menu bar as a speaker icon. Click it to:
- see the current backend state
- start / stop / restart the backend
- open the UI in your browser (`http://localhost:3018`)
- open the Settings window

## Prerequisites

- macOS 13+ (uses `MenuBarExtra`)
- A working `vocalie-backend` CLI on the host (see main project README)
  - The app looks for `.venv/bin/vocalie-backend` by default
  - Override with the `VOCALIE_VENV` environment variable

## Auto-start at login (optional)

```bash
cp Scripts/com.vocalie.tts.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.vocalie.tts.plist
```

This registers the app as a user LaunchAgent. It starts when you
log in and quits cleanly when you log out.

## Project structure

```
apps/macos/
├── Package.swift                       Swift Package manifest
├── README.md                           this file
├── Sources/VocalieTTS/
│   ├── VocalieApp.swift                @main entry, NSApp setup
│   ├── MenuBarController.swift         MenuBarExtra + Settings scenes
│   ├── BackendManager.swift            subprocess wrapper around vocalie-backend
│   ├── BackendState.swift              Codable mirror of Python BackendState
│   ├── HealthMonitor.swift             timer-driven /v1/health poll
│   └── Constants.swift                 paths, ports, poll interval
├── Scripts/
│   ├── build-app.sh                    release → .app bundle
│   └── com.vocalie.tts.plist           LaunchAgent template
└── build/                              output (gitignored)
    └── Vocalie-TTS.app
```

## Future work (out of scope for v0.1)

- **Code signing + notarization** so the .app can be distributed
  without "unidentified developer" warnings.
- **DMG packaging** for friendlier distribution.
- **Sparkle auto-update** to push new versions without re-download.
- **Per-engine status** (currently the menu shows only aggregate
  backend state; the Python CLI already knows per-engine availability
  via `vocalie_backend.status.backend_status`).
- **Voice ref management UI** in the Settings window.
