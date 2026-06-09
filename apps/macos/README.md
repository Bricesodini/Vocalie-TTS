# Vocalie-TTS — macOS App

A lightweight SwiftUI app that controls the Vocalie-TTS Python
backend and embeds the Next.js UI in the same window.

The app is intentionally a thin shell: all the real work (start
uvicorn, hit `/v1/health`, write PID files) is done by the
`vocalie-backend` Python CLI. The Swift app just calls that CLI
as a subprocess, decodes the JSON state, and renders the UI in
a `WKWebView` against the backend's HTTP root.

The user used to want a MenuBarExtra-only accessory app, but
asked for a regular window with a Dock icon because they had
too many menu-bar icons already. So the new shape is:

```
+-----------------------------------+
| Vocalie-TTS.app                   |    SwiftUI WindowGroup
|   BackendManager.swift  <---------+---- subprocess ----+
|   HealthMonitor.swift  <----------+                    v
|   AppController.swift             |        +-------------------------+
|   MainWindowView.swift  ----------+        | vocalie-backend CLI     |
|     (StatusHeader + WKWebView)    |        |   start / stop / status |
+-----------------------------------+        |   install / doctor      |
        |                                   +-----------+-------------+
        v                                               |
   http://127.0.0.1:8018                                v
        |                                   +-------------------------+
        +-------> uvicorn (Python) <-------+ backend.app:app         |
                          |                 | http://127.0.0.1:8018   |
                          +-> serves UI     +-------------------------+
```

## Build

```bash
cd apps/macos
./Scripts/build-app.sh
```

Produces two artifacts in `apps/macos/build/`:
- `Vocalie-TTS.app` — ad-hoc signed bundle
- `Vocalie-TTS-0.1.0-arm64.dmg` — drag-to-Applications installer

The build script:
1. `swift build -c release`
2. Wraps the binary in `Vocalie-TTS.app/Contents/MacOS/` with a
   minimal `Info.plist` (`LSUIElement=true` so the app stays out
   of the Dock)
3. Ad-hoc code-signs the bundle (`codesign --sign -`) so it
   passes `codesign --verify` and Gatekeeper doesn't block it on
   the developer's own Mac
4. Builds a compressed read-only DMG with `hdiutil` and a
   `/Applications` symlink for drag-install

No Xcode project, no notarization — the bundle is a "developer"
build suitable for personal use and for sharing with trusted
machines. To distribute to other Macs without the
"unidentified developer" warning, see [Notarization](#notarization-for-distribution) below.

## Run from source (debug)

```bash
cd apps/macos
swift run                 # debug build, console attached
```

A window opens with the status header on top (state dot, PID,
port, Start/Stop/Restart buttons) and the Next.js UI rendered in
a `WKWebView` below. Closing the window does NOT quit the app
(it stays alive in the Dock); use Cmd-Q to actually exit.

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
│   ├── AppController.swift             state + start/stop/restart
│   ├── MainWindowView.swift            status header + embedded WKWebView
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

## Notarization for distribution

Ad-hoc signing works for the developer but other Macs will see
"unidentified developer" and refuse to launch. To distribute
properly, you need an Apple Developer ID ($99/year) and
notarization. The full flow:

```bash
# 1. Build as usual
./Scripts/build-app.sh

# 2. Replace the ad-hoc sign with your real identity
codesign --force --deep \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    build/Vocalie-TTS.app

# 3. Zip the .app (notarization needs a stable container)
ditto -c -k --sequesterRsrc --keepParent \
    build/Vocalie-TTS.app \
    build/Vocalie-TTS.zip

# 4. Submit to Apple for notarization
xcrun notarytool submit build/Vocalie-TTS.zip \
    --apple-id "you@example.com" \
    --team-id "TEAMID" \
    --password "app-specific-password" \
    --wait

# 5. Staple the notarization ticket to the .app
xcrun stapler staple build/Vocalie-TTS.app

# 6. Rebuild the DMG so the stapled ticket is inside it
hdiutil create -ov -format UDZO -fs HFS+ \
    -srcfolder build/dmg-stage \
    -volname "Vocalie-TTS" \
    build/Vocalie-TTS-0.1.0-arm64.dmg
```

The current `build-app.sh` only does step 1 (ad-hoc). To
automate the rest, set `CODESIGN_IDENTITY` and the credentials
in your environment and wrap steps 2–6 in a second script.

## Future work (out of scope for v0.1)

- **Code signing + notarization** so the .app can be distributed
  without "unidentified developer" warnings.
- **Sparkle auto-update** to push new versions without re-download.
- **Per-engine status** (currently the menu shows only aggregate
  backend state; the Python CLI already knows per-engine availability
  via `vocalie_backend.status.backend_status`).
- **Voice ref management UI** in the Settings window.
