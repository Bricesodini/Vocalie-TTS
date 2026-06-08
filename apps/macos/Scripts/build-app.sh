#!/usr/bin/env bash
#
# Build a release .app bundle for the Vocalie-TTS menu-bar app, then
# ad-hoc code-sign it and wrap it in a DMG for distribution.
#
# Output:
#   build/Vocalie-TTS.app                     signed, ready to open
#   build/Vocalie-TTS-0.1.0-arm64.dmg         drag-to-Applications installer
#
# The .app expects a `vocalie-backend` CLI to be installed in
# $ROOT_DIR/.venv (or wherever VOCALIE_VENV points). It's NOT bundled
# inside the .app — that keeps the bundle tiny and lets the user
# upgrade the backend independently of the app.
#
# Ad-hoc signing is used (no Apple Developer ID required). It satisfies
# Gatekeeper for the developer's own machine; for distribution to other
# Macs without the "unidentified developer" warning, you'll need to
# replace the `codesign --sign -` call below with your real identity
# (and add a `xcrun notarytool submit` step + `xcrun stapler staple`).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="release"
BIN_NAME="vocalie-tts"
APP_NAME="Vocalie-TTS"
BUNDLE_ID="com.vocalie.tts"
VERSION="0.1.0"
BUILD_DIR="$ROOT_DIR/build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"
DMG_PATH="$BUILD_DIR/${APP_NAME}-${VERSION}-arm64.dmg"
STAGE_DIR="$BUILD_DIR/dmg-stage"

# ---------------------------------------------------------------------------
# 1. swift build
# ---------------------------------------------------------------------------
echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

BIN_PATH="$(swift build -c "$CONFIG" --show-bin-path)/$BIN_NAME"
if [[ ! -x "$BIN_PATH" ]]; then
    echo "error: built binary not found at $BIN_PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. package .app
# ---------------------------------------------------------------------------
echo "==> packaging $APP_BUNDLE"
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$BIN_PATH" "$MACOS_DIR/$APP_NAME"
chmod +x "$MACOS_DIR/$APP_NAME"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>             <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>      <string>Vocalie-TTS</string>
    <key>CFBundleIdentifier</key>       <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>          <string>1</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>CFBundleExecutable</key>       <string>$APP_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key>   <string>13.0</string>
    <key>LSUIElement</key>              <true/>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>NSAppleScriptEnabled</key>     <false/>
</dict>
</plist>
PLIST

cat > "$CONTENTS/PkgInfo" <<<"APPL????"

# ---------------------------------------------------------------------------
# 3. ad-hoc code-sign
# ---------------------------------------------------------------------------
# Ad-hoc signing replaces the linker-signed hash with a real code
# signature. That makes codesign --verify pass and stops Gatekeeper
# from blocking the .app on the developer's own Mac. For real
# distribution you'd swap "-" for your Apple Developer ID identity.
echo "==> ad-hoc codesign"
codesign --force --deep --sign - "$APP_BUNDLE"
codesign --verify --verbose "$APP_BUNDLE"

# ---------------------------------------------------------------------------
# 4. DMG packaging
# ---------------------------------------------------------------------------
echo "==> building DMG"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp -R "$APP_BUNDLE" "$STAGE_DIR/"
ln -s /Applications "$STAGE_DIR/Applications"

hdiutil create -ov -format UDZO -fs HFS+ \
    -srcfolder "$STAGE_DIR" \
    -volname "$APP_NAME" \
    "$DMG_PATH"

rm -rf "$STAGE_DIR"

echo
echo "Built:"
echo "  $APP_BUNDLE"
echo "  $DMG_PATH"
echo
echo "Install:"
echo "  open $DMG_PATH && drag Vocalie-TTS.app to /Applications"
echo "  (or) open $APP_BUNDLE      # developer install, no copy"
