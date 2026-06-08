// swift-tools-version: 5.9
// Package manifest for the VocalieTTS menu-bar app.
//
// We deliberately keep the entire app as a single executable target —
// the app is a thin shell around the Python backend CLI, so a heavy
// module graph would be over-engineered.

import PackageDescription

let package = Package(
    name: "VocalieTTS",
    platforms: [
        .macOS(.v13)  // MenuBarExtra requires macOS 13+
    ],
    products: [
        .executable(name: "vocalie-tts", targets: ["VocalieTTS"]),
    ],
    targets: [
        .executableTarget(
            name: "VocalieTTS",
            path: "Sources/VocalieTTS",
        ),
    ],
)
