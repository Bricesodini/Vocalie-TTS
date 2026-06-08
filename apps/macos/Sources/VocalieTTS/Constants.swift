// Path and runtime constants for the menu-bar app.
//
// We don't read these from the Python side (the Swift app is the
// driver, not the other way around). Defaults match the Python
// CLI's `vocalie_backend.config` so both ends agree.

import Foundation

enum Constants {
    /// Bundle identifier used in the .app Info.plist.
    static let bundleId = "com.vocalie.tts"

    /// Where the menu-bar app looks for the Python venv that contains
    /// the `vocalie-backend` entry point. Can be overridden at runtime
    /// via the VOCALIE_VENV environment variable (handy for staging
    /// builds).
    static let venvPath: String = {
        if let override = ProcessInfo.processInfo.environment["VOCALIE_VENV"] {
            return override
        }
        // Default: the project's .venv, three directories up from here.
        // apps/macos/Sources/VocalieTTS/Constants.swift -> repo root.
        let file = #filePath
        let root = file
            .components(separatedBy: "/")
            .prefix(while: { $0 != "apps" })
            .joined(separator: "/")
        return root + "/.venv"
    }()

    /// Absolute path to the `vocalie-backend` entry point inside the venv.
    static var vocalieBackendCLI: String { venvPath + "/bin/vocalie-backend" }

    /// URL the menu-bar app opens when the user clicks "Open UI".
    /// Matches VOCALIE_CORS_ORIGINS default in vocalie_backend/config.py.
    static let uiURL = URL(string: "http://localhost:3018")!

    /// How often the menu-bar app polls `vocalie-backend health` while
    /// the backend is running. Cheap (localhost HTTP) but we don't
    /// want to spam the log file.
    static let healthPollInterval: TimeInterval = 5.0
}
