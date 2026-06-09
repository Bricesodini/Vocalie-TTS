// AppController — replaces the old MenuBarController. Holds the
// BackendManager and HealthMonitor; the SwiftUI views read
// `manager.state` / `monitor.isHealthy` via @EnvironmentObject.

import SwiftUI
import AppKit

@MainActor
final class AppController: ObservableObject {
    let manager: BackendManager
    let monitor: HealthMonitor

    /// Set to true while a Start/Stop/Restart is in flight. Used by
    /// the buttons to show a spinner and to disable themselves so
    /// the user can't double-click. The previous menu-bar version
    /// had no visible feedback during the 14s cold-start wait,
    /// which made it look like the click was ignored.
    @Published var actionInFlight: Bool = false

    init() {
        let m = BackendManager()
        self.manager = m
        self.monitor = HealthMonitor(manager: m)
    }

    // MARK: - User actions

    func start() async {
        actionInFlight = true
        defer { actionInFlight = false }
        monitor.start()
        await manager.start(wait: true)
    }

    func stop() async {
        actionInFlight = true
        defer { actionInFlight = false }
        await manager.stop()
        monitor.stop()
    }

    func restart() async {
        actionInFlight = true
        defer { actionInFlight = false }
        await manager.stop()
        monitor.start()
        await manager.start(wait: true)
    }

    /// Open the Next.js UI in the system browser. The WKWebView
    /// inside the main window does the same thing — this is the
    /// fallback for users who'd rather pop out the UI to a
    /// dedicated browser tab (e.g. to share with another screen).
    func openInBrowser() {
        guard let url = URL(string: "http://\(manager.state?.host ?? "127.0.0.1"):\(manager.state?.port ?? 8018)") else { return }
        NSWorkspace.shared.open(url)
    }
}

// MARK: - Settings window content

struct SettingsView: View {
    @EnvironmentObject var manager: BackendManager
    var body: some View {
        Form {
            LabeledContent("Venv path") {
                Text(Constants.venvPath)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
            LabeledContent("CLI") {
                Text(Constants.vocalieBackendCLI)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
            LabeledContent("State") {
                Text(manager.state?.running == true ? "running" : "stopped")
            }
            if let err = manager.lastError {
                Text(err).foregroundStyle(.red).font(.caption)
            }
        }
        .padding()
    }
}
