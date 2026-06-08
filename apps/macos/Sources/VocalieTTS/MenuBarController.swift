// MenuBarController — the SwiftUI scene that defines the menu-bar
// icon and its dropdown menu. We use the modern `MenuBarExtra` API
// (macOS 13+) instead of NSStatusItem so the whole UI is SwiftUI.

import SwiftUI
import AppKit

@MainActor
final class MenuBarController: ObservableObject {
    let manager: BackendManager
    let monitor: HealthMonitor

    init() {
        let m = BackendManager()
        self.manager = m
        self.monitor = HealthMonitor(manager: m)
    }

    // MARK: - SwiftUI scenes

    func menuBarScene() -> some Scene {
        MenuBarExtra {
            MenuContent(controller: self)
                .environmentObject(manager)
                .environmentObject(monitor)
        } label: {
            // The icon switches between the three states so the user
            // can tell at a glance what the backend is doing.
            MenuBarIcon(state: iconState())
        }
        .menuBarExtraStyle(.menu)
    }

    func settingsScene() -> some Scene {
        Settings {
            SettingsView()
                .environmentObject(manager)
                .frame(width: 420, height: 240)
        }
    }

    // MARK: - State helpers

    enum IconState { case stopped, starting, healthy, unhealthy }

    private func iconState() -> IconState {
        switch (manager.state?.running ?? false, manager.healthOK) {
        case (false, _): return .stopped
        case (true, true): return .healthy
        case (true, false): return .unhealthy
        }
    }

    func start() async {
        monitor.start()
        await manager.start(wait: true)
    }

    func stop() async {
        await manager.stop()
        monitor.stop()
    }

    func openUI() {
        // Force the system browser to open. If the backend isn't up
        // the user will get a connection refused, which is the
        // correct behaviour.
        NSWorkspace.shared.open(Constants.uiURL)
    }
}

// MARK: - Icon

private struct MenuBarIcon: View {
    let state: MenuBarController.IconState
    var body: some View {
        switch state {
        case .stopped:
            Image(systemName: "speaker.slash")
        case .starting:
            Image(systemName: "speaker.wave.1")
        case .healthy:
            Image(systemName: "speaker.wave.2.fill")
                .foregroundStyle(.green)
        case .unhealthy:
            Image(systemName: "speaker.wave.1")
                .foregroundStyle(.orange)
        }
    }
}

// MARK: - Menu content

private struct MenuContent: View {
    @ObservedObject var controller: MenuBarController
    @EnvironmentObject var manager: BackendManager
    @EnvironmentObject var monitor: HealthMonitor

    var body: some View {
        if let s = manager.state, s.running {
            runningSection(state: s)
        } else {
            stoppedSection
        }
        Divider()
        Button("Open UI in Browser") { controller.openUI() }
            .disabled(!(manager.state?.running ?? false))
        Divider()
        Button("Settings…") {
            // macOS opens the Settings scene declared in MenuBarController.
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        }
        .keyboardShortcut(",", modifiers: .command)
        Divider()
        Button("Quit Vocalie-TTS") { NSApp.terminate(nil) }
            .keyboardShortcut("q", modifiers: .command)
    }

    private func runningSection(state: BackendState) -> some View {
        Group {
            Text("Running")
            if let pid = state.pid {
                Text("pid \(pid)").font(.caption).foregroundStyle(.secondary)
            }
            Text("http://\(state.host):\(state.port)").font(.caption).foregroundStyle(.secondary)
            Divider()
            Button("Restart") {
                Task { await controller.stop(); await controller.start() }
            }
            Button("Stop") { Task { await controller.stop() } }
                .keyboardShortcut("s", modifiers: .command)
        }
    }

    private var stoppedSection: some View {
        Group {
            Text("Backend is stopped")
            Button("Start Backend") {
                Task { await controller.start() }
            }
            .keyboardShortcut("r", modifiers: .command)
        }
    }
}

// MARK: - Settings window

private struct SettingsView: View {
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
