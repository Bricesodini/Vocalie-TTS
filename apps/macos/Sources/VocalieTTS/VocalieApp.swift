// Top-level @main App. The whole app is a SwiftUI App with a main
// window (status + start/stop + embedded UI) and a Settings scene.
//
// We used to be a MenuBarExtra-only accessory app. The user has
// too many menu-bar icons already, so we now show up in the Dock
// and have a proper window. The UI is rendered with WKWebView so
// the Next.js frontend is embedded in the same window as the
// backend status — no browser tab juggling.

import SwiftUI
import AppKit

@main
struct VocalieApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var controller = AppController()

    var body: some Scene {
        // Main window: backend status header + embedded web UI.
        WindowGroup("Vocalie-TTS") {
            MainWindowView()
                .environmentObject(controller.manager)
                .environmentObject(controller.monitor)
                .environmentObject(controller)
                .frame(minWidth: 720, minHeight: 520)
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 960, height: 640)

        // Cmd-, settings window.
        Settings {
            SettingsView()
                .environmentObject(controller.manager)
                .frame(width: 420, height: 240)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Show up in the Dock (regular app). The user wants a real
        // window, not just a menu-bar icon.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Closing the window does NOT quit the app — the user may
        // want to keep the backend running headlessly. They use the
        // Dock "Quit" menu or Cmd-Q to actually exit.
        return false
    }
}
