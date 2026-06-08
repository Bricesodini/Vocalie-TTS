// Top-level @main App. The whole menu-bar app is a SwiftUI App
// whose only scenes are the menu-bar extra and the settings
// window. No main window — `LSUIElement`-equivalent behaviour is
// achieved by setting `setActivationPolicy(.accessory)` in
// `AppDelegate`.

import SwiftUI
import AppKit

@main
struct VocalieApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var controller = MenuBarController()

    var body: some Scene {
        controller.menuBarScene()
        controller.settingsScene()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from the Dock — we only live in the menu bar.
        NSApp.setActivationPolicy(.accessory)
    }
}
