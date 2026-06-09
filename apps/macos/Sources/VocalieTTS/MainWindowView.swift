// MainWindowView — the whole point of the app. A single window
// with two halves:
//
//   ┌─────────────────────────────┐
//   │ ◉ Vocalie-TTS   ▶ Start  ⏏  │  status header
//   │   pid 73246 · 127.0.0.1:8018│
//   │   chatterbox ✓ qwen3 ✓      │
//   ├─────────────────────────────┤
//   │                             │
//   │   WKWebView rendering the   │  embedded Next.js UI
//   │   Next.js frontend from     │
//   │   http://127.0.0.1:8018     │
//   │                             │
//   └─────────────────────────────┘

import SwiftUI
import WebKit

struct MainWindowView: View {
    @EnvironmentObject var manager: BackendManager
    @EnvironmentObject var monitor: HealthMonitor
    @EnvironmentObject var controller: AppController

    var body: some View {
        VStack(spacing: 0) {
            StatusHeader()
            Divider()
            EmbeddedUI()
        }
    }
}

// MARK: - Status header

private struct StatusHeader: View {
    @EnvironmentObject var manager: BackendManager
    @EnvironmentObject var monitor: HealthMonitor
    @EnvironmentObject var controller: AppController

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            statusDot
                .frame(width: 12, height: 12)
            VStack(alignment: .leading, spacing: 2) {
                Text(headerTitle)
                    .font(.headline)
                if let s = manager.state, s.running {
                    Text(headerSubtitle(state: s))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if controller.actionInFlight {
                    Text("Starting…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Backend is stopped")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let err = manager.lastError {
                    Text(err)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }
            Spacer()
            actionButtons
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var statusDot: some View {
        Circle()
            .fill(dotColor)
            .overlay(Circle().stroke(dotColor.opacity(0.3), lineWidth: 4))
    }

    private var dotColor: Color {
        if !controller.actionInFlight && (manager.state?.running ?? false) == false {
            return .gray
        }
        if controller.actionInFlight {
            return .yellow
        }
        return monitor.isHealthy ? .green : .orange
    }

    private var headerTitle: String {
        switch (manager.state?.running ?? false, monitor.isHealthy) {
        case (false, _): return "Vocalie-TTS"
        case (true, true): return "Backend running"
        case (true, false): return "Backend starting…"
        }
    }

    private func headerSubtitle(state: BackendState) -> String {
        let pid = state.pid.map { "pid \($0)" } ?? ""
        let url = "http://\(state.host):\(state.port)"
        return [pid, url].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    @ViewBuilder
    private var actionButtons: some View {
        HStack(spacing: 8) {
            if controller.actionInFlight {
                ProgressView()
                    .controlSize(.small)
                    .padding(.trailing, 4)
            }
            if manager.state?.running == true {
                Button("Stop") {
                    Task { await controller.stop() }
                }
                .disabled(controller.actionInFlight)
                .keyboardShortcut("s", modifiers: .command)
                Button("Restart") {
                    Task { await controller.restart() }
                }
                .disabled(controller.actionInFlight)
            } else {
                Button("Start Backend") {
                    Task { await controller.start() }
                }
                .disabled(controller.actionInFlight)
                .keyboardShortcut("r", modifiers: .command)
                .buttonStyle(.borderedProminent)
            }
            Button {
                controller.openInBrowser()
            } label: {
                Label("Open in Browser", systemImage: "arrow.up.right.square")
            }
            .disabled(!(manager.state?.running ?? false))
            .help("Open the UI in your system browser")
        }
    }
}

// MARK: - Embedded WebView

private struct EmbeddedUI: View {
    @EnvironmentObject var manager: BackendManager
    @EnvironmentObject var monitor: HealthMonitor

    var body: some View {
        Group {
            if let s = manager.state, s.running, monitor.isHealthy {
                VocalieWebView(url: uiURL(state: s))
            } else {
                placeholder
            }
        }
    }

    private func uiURL(state: BackendState) -> URL {
        // We hit the same /v1/info or / route as the browser would.
        // The frontend reads /v1/* via its own proxy, so the
        // WebView just needs to load the root path.
        URL(string: "http://\(state.host):\(state.port)/")!
    }

    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: manager.state?.running == true ? "hourglass" : "speaker.slash")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text(manager.state?.running == true ? "Waiting for backend to become healthy…" : "Backend is stopped")
                .foregroundStyle(.secondary)
            if manager.state?.running != true {
                Text("Click ▶ Start Backend above.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - WKWebView wrapper

struct VocalieWebView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        // The backend serves HTTP on localhost. WKWebView needs
        // no extra config for that — Apple's ATS only blocks HTTPS
        // *upgrade* on non-localhost origins.
        let view = WKWebView(frame: .zero, configuration: config)
        view.allowsBackForwardNavigationGestures = true
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        // Only reload if the URL actually changed (state.host /
        // state.port may shift between calls). Avoid reloading on
        // every redraw or we'd thrash the page.
        if nsView.url != url {
            nsView.load(URLRequest(url: url))
        }
    }
}
