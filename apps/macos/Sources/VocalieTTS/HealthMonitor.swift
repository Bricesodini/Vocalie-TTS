// Polls the backend health endpoint on a fixed interval. The
// menu-bar icon reflects the latest state — green when healthy,
// amber when started but not yet healthy, grey when stopped.

import Foundation
import Combine

@MainActor
final class HealthMonitor: ObservableObject {
    @Published private(set) var isHealthy: Bool = false
    @Published private(set) var lastChecked: Date?

    private let manager: BackendManager
    private var timer: Timer?

    init(manager: BackendManager) {
        self.manager = manager
    }

    func start() {
        stop()
        // Fire immediately, then every Constants.healthPollInterval.
        Task { await tick() }
        timer = Timer.scheduledTimer(withTimeInterval: Constants.healthPollInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.tick() }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        isHealthy = false
        lastChecked = nil
    }

    private func tick() async {
        await manager.refreshState()
        await manager.refreshHealth()
        isHealthy = manager.healthOK
        lastChecked = Date()
    }
}
