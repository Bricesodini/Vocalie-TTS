// Codable model mirroring `vocalie_backend.process.BackendState`.
//
// The Python CLI emits this as JSON (status --json), so the Swift
// app doesn't need to call the HTTP API to know whether the
// backend is alive — it just decodes the JSON.

import Foundation

struct BackendState: Codable, Equatable {
    let running: Bool
    let pid: Int?
    let pidAlive: Bool
    let host: String
    let port: Int
    let startedAt: Double?
    let logFile: String
    let pidFile: String

    enum CodingKeys: String, CodingKey {
        case running
        case pid
        case pidAlive = "pid_alive"
        case host
        case port
        case startedAt = "started_at"
        case logFile = "log_file"
        case pidFile = "pid_file"
    }
}
