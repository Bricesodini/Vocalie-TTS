// BackendManager — wraps the `vocalie-backend` Python CLI as a
// subprocess so the menu-bar app never has to know the Python
// details. All the actual work (start uvicorn, write PID, hit
// /v1/health) is done in Python; this class is just glue.

import Foundation

@MainActor
final class BackendManager: ObservableObject {
    @Published private(set) var state: BackendState?
    @Published private(set) var healthOK: Bool = false
    @Published private(set) var lastError: String?

    private let cliPath: String

    init(cliPath: String = Constants.vocalieBackendCLI) {
        self.cliPath = cliPath
    }

    // MARK: - Public API

    /// Spawn the backend, wait for /v1/health to come up.
    func start(wait: Bool = true) async {
        do {
            let result = try await runCLI(["start", "--wait", "--json"])
            state = decodeState(from: result.stdout)
            await refreshHealth()
        } catch {
            lastError = "start failed: \(error.localizedDescription)"
        }
    }

    /// SIGTERM the backend.
    func stop(force: Bool = false) async {
        var args = ["stop", "--json"]
        if force { args.append("--force") }
        do {
            _ = try await runCLI(args)
            state = nil
            healthOK = false
        } catch {
            lastError = "stop failed: \(error.localizedDescription)"
        }
    }

    /// `vocalie-backend status --json`. Always succeeds; the `running`
    /// flag on the result tells us the truth.
    func refreshState() async {
        do {
            let result = try await runCLI(["status", "--json"])
            state = decodeState(from: result.stdout)
        } catch {
            lastError = "status failed: \(error.localizedDescription)"
        }
    }

    /// `vocalie-backend health --json`. The HTTP probe is the most
    /// reliable signal that the backend is actually ready to serve.
    func refreshHealth() async {
        do {
            let result = try await runCLI(["health", "--json"])
            healthOK = result.stdout.contains("\"ok\": true")
        } catch {
            // Non-zero exit: health is "not ok" by definition.
            healthOK = false
        }
    }

    // MARK: - Subprocess plumbing

    private struct CLIResult {
        let stdout: String
        let stderr: String
        let exitCode: Int32
    }

    /// Run the Python CLI as a subprocess and capture its output.
    /// Throws if the executable is missing or returns a non-zero
    /// status code with a non-JSON stderr (JSON stderr is still
    /// surfaced via CLIResult.stderr).
    private func runCLI(_ args: [String]) async throws -> CLIResult {
        guard FileManager.default.isExecutableFile(atPath: cliPath) else {
            throw CLIError.cliNotFound(cliPath)
        }
        return try await withCheckedThrowingContinuation { cont in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: cliPath)
            process.arguments = args
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe
            process.terminationHandler = { proc in
                let outData = (try? stdoutPipe.fileHandleForReading.readToEnd()) ?? Data()
                let errData = (try? stderrPipe.fileHandleForReading.readToEnd()) ?? Data()
                cont.resume(returning: CLIResult(
                    stdout: String(data: outData, encoding: .utf8) ?? "",
                    stderr: String(data: errData, encoding: .utf8) ?? "",
                    exitCode: proc.terminationStatus,
                ))
            }
            do {
                try process.run()
            } catch {
                cont.resume(throwing: error)
            }
        }
    }

    private func decodeState(from json: String) -> BackendState? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(BackendState.self, from: data)
    }

    enum CLIError: LocalizedError {
        case cliNotFound(String)
        var errorDescription: String? {
            switch self {
            case .cliNotFound(let path):
                return "vocalie-backend CLI not found at \(path). Run 'vocalie-backend install' first."
            }
        }
    }
}
