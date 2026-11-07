import Foundation
import os
import ServiceManagement

/// Opening the app at login, as an item in System Settings › General › Login Items. The system's record is the
/// setting: `refresh()` reads it, so an item turned off in System Settings shows as off here.
@MainActor @Observable
final class LoginItem {
    enum State: Equatable {
        case off
        case on
        /// Registered, but switched off in System Settings › General › Login Items.
        case needsApproval
        /// The app is running from the DMG, or from the temporary read-only copy macOS runs a downloaded app
        /// from until it's moved (App Translocation). A login item would point at a copy that goes away.
        case temporaryCopy
    }

    private(set) var state: State
    /// Why the last change failed, as the system put it.
    private(set) var failure: String?
    @ObservationIgnored let fixed: Bool

    init() {
        state = .off
        fixed = false
        refresh()
    }

    /// A stand-in that shows `state` and changes nothing (for snapshots).
    init(fixed state: State) {
        self.state = state
        fixed = true
    }

    func refresh() {
        guard !fixed else { return }
        state = Self.current()
    }

    func set(_ on: Bool) {
        guard !fixed else { return }
        failure = nil
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
        } catch {
            failure = error.localizedDescription
            log.error("login item \(on ? "register" : "unregister", privacy: .public) failed: \(error.localizedDescription, privacy: .public)")
        }
        refresh()
    }

    func openSystemSettings() { SMAppService.openSystemSettingsLoginItems() }

    static func current() -> State {
        switch SMAppService.mainApp.status {
        case .enabled: return .on
        case .requiresApproval: return .needsApproval
        default: return isTemporaryCopy ? .temporaryCopy : .off
        }
    }

    /// Translocated, or on a read-only volume such as the DMG.
    static var isTemporaryCopy: Bool {
        let app = Bundle.main.bundleURL
        if app.path.contains("/AppTranslocation/") { return true }
        return (try? app.resourceValues(forKeys: [.volumeIsReadOnlyKey]).volumeIsReadOnly) ?? false
    }
}
