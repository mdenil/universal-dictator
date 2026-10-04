import AVFoundation
import AppKit
import DictatorCore
import Foundation
import Observation
import OSLog

let timing = Logger(subsystem: "dev.universaldictator.UniversalDictator", category: "timing")

// The app's dictation logic, independent of the UI: what happens when the panel opens or closes, and on
// pause, resume, stop and copy. The SwiftUI views only display `DictationController` and call its methods.

/// A dictation session (`DictationSession`, or a fake in tests).
public protocol DictationSessionType: AnyObject, Sendable {
    var updates: AsyncStream<TranscriptUpdate> { get }
    func push(_ samples: [Float])
    func pause()
    /// The whole text once a pause has finished the last words.
    func settledText() async -> String
    /// While paused: replaces the text so far with the user's edit.
    func replaceText(_ text: String)
    func resume()
    func stop() async -> String
}
extension DictationSession: DictationSessionType {}

/// A loaded model that starts sessions.
public protocol SessionFactory: AnyObject, Sendable {
    func makeSession() throws -> DictationSessionType
}
extension LoadedModel: SessionFactory {
    public func makeSession() throws -> DictationSessionType { try startSession() }
}

/// Loads a model choice.
public protocol ModelLoader: Sendable {
    func load(_ choice: ModelChoice) async throws -> SessionFactory
}

/// The engine's loader for a model folder (in the app, the bundle's NemotronAssets folder).
public struct EngineLoader: ModelLoader {
    public let assets: ModelAssets
    public init(assets: ModelAssets) { self.assets = assets }
    public func load(_ choice: ModelChoice) async throws -> SessionFactory { try await Engine.load(choice, assets: assets) }
}

/// Microphone input (16 kHz mono).
public protocol AudioInput: AnyObject {
    func start(_ onSamples: @escaping ([Float]) -> Void) throws
    func stop()
}
extension Microphone: AudioInput {}

/// Microphone permission.
public protocol MicrophonePermission {
    var isGranted: Bool { get }
    var isDenied: Bool { get }
    func request() async -> Bool
}

public struct SystemMicrophonePermission: MicrophonePermission {
    public init() {}
    public var isGranted: Bool { AVCaptureDevice.authorizationStatus(for: .audio) == .authorized }
    public var isDenied: Bool { [.denied, .restricted].contains(AVCaptureDevice.authorizationStatus(for: .audio)) }
    public func request() async -> Bool { await AVCaptureDevice.requestAccess(for: .audio) }
}

/// The clipboard.
public protocol Clipboard { func copy(_ text: String) }
public struct SystemClipboard: Clipboard {
    public init() {}
    public func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

@MainActor @Observable
public final class DictationController {
    public enum Phase: Equatable {
        /// No dictation; opening the panel starts one.
        case idle
        /// A model is loading.
        case preparing
        case listening
        /// The microphone is off and the text is editable; Resume appends to it. Survives closing the panel.
        case paused
        /// Stopped; the text is editable.
        case done
        case microphoneDenied
        /// The direct-dictation shortcut was used before typing into other apps was allowed.
        case accessibilityNeeded
        case failed(String)
    }

    public private(set) var phase: Phase = .idle
    public private(set) var model: ModelChoice
    /// Text that will not change, and the word still being recognised.
    public private(set) var committed = ""
    public private(set) var partial = ""
    /// The editable text while paused and after Stop.
    public var editedText = ""
    /// The text as of the pause, before any edit: Resume and Stop hand an edit to the session only if
    /// `editedText` differs from it.
    var pausedText = ""
    /// Seconds spent listening in this dictation.
    public private(set) var elapsed: TimeInterval = 0
    /// Microphone level, 0...1, while listening.
    public private(set) var level: Float = 0
    /// Whether the panel is showing. The microphone only runs while it is.
    public private(set) var isPanelOpen = false
    /// Finished dictations, newest first, saved between launches.
    public private(set) var history: [HistoryEntry] = []
    /// How many dictations the history keeps; lowering it drops the oldest at once.
    public var historyLimit = 50 {
        didSet { if trimHistory() { historyStore.save(history) } }
    }
    public static let historyLimits = [10, 25, 50, 100, 200]
    let historyStore: HistoryStore

    let loader: ModelLoader
    let microphone: AudioInput
    let permission: MicrophonePermission
    let clipboard: Clipboard
    var factory: SessionFactory?
    var loadedModel: ModelChoice?
    var session: DictationSessionType?
    var updatesTask: Task<Void, Never>?
    var loadTask: Task<Void, Never>?
    var startWhenReady = false
    var listenStart: Date?
    var elapsedBefore: TimeInterval = 0
    var tick: Timer?
    /// A stop is waiting for the microphone and the session to finish.
    var stopping = false
    /// Whether the microphone is actually capturing: it takes a moment (sometimes seconds) to wake after
    /// starting. The menu-bar icon turns red only then.
    public private(set) var isMicrophoneLive = false
    /// Counts microphone starts and stops, so a start that finishes after a later stop doesn't count as live.
    var microphoneGeneration = 0

    /// Where the microphone is started and stopped. Starting can take a second or two (waking the audio
    /// device), so the app passes a serial background queue and the panel draws at once; starts and stops
    /// stay in order on it, so a stop always follows a start still in progress. nil runs them inline (tests).
    let microphoneQueue: DispatchQueue?

    public init(model: ModelChoice, loader: ModelLoader, microphone: AudioInput, permission: MicrophonePermission, clipboard: Clipboard,
                microphoneQueue: DispatchQueue? = nil, history: HistoryStore = MemoryHistoryStore(), historyLimit: Int = 50) {
        self.microphoneQueue = microphoneQueue
        self.historyStore = history
        self.historyLimit = historyLimit
        self.history = Array(history.load().prefix(historyLimit))
        self.model = model
        self.loader = loader
        self.microphone = microphone
        self.permission = permission
        self.clipboard = clipboard
    }

    /// The text shown: live while listening, the editable text while paused or stopped.
    public var text: String { phase == .paused || phase == .done ? editedText : committed + partial }
    public var wordCount: Int { (phase == .done ? editedText : text).split(whereSeparator: \.isWhitespace).count }
    public var isModelReady: Bool { factory != nil && loadedModel == model }
    /// The model menu is usable when no dictation is in progress.
    public var canChangeModel: Bool { phase != .listening && phase != .paused && phase != .preparing && !isDirectDictating }

    /// Loads the current model (at launch, or after changing models).
    public func loadModel() {
        loadTask?.cancel()
        factory = nil
        loadedModel = nil
        let choice = model
        if phase == .idle || phase == .done || isFailure { phase = .preparing }
        loadTask = Task {
            do {
                let f = try await loader.load(choice)
                guard !Task.isCancelled, choice == self.model else { return }
                self.factory = f
                self.loadedModel = choice
                if self.phase == .preparing { self.phase = self.text.isEmpty && self.editedText.isEmpty ? .idle : .done }
                if self.startWhenReady { self.startWhenReady = false; await self.start() }
            } catch {
                guard !Task.isCancelled else { return }
                self.phase = .failed("Couldn’t load \(choice.displayName): \(error)")
            }
        }
    }

    var isFailure: Bool { if case .failed = phase { return true }; return false }

    public func selectModel(_ choice: ModelChoice) {
        guard canChangeModel, choice != model else { return }
        model = choice
        loadModel()
    }

    // MARK: Panel

    /// The panel opened: a new dictation starts, unless one is paused (closing the panel keeps a paused
    /// dictation; every other one ended when the panel closed).
    public func panelOpened() async {
        isPanelOpen = true
        guard !isDirectDictating else { return }       // the app stops a direct dictation before opening the panel
        switch phase {
        case .idle: await start()
        case .preparing: startWhenReady = true
        case .microphoneDenied: if permission.isGranted { phase = .idle; await start() }
        default: break
        }
    }

    /// The panel closed: a paused dictation is kept for the next opening. Any other dictation ends: the
    /// microphone stops at once, the text goes to the history, and the next opening starts afresh.
    public func panelClosed() {
        isPanelOpen = false
        startWhenReady = false
        switch phase {
        case .listening:
            let stopped = stopMicrophone()
            let s = session, shown = text, choice = model
            session = nil
            updatesTask?.cancel()
            reset()
            Task {
                await stopped.value                      // the microphone's last audio reaches the session first
                let final = await s?.stop() ?? shown
                self.archive(final, model: choice, copied: false)
            }
        case .done:
            archive(editedText, model: model, copied: false)
            reset()
        case .accessibilityNeeded:
            phase = .idle
        default:
            break
        }
    }

    /// Adds a finished dictation to the history (empty text is not kept).
    func archive(_ text: String, model: ModelChoice, copied: Bool, typed: Bool = false) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        history.insert(HistoryEntry(text: t, date: Date(), model: model, copied: copied, typed: typed), at: 0)
        _ = trimHistory()
        historyStore.save(history)
    }

    /// Drops the oldest entries beyond the limit; true if any went.
    func trimHistory() -> Bool {
        guard history.count > historyLimit else { return false }
        history.removeLast(history.count - historyLimit)
        return true
    }

    /// Copies a history entry.
    public func copyFromHistory(_ entry: HistoryEntry) { clipboard.copy(entry.text) }

    /// Saves an edit made in the History window. The entry keeps its place and time and is marked edited;
    /// empty text is refused (Delete removes an entry).
    public func editHistory(_ id: UUID, text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, let i = history.firstIndex(where: { $0.id == id }), history[i].text != t else { return }
        history[i].text = t
        history[i].edited = true
        historyStore.save(history)
    }

    public func deleteHistory(_ id: UUID) {
        history.removeAll { $0.id == id }
        historyStore.save(history)
    }

    public func clearHistory() {
        history.removeAll()
        historyStore.save(history)
    }

    /// The app is quitting: whatever is dictated or edited goes to the history (synchronously, with the text
    /// as shown; a listening session's last words are not waited for).
    public func finishForQuit() {
        if isDirectDictating, directSession != nil {
            stopMicrophone(timed: false)
            isDirectDictating = false
            archive(directTyped, model: model, copied: false, typed: true)
            directSession = nil
        }
        switch phase {
        case .listening, .paused, .done:
            if phase == .listening { stopMicrophone() }
            archive(text, model: model, copied: false)
            session = nil
            reset()
        default:
            break
        }
    }

    /// Asks for microphone access up front (at launch), so the prompt never appears in the middle of a click.
    public func requestMicrophoneAccessIfNeeded() async {
        if !permission.isGranted && !permission.isDenied { _ = await permission.request() }
    }

    // MARK: Dictation

    public func start() async {
        guard factory != nil, loadedModel == model else { startWhenReady = true; if phase != .preparing { loadModel() }; return }
        if !permission.isGranted {
            if permission.isDenied { phase = .microphoneDenied; return }
            let granted = await permission.request()
            if !granted { phase = .microphoneDenied; return }
        }
        // The panel may have closed while this was starting (a permission prompt takes focus): the microphone
        // never runs with the panel hidden.
        guard isPanelOpen else { return }
        do {
            let t0 = Date()
            let s = try factory!.makeSession()
            let t1 = Date()
            session = s
            committed = ""; partial = ""; editedText = ""; pausedText = ""; elapsed = 0; elapsedBefore = 0
            updatesTask?.cancel()
            updatesTask = Task { [weak self] in
                for await u in s.updates {
                    guard let self, self.session === s else { break }
                    self.committed = u.committed
                    self.partial = u.partial
                }
            }
            try startMicrophone(s)
            phase = .listening
            timing.notice("start: session \(Int(t1.timeIntervalSince(t0) * 1000)) ms")
        } catch {
            phase = .failed("Couldn’t start the microphone: \(error)")
        }
    }

    /// Starts the microphone into `s`. `timed`: counts towards the panel dictation's elapsed time and shows
    /// the level (not for direct dictation, which has no panel).
    func startMicrophone(_ s: DictationSessionType, timed: Bool = true) throws {
        var lastLevel = Date.distantPast
        let mic = microphone
        let t0 = Date()
        microphoneGeneration += 1
        let generation = microphoneGeneration
        let onSamples: ([Float]) -> Void = { [weak self] samples in
            s.push(samples)
            guard timed else { return }
            let now = Date()
            if now.timeIntervalSince(lastLevel) > 0.05 {
                lastLevel = now
                var sum: Float = 0
                for x in samples { sum += x * x }
                let rms = (sum / Float(max(samples.count, 1))).squareRoot()
                let level = min(1, max(0, (20 * log10(max(rms, 1e-5)) + 50) / 40))       // -50 dB ... -10 dB
                Task { @MainActor in self?.level = level }
            }
        }
        if let q = microphoneQueue {
            q.async { [weak self] in
                do {
                    try mic.start(onSamples)
                    timing.notice("microphone started in \(Int(Date().timeIntervalSince(t0) * 1000)) ms")
                    Task { @MainActor in
                        guard let self, self.microphoneGeneration == generation else { return }
                        self.isMicrophoneLive = true
                    }
                } catch {
                    Task { @MainActor in
                        guard let self, self.session === s else { return }
                        self.stopMicrophone()
                        self.phase = .failed("Couldn’t start the microphone: \(error)")
                    }
                }
            }
        } else {
            try mic.start(onSamples)
            isMicrophoneLive = true
        }
        guard timed else { return }
        listenStart = Date()
        tick?.invalidate()
        tick = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.updateElapsed() }
        }
    }

    /// Stops the microphone. It hands over the audio it captured last while stopping, so a session must not
    /// finish (stop or pause) until it has: the returned task completes then. `wait` blocks until then instead
    /// (for pause, whose session calls must stay in order with a resume).
    @discardableResult
    func stopMicrophone(timed: Bool = true, wait: Bool = false) -> Task<Void, Never> {
        let mic = microphone
        microphoneGeneration += 1
        isMicrophoneLive = false
        let stopped: Task<Void, Never>
        if let q = microphoneQueue {
            if wait {
                q.sync { mic.stop() }
                stopped = Task {}
            } else {
                stopped = Task.detached { await withCheckedContinuation { c in q.async { mic.stop(); c.resume() } } }
            }
        } else {
            mic.stop()
            stopped = Task {}
        }
        defer { if timed { stopTimer() } }
        return stopped
    }

    func stopTimer() {
        tick?.invalidate(); tick = nil
        if let t = listenStart { elapsedBefore += Date().timeIntervalSince(t) }
        listenStart = nil
        elapsed = elapsedBefore
        level = 0
    }

    func updateElapsed() { if let t = listenStart { elapsed = elapsedBefore + Date().timeIntervalSince(t) } }

    /// Pause: the microphone stops and the text becomes editable. It shows at once; the last words, which
    /// the session finishes a moment later, are added unless the user has already started editing.
    public func pause() {
        guard phase == .listening, let s = session else { return }
        stopMicrophone(wait: true)
        s.pause()
        let shown = committed + partial
        editedText = shown; pausedText = shown
        phase = .paused
        Task {
            let full = await s.settledText()
            guard self.session === s, self.phase == .paused, self.pausedText == shown else { return }
            if self.editedText == shown { self.editedText = full }
            self.pausedText = full
        }
    }

    /// Resume: an edit made while paused replaces the text so far, and new words are appended to it.
    public func resume() {
        guard phase == .paused, isPanelOpen, let s = session else { return }
        handOverEdit(s)
        committed = editedText; partial = ""
        s.resume()
        do { try startMicrophone(s); phase = .listening } catch { phase = .failed("Couldn’t start the microphone: \(error)") }
    }

    func handOverEdit(_ s: DictationSessionType) {
        if editedText != pausedText { s.replaceText(editedText) }
    }

    public func stop() async {
        guard phase == .listening || phase == .paused, let s = session, !stopping else { return }
        stopping = true
        defer { stopping = false }
        if phase == .listening { await stopMicrophone().value } else { handOverEdit(s) }
        let final = await s.stop()
        updatesTask?.cancel()
        session = nil
        committed = final; partial = ""
        editedText = final
        phase = .done
    }

    /// Copy ⌘C (the button or the shortcut) while listening or paused: stops (the microphone goes off and the last words are finished),
    /// copies the whole text and clears the dictation. The panel closes.
    public func stopAndCopy() async {
        guard phase == .listening || phase == .paused else { return }
        await stop()
        clipboard.copy(editedText)
        archive(editedText, model: model, copied: true)
        reset()
    }

    /// Copy after Stop: copies the edited text and clears the dictation. The panel closes.
    public func copyAndFinish() {
        guard phase == .done else { return }
        clipboard.copy(editedText)
        archive(editedText, model: model, copied: true)
        reset()
    }

    /// New: clears the dictation and starts listening again.
    public func newDictation() async {
        guard phase == .done else { return }
        reset()
        await start()
    }

    // MARK: Direct dictation

    /// Whether a direct dictation is listening (the menu-bar icon is red; there is no panel).
    public private(set) var isDirectDictating = false
    var directSession: DictationSessionType?
    var directUpdates: Task<Void, Never>?
    /// What has been typed so far in this direct dictation.
    var directTyped = ""
    var typer: TextTyper?
    var accessibility: AccessibilityPermission?

    /// Connects direct dictation to the app's typing and permission (the app calls this once at launch).
    public func enableDirectDictation(typer: TextTyper, accessibility: AccessibilityPermission) {
        self.typer = typer
        self.accessibility = accessibility
    }

    /// Starts a direct dictation: the microphone on, each finished word typed into the focused field. The
    /// panel must be closed (the app closes it first). A paused panel dictation is left as it is.
    public func startDirectDictation() async -> DirectStartResult {
        guard !isDirectDictating, !isPanelOpen, let typer else { return .ignored }
        guard accessibility?.isGranted ?? false else { return .needsAccessibility }
        guard factory != nil, loadedModel == model, permission.isGranted else { return .needsPanel }
        do {
            let s = try factory!.makeSession()
            directSession = s
            directTyped = ""
            isDirectDictating = true
            directUpdates = Task { [weak self] in
                for await u in s.updates {
                    guard let self, self.directSession === s else { break }
                    self.typeNewWords(u.committed, typer)
                }
            }
            try startMicrophone(s, timed: false)
            timing.notice("direct dictation started")
            return .started
        } catch {
            isDirectDictating = false
            directSession = nil
            return .needsPanel
        }
    }

    /// Stops a direct dictation: the microphone off, the last words typed, the text to the history.
    public func stopDirectDictation() async {
        guard isDirectDictating, let s = directSession, let typer else { return }
        isDirectDictating = false
        await stopMicrophone(timed: false).value
        let final = await s.stop()
        directUpdates?.cancel()
        directSession = nil
        typeNewWords(final, typer)
        archive(final, model: model, copied: false, typed: true)
        directTyped = ""
    }

    func typeNewWords(_ text: String, _ typer: TextTyper) {
        guard let new = untypedSuffix(of: text, typed: directTyped) else {
            timing.error("direct dictation: text no longer starts with what was typed; not typing more")
            return
        }
        guard !new.isEmpty else { return }
        typer.type(new)
        directTyped += new
    }

    /// The shortcut was used before Accessibility was allowed: the panel (which the app opens) explains.
    public func showAccessibilityNeeded() {
        if phase == .idle { phase = .accessibilityNeeded }
    }

    /// Opens the Accessibility pane of System Settings.
    public func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") { NSWorkspace.shared.open(url) }
    }

    /// Opens the Microphone pane of System Settings.
    public func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") { NSWorkspace.shared.open(url) }
    }

    func reset() {
        committed = ""; partial = ""; editedText = ""; pausedText = ""; elapsed = 0; elapsedBefore = 0
        phase = .idle
    }
}
