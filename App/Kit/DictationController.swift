import AVFoundation
import AppKit
import DictatorCore
import Foundation
import Observation

// The app's dictation logic, independent of the UI: what happens when the panel opens or closes, and on
// pause, resume, stop and copy. The SwiftUI views only display `DictationController` and call its methods.

/// A dictation session (`DictationSession`, or a fake in tests).
public protocol DictationSessionType: AnyObject, Sendable {
    var updates: AsyncStream<TranscriptUpdate> { get }
    func push(_ samples: [Float])
    func pause()
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
        case paused
        /// Stopped; the text is editable.
        case done
        case microphoneDenied
        case failed(String)
    }

    public private(set) var phase: Phase = .idle
    public private(set) var model: ModelChoice
    /// Text that will not change, and the word still being recognised.
    public private(set) var committed = ""
    public private(set) var partial = ""
    /// The editable text after Stop.
    public var editedText = ""
    /// Seconds spent listening in this dictation.
    public private(set) var elapsed: TimeInterval = 0
    /// Microphone level, 0...1, while listening.
    public private(set) var level: Float = 0
    /// Briefly true after ⌘C copied the text so far.
    public private(set) var copiedSoFar = false
    /// Whether the panel is showing. The microphone only runs while it is.
    public private(set) var isPanelOpen = false

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

    public init(model: ModelChoice, loader: ModelLoader, microphone: AudioInput, permission: MicrophonePermission, clipboard: Clipboard) {
        self.model = model
        self.loader = loader
        self.microphone = microphone
        self.permission = permission
        self.clipboard = clipboard
    }

    public var text: String { committed + partial }
    public var wordCount: Int { (phase == .done ? editedText : text).split(whereSeparator: \.isWhitespace).count }
    public var isModelReady: Bool { factory != nil && loadedModel == model }
    /// The model menu is usable when no dictation is in progress.
    public var canChangeModel: Bool { phase != .listening && phase != .paused && phase != .preparing }

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

    /// The panel opened: start listening, or resume a dictation paused by closing the panel.
    public func panelOpened() async {
        isPanelOpen = true
        switch phase {
        case .idle: await start()
        case .paused: resume()
        case .preparing: startWhenReady = true
        case .microphoneDenied: if permission.isGranted { phase = .idle; await start() }
        default: break
        }
    }

    /// The panel closed: listening pauses, so the microphone is never on while the panel is hidden.
    public func panelClosed() {
        isPanelOpen = false
        startWhenReady = false
        if phase == .listening { pause() }
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
            let s = try factory!.makeSession()
            session = s
            committed = ""; partial = ""; editedText = ""; elapsed = 0; elapsedBefore = 0
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
        } catch {
            phase = .failed("Couldn’t start the microphone: \(error)")
        }
    }

    func startMicrophone(_ s: DictationSessionType) throws {
        var lastLevel = Date.distantPast
        try microphone.start { [weak self] samples in
            s.push(samples)
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
        listenStart = Date()
        tick?.invalidate()
        tick = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.updateElapsed() }
        }
    }

    func stopMicrophone() {
        microphone.stop()
        tick?.invalidate(); tick = nil
        if let t = listenStart { elapsedBefore += Date().timeIntervalSince(t) }
        listenStart = nil
        elapsed = elapsedBefore
        level = 0
    }

    func updateElapsed() { if let t = listenStart { elapsed = elapsedBefore + Date().timeIntervalSince(t) } }

    public func pause() {
        guard phase == .listening, let s = session else { return }
        stopMicrophone()
        s.pause()
        phase = .paused
    }

    public func resume() {
        guard phase == .paused, isPanelOpen, let s = session else { return }
        s.resume()
        do { try startMicrophone(s); phase = .listening } catch { phase = .failed("Couldn’t start the microphone: \(error)") }
    }

    public func stop() async {
        guard phase == .listening || phase == .paused, let s = session else { return }
        if phase == .listening { stopMicrophone() }
        let final = await s.stop()
        updatesTask?.cancel()
        session = nil
        committed = final; partial = ""
        editedText = final
        phase = .done
    }

    /// ⌘C while listening or paused: copies the text so far without stopping.
    public func copySoFar() {
        guard phase == .listening || phase == .paused else { return }
        clipboard.copy(text)
        copiedSoFar = true
        Task { try? await Task.sleep(for: .seconds(1.5)); copiedSoFar = false }
    }

    /// Copy after Stop: copies the edited text and clears the dictation. The panel closes.
    public func copyAndFinish() {
        guard phase == .done else { return }
        clipboard.copy(editedText)
        reset()
    }

    /// New: clears the dictation and starts listening again.
    public func newDictation() async {
        guard phase == .done else { return }
        reset()
        await start()
    }

    /// Opens the Microphone pane of System Settings.
    public func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") { NSWorkspace.shared.open(url) }
    }

    func reset() {
        committed = ""; partial = ""; editedText = ""; elapsed = 0; elapsedBefore = 0
        phase = .idle
    }
}
