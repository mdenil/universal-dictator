import AppKit
import DictatorCore
import SwiftUI
import UniversalDictatorKit

// Development launch modes, for checking the app without clicking through it:
//   UniversalDictator --selftest <audio> [--model nemotron160|nemotron560]
//       runs <audio> through the app's controller and bundled models. Halfway it pauses, closes and reopens
//       the panel, edits the paused text (prefixes "[edited] ") and resumes; prints the final text and exits.
//   UniversalDictator --snapshot <dir>
//       renders the panel in each state to <dir>/<state>.png with stand-in sessions and exits.
enum DevModes {
    static func argument(after flag: String) -> String? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    /// Runs a development mode if one was requested; returns false otherwise.
    @MainActor static func run(assets: ModelAssets) -> Bool {
        if let audio = argument(after: "--selftest") {
            let choice = argument(after: "--model").flatMap(ModelChoice.init(rawValue:)) ?? .nemotron160
            Task { await selftest(audio: URL(fileURLWithPath: audio), choice: choice, assets: assets); exit(0) }
            return true
        }
        if let dir = argument(after: "--snapshot") {
            Task { await snapshots(to: URL(fileURLWithPath: dir)); exit(0) }
            return true
        }
        return false
    }

    // MARK: Self-test

    /// Audio from a file, delivered in 20 ms pieces on a background thread, as the microphone would.
    final class FileInput: AudioInput {
        let audio: [Float]
        var position = 0
        var running = false
        let lock = NSLock()
        init(_ audio: [Float]) { self.audio = audio }
        var finished: Bool { lock.lock(); defer { lock.unlock() }; return position >= audio.count }
        func start(_ onSamples: @escaping ([Float]) -> Void) throws {
            lock.lock(); running = true; lock.unlock()
            Thread.detachNewThread { [self] in
                while true {
                    lock.lock()
                    guard running, position < audio.count else { running = false; lock.unlock(); return }
                    let k = min(320, audio.count - position)
                    let piece = Array(audio[position..<(position + k)])
                    position += k
                    lock.unlock()
                    onSamples(piece)
                    usleep(2_000)                                   // ~10x faster than real time
                }
            }
        }
        func stop() { lock.lock(); running = false; lock.unlock(); usleep(10_000) }
    }

    struct Granted: MicrophonePermission { var isGranted: Bool { true }; var isDenied: Bool { false }; func request() async -> Bool { true } }
    final class NoClipboard: Clipboard { var text = ""; func copy(_ t: String) { text = t } }

    @MainActor static func selftest(audio url: URL, choice: ModelChoice, assets: ModelAssets) async {
        func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
        do {
            let audio = try AudioFile.readMono16k(url)
            let input = FileInput(audio)
            let c = DictationController(model: choice, loader: EngineLoader(assets: assets), microphone: input, permission: Granted(), clipboard: NoClipboard())
            let t0 = Date()
            c.loadModel()
            while c.phase == .preparing { try await Task.sleep(for: .milliseconds(50)) }
            if case .failed(let m) = c.phase { log("load failed: \(m)"); return }
            log(String(format: "model %@ ready in %.1f s", choice.displayName, Date().timeIntervalSince(t0)))
            await c.panelOpened()
            log("phase after opening: \(c.phase)")
            while input.position < audio.count / 2 { try await Task.sleep(for: .milliseconds(10)) }
            c.pause()
            try await Task.sleep(for: .milliseconds(300))
            log("paused at \(String(format: "%.1f", Double(input.position) / 16000)) s: \(c.text)")
            c.panelClosed()                                       // a paused dictation survives closing
            await c.panelOpened()
            log("phase after closing and reopening: \(c.phase)")
            c.editedText = "[edited] " + c.editedText
            c.resume()
            while !input.finished { try await Task.sleep(for: .milliseconds(20)) }
            try await Task.sleep(for: .milliseconds(300))
            await c.stop()
            log("phase after stop: \(c.phase), \(c.wordCount) words")
            print(c.editedText)
        } catch {
            log("selftest failed: \(error)")
        }
    }

    // MARK: Snapshots

    final class StandInSession: DictationSessionType, @unchecked Sendable {
        let updates: AsyncStream<TranscriptUpdate>
        let cont: AsyncStream<TranscriptUpdate>.Continuation
        let final: String
        init(final: String) { self.final = final; (updates, cont) = AsyncStream<TranscriptUpdate>.makeStream() }
        func push(_ samples: [Float]) {}
        var said = ""
        func pause() {}
        func settledText() async -> String { said }
        func replaceText(_ text: String) { said = text }
        func resume() {}
        func stop() async -> String { cont.finish(); return final }
        func say(_ c: String, _ p: String) { said = c + p; cont.yield(TranscriptUpdate(committed: c, partial: p, isPaused: false, isFinished: false)) }
    }
    final class StandInFactory: SessionFactory, @unchecked Sendable {
        var last: StandInSession?
        func makeSession() throws -> DictationSessionType {
            let s = StandInSession(final: "Hi Sarah, thanks for sending the draft over. I’ll take a look this afternoon and get back to you by Thursday.")
            last = s; return s
        }
    }
    struct StandInLoader: ModelLoader {
        let factory: StandInFactory
        var hang = false, fail = false
        func load(_ choice: ModelChoice) async throws -> SessionFactory {
            if fail { throw EngineError.io("missing model files: nemotron_160_state.mlmodelc") }
            if hang { try await Task.sleep(for: .seconds(3600)) }
            return factory
        }
    }
    final class SilentInput: AudioInput { func start(_ onSamples: @escaping ([Float]) -> Void) throws {}; func stop() {} }
    struct Permission: MicrophonePermission { let granted: Bool; var isGranted: Bool { granted }; var isDenied: Bool { !granted }; func request() async -> Bool { granted } }

    @MainActor static func snapshots(to dir: URL) async {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        func make(hang: Bool = false, fail: Bool = false, granted: Bool = true) -> (DictationController, StandInFactory) {
            let f = StandInFactory()
            let c = DictationController(model: .nemotron160, loader: StandInLoader(factory: f, hang: hang, fail: fail),
                                        microphone: SilentInput(), permission: Permission(granted: granted), clipboard: NoClipboard())
            return (c, f)
        }
        func settle() async { try? await Task.sleep(for: .milliseconds(150)) }

        var (c, f) = make()
        c.loadModel(); await settle(); await c.panelOpened()
        f.last?.say("Hi Sarah, thanks for sending the draft over. I’ll take a look this ", "after")
        await settle(); render(c, "1-listening", dir)
        c.pause(); await settle(); render(c, "2-paused", dir)
        await c.stop(); await settle(); render(c, "3-done", dir)
        (c, _) = make(hang: true); c.loadModel(); await settle(); render(c, "4-preparing", dir)
        (c, _) = make(granted: false); c.loadModel(); await settle(); await c.panelOpened(); render(c, "5-microphone-denied", dir)
        (c, _) = make(fail: true); c.loadModel(); await settle(); render(c, "6-failed", dir)
    }

    @MainActor static func render(_ c: DictationController, _ name: String, _ dir: URL) {
        let host = NSHostingView(rootView: PanelView(controller: c, close: {}))
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        let bg = PanelBackground(content: host)
        let window = NSWindow(contentRect: NSRect(x: -4000, y: -4000, width: host.frame.width, height: host.frame.height),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = bg
        window.orderFrontRegardless()
        bg.layoutSubtreeIfNeeded()
        bg.display()
        guard let rep = bg.bitmapImageRepForCachingDisplay(in: bg.bounds) else { return }
        bg.cacheDisplay(in: bg.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: dir.appendingPathComponent(name + ".png"))
        window.orderOut(nil)
    }
}
