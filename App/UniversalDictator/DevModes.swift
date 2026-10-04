import AppKit
import ApplicationServices
import DictatorCore
import SwiftUI
import UniversalDictatorKit

// Development launch modes, for checking the app without clicking through it:
//   UniversalDictator --selftest <audio> [--model nemotron160|nemotron560]
//       runs <audio> through the app's controller and bundled models. Halfway it pauses, closes and reopens
//       the panel, edits the paused text (prefixes "[edited] ") and resumes; prints the final text and exits.
//   UniversalDictator --measure-switch
//       loads Fast, then Accurate, then switches back through the app's controller, printing each step's
//       time and the process's memory footprint, and exits. Run it twice: the second run shows loads with
//       the Neural Engine compile already cached on disk.
//   UniversalDictator --check-accessibility
//       prints whether this app may type into other apps, and exits.
//   UniversalDictator --typing-test
//       types into a test window of its own by each route (HID tap, session tap, the front app), with no key
//       held, with Shift held and with ⌃⌥ held (simulated for under half a second), and prints what arrived.
//       Needs Accessibility. Takes the keyboard focus for a few seconds.
//   UniversalDictator --direct-selftest <audio> [--model nemotron160|nemotron560]
//       direct dictation of <audio> into a test window of its own, through the real model and real typing;
//       prints PASS if the window holds exactly the dictation's final text and a space. Needs Accessibility.
//   UniversalDictator --target-probe [rounds]
//       for the "keep typing into the same place" investigation, guided by sounds, `rounds` times (default 6):
//       click into a text field in any app; at the first chime (5 s) it remembers that app and field; switch to
//       another app; at the second chime (5 s later) it tries, with the remembered app in the background,
//       (A) keystrokes posted to that app's process and (B) inserting through Accessibility into the remembered
//       field, typing the markers ⟨A⟩ and ⟨B⟩. It prints what it could read back from the field. Needs Accessibility.
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
        if CommandLine.arguments.contains("--check-accessibility") {
            print(AXIsProcessTrusted() ? "allowed" : "not allowed")
            exit(0)
        }
        if CommandLine.arguments.contains("--typing-test") {
            Task { await typingTest(); exit(0) }
            return true
        }
        if let audio = argument(after: "--direct-selftest") {
            let choice = argument(after: "--model").flatMap(ModelChoice.init(rawValue:)) ?? .nemotron160
            Task { await directSelftest(audio: URL(fileURLWithPath: audio), choice: choice, assets: assets); exit(0) }
            return true
        }
        if CommandLine.arguments.contains("--target-probe") {
            let rounds = argument(after: "--target-probe").flatMap(Int.init) ?? 6
            Task { await targetProbe(rounds: rounds); exit(0) }
            return true
        }
        if CommandLine.arguments.contains("--measure-switch") {
            Task { await measureSwitch(assets: assets); exit(0) }
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

    // MARK: Typing

    /// A window with a text view, in front and focused.
    @MainActor static func focusedTextWindow() -> (NSWindow, NSTextView) {
        let w = NSWindow(contentRect: NSRect(x: 200, y: 400, width: 520, height: 200), styleMask: [.titled], backing: .buffered, defer: false)
        w.title = "Universal Dictator typing test"
        let scroll = NSTextView.scrollableTextView()
        let tv = scroll.documentView as! NSTextView
        w.contentView = scroll
        NSApp.setActivationPolicy(.accessory)
        NSApp.activate(ignoringOtherApps: true)
        w.makeKeyAndOrderFront(nil)
        w.makeFirstResponder(tv)
        return (w, tv)
    }

    /// Simulates holding (or releasing) modifier keys, as the keyboard would.
    static func hold(_ keys: [(UInt16, CGEventFlags)], down: Bool) {
        let src = CGEventSource(stateID: .hidSystemState)
        var flags: CGEventFlags = []
        for (code, flag) in (down ? keys : keys.reversed()) {
            if down { flags.insert(flag) } else { flags.remove(flag) }
            guard let e = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: down) else { continue }
            e.type = .flagsChanged
            e.flags = flags
            e.post(tap: .cghidEventTap)
        }
    }

    @MainActor static func typingTest() async {
        func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
        guard AXIsProcessTrusted() else { log("Accessibility is not allowed for this app; nothing typed"); return }
        let (w, tv) = focusedTextWindow()
        try? await Task.sleep(for: .milliseconds(500))
        let sample = "Hi there, abc s q."
        let holds: [(String, [(UInt16, CGEventFlags)])] = [("nothing held", []), ("Shift held", [(56, .maskShift)]),
                                                           ("⌃⌥ held", [(59, .maskControl), (58, .maskAlternate)])]
        for route in KeyboardTyper.Route.allCases {
            for (label, keys) in holds {
                tv.string = ""
                hold(keys, down: true)
                try? await Task.sleep(for: .milliseconds(60))
                let seen = NSEvent.modifierFlags.intersection([.shift, .control, .option]).rawValue
                KeyboardTyper.post(sample, route: route)
                try? await Task.sleep(for: .milliseconds(300))
                hold(keys, down: false)
                try? await Task.sleep(for: .milliseconds(150))
                let got = tv.string
                log("\(route.rawValue), \(label) (modifiers seen \(seen)): \(got == sample ? "OK" : "DIFFERENT") \"\(got)\"")
            }
        }
        w.orderOut(nil)
    }

    @MainActor static func directSelftest(audio url: URL, choice: ModelChoice, assets: ModelAssets) async {
        func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
        guard AXIsProcessTrusted() else { log("Accessibility is not allowed for this app; nothing typed"); return }
        do {
            let audio = try AudioFile.readMono16k(url)
            let input = FileInput(audio)
            let typer = KeyboardTyper()
            let c = DictationController(model: choice, loader: EngineLoader(assets: assets), microphone: input, permission: Granted(),
                                        clipboard: NoClipboard())
            c.enableDirectDictation(typer: typer, accessibility: SystemAccessibility())
            c.loadModel()
            while c.phase == .preparing { try await Task.sleep(for: .milliseconds(50)) }
            let (w, tv) = focusedTextWindow()
            try await Task.sleep(for: .milliseconds(500))
            let r = await c.startDirectDictation()
            log("start: \(r)")
            var typedWhileListening = ""
            while !input.finished {
                try await Task.sleep(for: .milliseconds(20))
                if typedWhileListening.isEmpty, !tv.string.isEmpty { typedWhileListening = tv.string; log("first words typed while listening: \"\(tv.string)\"") }
            }
            try await Task.sleep(for: .milliseconds(300))
            await c.stopDirectDictation()
            try await Task.sleep(for: .milliseconds(800))           // the typing queue finishes
            let final = c.history.first?.text ?? ""
            let got = tv.string
            log(got == final + " " && !final.isEmpty ? "PASS" : "FAIL")             // the final text and a space
            log("typed:   \(got)")
            log("history: \(final)")
            w.orderOut(nil)
        } catch {
            log("direct selftest failed: \(error)")
        }
    }

    // MARK: Typing into a fixed place

    static func axValue(_ e: AXUIElement, _ attr: String) -> CFTypeRef? {
        var v: CFTypeRef?
        return AXUIElementCopyAttributeValue(e, attr as CFString, &v) == .success ? v : nil
    }

    @MainActor static func targetProbe(rounds: Int) async {
        func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
        guard AXIsProcessTrusted() else { log("Accessibility is not allowed for this app"); return }
        let me = ProcessInfo.processInfo.processIdentifier
        for round in 1...rounds {
            log("round \(round): click into a text field in an app; first chime in 5 s")
            try? await Task.sleep(for: .seconds(5))
            guard let target = NSWorkspace.shared.frontmostApplication, target.processIdentifier != me else { log("  no app in front; skipped"); continue }
            let app = AXUIElementCreateApplication(target.processIdentifier)
            let field = axValue(app, kAXFocusedUIElementAttribute as String).map { $0 as! AXUIElement }
            let role = field.flatMap { axValue($0, kAXRoleAttribute as String) as? String } ?? "none"
            let settable: Bool = {
                guard let f = field else { return false }
                var ok = DarwinBoolean(false)
                return AXUIElementIsAttributeSettable(f, kAXSelectedTextAttribute as CFString, &ok) == .success && ok.boolValue
            }()
            _ = StartSound.play()
            log("  remembered \(target.localizedName ?? "?") (pid \(target.processIdentifier)); focused element role \(role); selected text settable: \(settable)")
            log("  now switch to a different app; second chime in 5 s")
            try? await Task.sleep(for: .seconds(5))
            let front = NSWorkspace.shared.frontmostApplication
            if front?.processIdentifier == target.processIdentifier { log("  \(target.localizedName ?? "?") is still in front; this round doesn't test the background") }
            // A: keystrokes posted to the remembered app's process.
            KeyboardTyper.post("⟨A⟩", route: .frontmostApp(pid: target.processIdentifier))
            try? await Task.sleep(for: .milliseconds(400))
            // B: insert at the remembered field's cursor through Accessibility.
            var bResult = "no field"
            if let f = field { bResult = "\(AXUIElementSetAttributeValue(f, kAXSelectedTextAttribute as CFString, "⟨B⟩" as CFString).rawValue)" }
            try? await Task.sleep(for: .milliseconds(400))
            _ = StartSound.play()
            let value = field.flatMap { axValue($0, kAXValueAttribute as String) as? String }
            let tail = value.map { String($0.suffix(60)).replacingOccurrences(of: "\n", with: "⏎") }
            log("  front while typing: \(front?.localizedName ?? "?"); B set-result \(bResult) (0 = success)")
            log("  field now ends: \(tail.map { "\"\($0)\"" } ?? "(can't read the field)")")
            log("  A arrived: \(value.map { $0.contains("⟨A⟩") ? "yes" : "no" } ?? "unknown")   B arrived: \(value.map { $0.contains("⟨B⟩") ? "yes" : "no" } ?? "unknown")")
        }
        log("done")
    }

    // MARK: Model switching

    /// The process's memory footprint in MB (what Activity Monitor shows as Memory).
    static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }

    @MainActor static func measureSwitch(assets: ModelAssets) async {
        func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
        let c = DictationController(model: .nemotron160, loader: EngineLoader(assets: assets), microphone: SilentInput(),
                                    permission: Granted(), clipboard: NoClipboard())
        log("start: \(footprintMB()) MB")
        for (label, choice) in [("load Fast", ModelChoice.nemotron160), ("switch to Accurate", .nemotron560), ("switch back to Fast", .nemotron160),
                                ("switch back to Accurate", .nemotron560)] {
            let t0 = Date()
            if c.model == choice { c.loadModel() } else { c.selectModel(choice) }
            while !c.isModelReady {
                if case .failed(let m) = c.phase { log("failed: \(m)"); return }
                try? await Task.sleep(for: .milliseconds(20))
            }
            log(String(format: "%@: %.2f s, %d MB", label, Date().timeIntervalSince(t0), footprintMB()))
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
        (c, _) = make(); c.loadModel(); await settle(); c.showAccessibilityNeeded(); render(c, "6b-accessibility-needed", dir)

        // Settings and History, with a few saved dictations.
        let now = Date()
        let saved = MemoryHistoryStore([
            HistoryEntry(text: "Hi Sarah, thanks for sending the draft over. I’ll take a look this afternoon and get back to you by Thursday.",
                         date: now.addingTimeInterval(-120), model: .nemotron160, copied: true),
            HistoryEntry(text: "Remind me to pick up milk, eggs and a loaf of sourdough on the way home.",
                         date: now.addingTimeInterval(-840), model: .nemotron160, copied: false),
            HistoryEntry(text: "The build is failing on the main branch because the Metal shader compiler can’t find the header file.",
                         date: now.addingTimeInterval(-3900), model: .nemotron560, copied: true, edited: true),
        ])
        let h = DictationController(model: .nemotron160, loader: StandInLoader(factory: StandInFactory()), microphone: SilentInput(),
                                    permission: Permission(granted: true), clipboard: NoClipboard(), history: saved)
        h.loadModel(); await settle()
        let scratch = UserDefaults(suiteName: "dev.universaldictator.snapshot")!
        scratch.removePersistentDomain(forName: "dev.universaldictator.snapshot")
        render(SettingsView(controller: h, direct: DirectSettings(defaults: scratch)), "7-settings", dir, background: false)
        await h.panelOpened(); h.pause(); await settle()
render(SettingsView(controller: h, direct: DirectSettings(defaults: scratch)), "8-settings-paused", dir, background: false)
        let combo = DirectSettings(defaults: scratch)
        combo.shortcut = .combination(keyCode: 0, carbonModifiers: 0x0200, display: "⇧A")
        combo.mode = .pushToTalk
        render(SettingsView(controller: h, direct: combo), "8b-settings-combination", dir, background: false)
        scratch.removePersistentDomain(forName: "dev.universaldictator.snapshot")
        render(HistoryView(controller: h).frame(width: 520, height: 400), "9-history", dir, background: false)
        let second = h.history[1]
        render(HistoryView(controller: h, editingID: second.id, draft: second.text).frame(width: 520, height: 400), "9b-history-editing", dir,
               background: false)
    }

    @MainActor static func render(_ c: DictationController, _ name: String, _ dir: URL) {
        render(PanelView(controller: c, close: {}, openSettings: {}), name, dir, background: true)
    }

    /// Renders a view to <dir>/<name>.png: on the panel's material, or on the window background.
    @MainActor static func render<V: View>(_ view: V, _ name: String, _ dir: URL, background: Bool) {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        let bg: NSView
        if background { bg = PanelBackground(content: host) } else {
            let plain = NSView(frame: host.frame); plain.wantsLayer = true
            plain.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
            plain.addSubview(host); bg = plain
        }
        bg.frame = host.frame
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
