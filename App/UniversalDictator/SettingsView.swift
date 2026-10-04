import Carbon
import DictatorCore
import SwiftUI
import UniversalDictatorKit

/// The Settings window (design: Figma "Settings (draft)"): labels on the left, controls on the right.
struct SettingsView: View {
    @Bindable var controller: DictationController
    @Bindable var direct: DirectSettings
    @State var confirmingClear = false

    var body: some View {
        VStack(spacing: 12) {
            row("Model") {
                Picker("Model", selection: Binding(get: { controller.model }, set: { choice in
                    controller.selectModel(choice)
                    UserDefaults.standard.set(choice.rawValue, forKey: "model")
                })) {
                    ForEach(ModelChoice.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                .disabled(!controller.canChangeModel)
                .help(modelHelp)
            }
            row("History") {
                Button("Clear History…") { confirmingClear = true }
                    .disabled(controller.history.isEmpty)
                Picker("History", selection: Binding(get: { controller.historyLimit }, set: { limit in
                    controller.historyLimit = limit
                    UserDefaults.standard.set(limit, forKey: "historyLimit")
                })) {
                    ForEach(DictationController.historyLimits, id: \.self) { Text("\($0)").tag($0) }
                }
                .labelsHidden()
                .fixedSize()
            }
            Divider()
            row("Shortcut") { ShortcutField(settings: direct) }
            row("Mode") {
                Picker("Mode", selection: $direct.mode) {
                    ForEach(DirectMode.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
            }
            row("Accessibility") {
                if direct.accessibilityGranted {
                    Text("Allowed ✓").foregroundStyle(.secondary)
                } else {
                    Button("Allow…") { controller.openAccessibilitySettings() }
                        .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 22)
        .frame(width: 360)
        .alert("Delete all \(controller.history.count) dictations?", isPresented: $confirmingClear) {
            Button("Delete", role: .destructive) { controller.clearHistory() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This can’t be undone.")
        }
    }

    func row<Controls: View>(_ label: String, @ViewBuilder controls: () -> Controls) -> some View {
        HStack(spacing: 8) {
            Text(label)
            Spacer()
            controls()
        }
    }

    /// Why the Model menu is disabled; nothing otherwise (the choices are just Fast and Accurate).
    var modelHelp: String {
        switch controller.phase {
        case .paused: return "A dictation is paused. Stop it or copy it to change models."
        case .listening: return "Stop the dictation to change models."
        case .preparing: return "The model is getting ready."
        default: return ""
        }
    }
}

/// The shortcut field: shows the shortcut; click it, then double-tap a modifier key or press a key combination
/// (with ⌘, ⌃ or ⌥, or a function key). Esc cancels.
struct ShortcutField: View {
    @Bindable var settings: DirectSettings
    @State var monitor: Any?
    @State var detectors: [ModifierKey: DoubleTapDetector] = [:]

    var body: some View {
        Button(action: { settings.isRecording ? stop() : start() }) {
            Text(settings.isRecording ? "Press keys…" : settings.shortcut.display)
                .foregroundStyle(settings.isRecording ? .secondary : .primary)
                .frame(minWidth: 100)
        }
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.accentColor, lineWidth: settings.isRecording ? 2 : 0))
        .onDisappear { stop() }
    }

    func start() {
        settings.isRecording = true
        detectors = [:]
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { e in
            handle(e) ? nil : e
        }
    }

    func stop() {
        if let m = monitor { NSEvent.removeMonitor(m) }
        monitor = nil
        settings.isRecording = false
    }

    /// True if the event was used.
    func handle(_ e: NSEvent) -> Bool {
        if e.type == .flagsChanged {
            guard let key = ModifierKey(keyCode: e.keyCode) else { return false }
            var d = detectors[key] ?? DoubleTapDetector(key: key)
            let event = HotkeyCenter.detectorEvent(e)
            for other in detectors.keys where other != key { _ = detectors[other]?.handle(event) }
            if d.handle(event) == .doubleTapDown {
                settings.shortcut = .doubleTap(key)
                stop()
            }
            detectors[key] = d
            return true
        }
        let flags = e.modifierFlags.intersection([.command, .control, .option, .shift])
        if e.keyCode == 53, flags.isEmpty { stop(); return true }                   // Esc
        let fKeys: Set<UInt16> = [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 64, 79, 80]
        guard !flags.intersection([.command, .control, .option]).isEmpty || fKeys.contains(e.keyCode) else { NSSound.beep(); return true }
        var carbon: UInt32 = 0
        if flags.contains(.command) { carbon |= UInt32(cmdKey) }
        if flags.contains(.option) { carbon |= UInt32(optionKey) }
        if flags.contains(.control) { carbon |= UInt32(controlKey) }
        if flags.contains(.shift) { carbon |= UInt32(shiftKey) }
        var symbols = ""
        if flags.contains(.control) { symbols += "⌃" }
        if flags.contains(.option) { symbols += "⌥" }
        if flags.contains(.shift) { symbols += "⇧" }
        if flags.contains(.command) { symbols += "⌘" }
        settings.shortcut = .combination(keyCode: UInt32(e.keyCode), carbonModifiers: carbon, display: symbols + Self.keyName(e))
        stop()
        return true
    }

    static func keyName(_ e: NSEvent) -> String {
        let named: [UInt16: String] = [49: "Space", 36: "↩", 48: "⇥", 51: "⌫", 122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5",
                                       97: "F6", 98: "F7", 100: "F8", 101: "F9", 109: "F10", 103: "F11", 111: "F12", 105: "F13",
                                       107: "F14", 113: "F15", 106: "F16", 64: "F17", 79: "F18", 80: "F19"]
        if let n = named[e.keyCode] { return n }
        return (e.charactersIgnoringModifiers ?? "?").uppercased()
    }
}
