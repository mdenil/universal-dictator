import Carbon
import DictatorCore
import SwiftUI
import UniversalDictatorKit

/// The Settings window (design: Figma "Settings (draft)"): labels on the left, controls on the right.
struct SettingsView: View {
    @Bindable var controller: DictationController
    @Bindable var direct: DirectSettings
    @State var confirmingClear = false
    /// Key combination chosen but none recorded yet: the field records straight away.
    @State var choosingCombination = false
    @State var recordCombination = false

    enum ShortcutKind { case doubleTap, combination }

    var kind: ShortcutKind { direct.shortcut.isDoubleTap && !choosingCombination ? .doubleTap : .combination }

    var kindBinding: Binding<ShortcutKind> {
        Binding(get: { kind }, set: { new in
            switch new {
            case .doubleTap:
                choosingCombination = false
                if !direct.shortcut.isDoubleTap { direct.shortcut = .doubleTap(direct.lastDoubleTapKey) }
            case .combination:
                if let c = direct.lastCombination { direct.shortcut = c } else { choosingCombination = true; recordCombination = true }
            }
        })
    }

    var doubleTapKey: Binding<ModifierKey> {
        Binding(get: { if case .doubleTap(let k) = direct.shortcut { return k }; return direct.lastDoubleTapKey },
                set: { direct.shortcut = .doubleTap($0) })
    }

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
            row("Shortcut") {
                Picker("Shortcut", selection: kindBinding) {
                    Text("Double tap").tag(ShortcutKind.doubleTap)
                    Text("Key combination").tag(ShortcutKind.combination)
                }
                .labelsHidden()
                .fixedSize()
            }
            if kind == .doubleTap {
                row("Key") {
                    Picker("Key", selection: doubleTapKey) {
                        ForEach(ModifierKey.allCases, id: \.self) { Text($0.display).tag($0) }
                    }
                    .labelsHidden()
                    .fixedSize()
                }
            } else {
                row("Keys") { CombinationField(settings: direct, recordNow: $recordCombination) }
                if let w = direct.shortcut.warning, !direct.isRecording {
                    Text(w).font(.system(size: 11)).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
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
        // Recording ended: the menu follows the shortcut in force (a cancelled first recording leaves the double tap).
        .onChange(of: direct.isRecording) { if !direct.isRecording { choosingCombination = false } }
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

/// The key-combination field: shows the combination; click it (or it starts by itself when Key combination is
/// first chosen), then press the keys: any key with ⌘, ⌃, ⌥ or ⇧, or a function key on its own. Esc cancels.
struct CombinationField: View {
    @Bindable var settings: DirectSettings
    @Binding var recordNow: Bool
    @State var monitor: Any?

    var label: String {
        if settings.isRecording { return "Press keys…" }
        if case .combination = settings.shortcut { return settings.shortcut.display }
        return "Click to set"
    }

    var body: some View {
        Button(action: { settings.isRecording ? stop() : start() }) {
            Text(label)
                .foregroundStyle(settings.isRecording ? .secondary : .primary)
                .frame(minWidth: 90)
        }
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.accentColor, lineWidth: settings.isRecording ? 2 : 0))
        .onAppear { if recordNow { recordNow = false; start() } }
        .onChange(of: recordNow) { if recordNow { recordNow = false; start() } }
        .onDisappear { stop() }
    }

    func start() {
        guard monitor == nil else { return }
        settings.isRecording = true
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { e in handle(e) ? nil : e }
    }

    func stop() {
        if let m = monitor { NSEvent.removeMonitor(m) }
        monitor = nil
        settings.isRecording = false
    }

    /// True if the event was used.
    func handle(_ e: NSEvent) -> Bool {
        let flags = e.modifierFlags.intersection([.command, .control, .option, .shift])
        if e.keyCode == 53, flags.isEmpty { stop(); return true }                   // Esc
        let fKeys: Set<UInt16> = [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 64, 79, 80]
        guard !flags.isEmpty || fKeys.contains(e.keyCode) else { NSSound.beep(); return true }
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
        let shortcut = DictationShortcut.combination(keyCode: UInt32(e.keyCode), carbonModifiers: carbon, display: symbols + Self.keyName(e))
        settings.shortcut = shortcut
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
