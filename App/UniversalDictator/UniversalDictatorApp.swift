import AppKit
import ApplicationServices
import DictatorCore
import SwiftUI
import UniversalDictatorKit

@main
struct UniversalDictatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // A menu-bar app (LSUIElement). Its Settings and History windows are its own (MenuBarPanel); while one
        // is open the app shows its menus, and the app menu's Settings… opens that window.
        Settings { EmptyView() }
            .commands {
                CommandGroup(replacing: .appSettings) {
                    Button("Settings…") { appDelegate.menuBar?.showSettings() }.keyboardShortcut(",", modifiers: .command)
                }
            }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var controller: DictationController!
    var menuBar: MenuBarPanel!
    var direct: DirectSettings!
    var hotkeys: HotkeyCenter!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let assets = ModelAssets(directory: Bundle.main.resourceURL!.appendingPathComponent("NemotronAssets"))
        if DevModes.run(assets: assets) { return }
        let defaults = UserDefaults.standard
        let saved = defaults.string(forKey: "model").flatMap(ModelChoice.init(rawValue:)) ?? .nemotron160
        let limit = defaults.object(forKey: "historyLimit") as? Int ?? 50
        let microphone = Microphone()
        microphone.onRestart = { rate in
            log.notice("microphone input changed format; \(rate.map { "restarted at \(Int($0)) Hz" } ?? "restart failed", privacy: .public)")
        }
        controller = DictationController(model: saved, loader: EngineLoader(assets: assets), microphone: microphone,
                                         permission: SystemMicrophonePermission(), clipboard: SystemClipboard(),
                                         microphoneQueue: DispatchQueue(label: "microphone", qos: .userInitiated),
                                         history: FileHistoryStore.standard(bundleID: Bundle.main.bundleIdentifier ?? "dev.universaldictator.UniversalDictator"),
                                         historyLimit: limit)
        direct = DirectSettings()
        controller.enableDirectDictation(typer: KeyboardTyper(), accessibility: SystemAccessibility())
        menuBar = MenuBarPanel(controller: controller, direct: direct)
        hotkeys = HotkeyCenter(settings: direct)
        hotkeys.onDown = { [weak self] in self?.menuBar.shortcutDown() }
        hotkeys.onUp = { [weak self] in self?.menuBar.shortcutUp() }
        hotkeys.onTap = { [weak self] in self?.menuBar.shortcutTap() ?? false }
        if CommandLine.arguments.contains("--diagnose-start") {
            // Twice: open the panel and start a real dictation, time it, report the icon's position and whether
            // it is on screen while listening, close. The microphone runs for about three seconds each time.
            let tLoad = Date()
            controller.loadModel()
            Task { @MainActor in
                while self.controller.phase == .preparing { try? await Task.sleep(for: .milliseconds(100)) }
                FileHandle.standardError.write("model ready after \(Int(Date().timeIntervalSince(tLoad) * 1000)) ms\n".data(using: .utf8)!)
                try? await Task.sleep(for: .seconds(2))
                for round in 1...2 {
                    let t0 = Date()
                    self.menuBar.open()
                    FileHandle.standardError.write("round \(round) before open: \(self.menuBar.describeIcon())\n".data(using: .utf8)!)
                    while self.controller.phase != .listening { try? await Task.sleep(for: .milliseconds(5)) }
                    FileHandle.standardError.write("round \(round): open -> listening \(Int(Date().timeIntervalSince(t0) * 1000)) ms\n".data(using: .utf8)!)
                    // The microphone indicator appears a moment after the microphone starts. In round 1 the
                    // microphone also goes through its format-change restart halfway (as when a headset switches
                    // mode); the log shows whether audio kept arriving.
                    for i in 1...6 {
                        if round == 1, i == 3 {
                            microphone.restartForDiagnosis()
                            FileHandle.standardError.write("round 1: forced a microphone restart\n".data(using: .utf8)!)
                        }
                        try? await Task.sleep(for: .milliseconds(500))
                        FileHandle.standardError.write("round \(round) listening +\(i * 500) ms: audio blocks \(microphone.blocksDelivered); \(self.menuBar.describeIcon())\n".data(using: .utf8)!)
                    }
                    self.menuBar.close()
                    try? await Task.sleep(for: .milliseconds(800))
                }
                exit(0)
            }
            return
        }
        if CommandLine.arguments.contains("--diagnose-open") {
            // Shows the panel for a second without starting a dictation, and reports the app's on-screen windows.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                self.menuBar.open(startDictation: false)
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    let pid = ProcessInfo.processInfo.processIdentifier
                    let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
                    for w in list where (w[kCGWindowOwnerPID as String] as? Int32) == pid {
                        FileHandle.standardError.write("on screen: layer \(w[kCGWindowLayer as String] ?? "?") bounds \(w[kCGWindowBounds as String] ?? "?")\n".data(using: .utf8)!)
                    }
                    FileHandle.standardError.write("panel visible \(self.menuBar.panel.isVisible) key \(self.menuBar.panel.isKeyWindow) frame \(self.menuBar.panel.frame) mic phase \(self.controller.phase)\n".data(using: .utf8)!)
                    exit(0)
                }
            }
            return
        }
        if CommandLine.arguments.contains("--diagnose-panel") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                FileHandle.standardError.write((self.menuBar.describeGeometry() + "\n").data(using: .utf8)!)
                exit(0)
            }
            return
        }
        // Load at launch so a click starts listening straight away (the first launch after installing also
        // compiles the model for the Neural Engine).
        controller.loadModel()
        // Ask for the microphone now rather than on the first click: the system prompt takes focus, which
        // would close the panel mid-start.
        Task {
            await controller.requestMicrophoneAccessIfNeeded()
            // Direct dictation needs Accessibility: macOS's prompt, once.
            if !AXIsProcessTrusted(), !UserDefaults.standard.bool(forKey: "askedAccessibility") {
                UserDefaults.standard.set(true, forKey: "askedAccessibility")
                SystemAccessibility.prompt()
            }
        }
    }

    /// Quitting keeps a paused or unfinished dictation's text in the history.
    func applicationWillTerminate(_ notification: Notification) {
        controller?.finishForQuit()
    }
}
