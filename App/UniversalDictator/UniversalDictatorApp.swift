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
