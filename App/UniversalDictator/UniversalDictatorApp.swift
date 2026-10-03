import AppKit
import DictatorCore
import SwiftUI
import UniversalDictatorKit

@main
struct UniversalDictatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // A menu-bar-only app (LSUIElement): no windows besides the status item's panel.
        Settings { EmptyView() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var controller: DictationController!
    var menuBar: MenuBarPanel!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let assets = ModelAssets(directory: Bundle.main.resourceURL!.appendingPathComponent("NemotronAssets"))
        if DevModes.run(assets: assets) { return }
        let saved = UserDefaults.standard.string(forKey: "model").flatMap(ModelChoice.init(rawValue:)) ?? .nemotron160
        controller = DictationController(model: saved, loader: EngineLoader(assets: assets), microphone: Microphone(),
                                         permission: SystemMicrophonePermission(), clipboard: SystemClipboard())
        menuBar = MenuBarPanel(controller: controller)
        // Load at launch so a click starts listening straight away (the first launch after installing also
        // compiles the model for the Neural Engine).
        controller.loadModel()
        // Ask for the microphone now rather than on the first click: the system prompt takes focus, which
        // would close the panel mid-start.
        Task { await controller.requestMicrophoneAccessIfNeeded() }
    }
}
