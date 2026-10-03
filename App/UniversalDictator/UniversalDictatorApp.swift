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
        Task { await controller.requestMicrophoneAccessIfNeeded() }
    }
}
