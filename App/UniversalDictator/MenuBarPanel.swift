import AppKit
import SwiftUI
import UniversalDictatorKit

/// The menu-bar item and the panel that drops down under it. Clicking the item toggles the panel; the
/// panel also closes when it loses focus, and after Copy.
@MainActor
final class MenuBarPanel: NSObject, NSWindowDelegate {
    let controller: DictationController
    let item: NSStatusItem
    let panel: DropDownPanel
    var host: NSHostingView<PanelView>!

    init(controller: DictationController) {
        self.controller = controller
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        panel = DropDownPanel()
        super.init()
        host = NSHostingView(rootView: PanelView(controller: controller, close: { [weak self] in self?.close() }))
        panel.contentView = PanelBackground(content: host)
        panel.delegate = self
        item.button?.target = self
        item.button?.action = #selector(toggle)
        item.button?.toolTip = "Universal Dictator"
        updateIcon()
    }

    /// When the panel last closed: clicking the menu-bar item while the panel is open first takes focus
    /// from the panel (closing it), and that same click must not reopen it.
    var lastClose = Date.distantPast

    @objc func toggle() {
        if panel.isVisible { close() } else if Date().timeIntervalSince(lastClose) > 0.3 { open() }
    }

    /// Where the panel goes: under the menu-bar icon, inside the screen.
    func panelFrame() -> NSRect? {
        guard let button = item.button, let window = button.window else { return nil }
        let iconFrame = window.convertToScreen(button.convert(button.bounds, to: nil))
        // The SwiftUI content's own size (the material view around it reports zero).
        var size = host.fittingSize
        if size.width < 100 || size.height < 100 { size = NSSize(width: 360, height: 238) }
        let screen = window.screen?.visibleFrame ?? NSScreen.main!.visibleFrame
        var x = iconFrame.midX - size.width / 2
        x = min(max(x, screen.minX + 8), screen.maxX - size.width - 8)
        return NSRect(x: x, y: iconFrame.minY - size.height - 6, width: size.width, height: size.height)
    }

    /// For diagnosing placement without showing anything: the icon, screen and panel frames.
    func describeGeometry() -> String {
        let button = item.button
        let iconFrame = button?.window.map { $0.convertToScreen(button!.convert(button!.bounds, to: nil)) }
        return "icon \(String(describing: iconFrame)), screen \(String(describing: button?.window?.screen?.visibleFrame)), " +
            "content fitting \(String(describing: host.fittingSize)), panel \(String(describing: panelFrame()))"
    }

    func open(startDictation: Bool = true) {
        guard let frame = panelFrame(), frame.width > 0, frame.height > 0 else { return }
        panel.setFrame(frame, display: true)
        // The panel takes keyboard focus without activating the app, like Spotlight: the app's menus never
        // replace the frontmost app's (which on a notched MacBook pushes status icons, ours included, behind
        // the notch), and closing the panel leaves the previous app in front, ready for ⌘V.
        panel.makeKeyAndOrderFront(nil)
        item.button?.highlight(true)
        guard startDictation else { return }
        Task { [weak self] in
            guard let self, self.panel.isVisible else { return }
            await self.controller.panelOpened()
        }
    }

    func close() {
        guard panel.isVisible else { return }
        panel.orderOut(nil)
        lastClose = Date()
        item.button?.highlight(false)
        controller.panelClosed()
    }

    func windowDidResignKey(_ notification: Notification) { close() }

    /// Keeps the menu-bar icon in step with the dictation (listening, paused, preparing).
    func updateIcon() {
        withObservationTracking {
            // Safety net: the microphone never runs while the panel is hidden.
            if controller.phase == .listening && !panel.isVisible {
                Task { @MainActor [weak self] in self?.controller.panelClosed() }
            }
            let name: String
            switch controller.phase {
            case .listening: name = "waveform.badge.mic"
            case .paused: name = "pause.circle"
            case .preparing: name = "hourglass"
            case .microphoneDenied, .failed: name = "exclamationmark.triangle"
            default: name = "waveform"
            }
            let image = NSImage(systemSymbolName: name, accessibilityDescription: "Universal Dictator")
            image?.isTemplate = true
            item.button?.image = image
        } onChange: {
            Task { @MainActor [weak self] in self?.updateIcon() }
        }
    }
}

/// A borderless panel that can take keyboard focus (for editing and shortcuts).
final class DropDownPanel: NSPanel {
    init() {
        super.init(contentRect: NSRect(x: 0, y: 0, width: 360, height: 260), styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
                   backing: .buffered, defer: true)
        becomesKeyOnlyIfNeeded = false
        isFloatingPanel = true
        level = .popUpMenu
        hasShadow = true
        backgroundColor = .clear
        isOpaque = false
        isMovable = false
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
    }
    override var canBecomeKey: Bool { true }
}

/// The menu material with rounded corners behind the SwiftUI content, resizing with it.
final class PanelBackground: NSVisualEffectView {
    init(content: NSView) {
        super.init(frame: .zero)
        material = .popover
        blendingMode = .behindWindow
        state = .active
        wantsLayer = true
        layer?.cornerRadius = 12
        layer?.masksToBounds = true
        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: leadingAnchor), content.trailingAnchor.constraint(equalTo: trailingAnchor),
            content.topAnchor.constraint(equalTo: topAnchor), content.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
}
