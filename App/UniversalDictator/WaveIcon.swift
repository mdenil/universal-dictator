import AppKit

/// The menu-bar icon while the microphone is live: the waveform's six bars, in the menu bar's own colour, with
/// a wave flowing left to right through them and a little uneven movement in each bar.
enum WaveIcon {
    static let size = NSSize(width: 15, height: 16)
    static let bars = 6

    /// Bar heights at time `t` seconds, as a fraction of the icon's height.
    static func heights(at t: Double) -> [CGFloat] {
        (0..<bars).map { i in
            let x = Double(i) / Double(bars - 1), k = Double(i)
            let wave = 0.5 + 0.5 * sin(2 * .pi * (0.9 * t - x))
            // Three sines per bar at unrelated speeds (the fastest a flicker of 3–4.5 a second): never visibly repeats.
            let a = sin(2 * .pi * (1.1 + 0.2 * k) * t + k * 1.9)
            let b = sin(2 * .pi * (0.6 + 0.11 * k) * t + k * 0.7)
            let c = sin(2 * .pi * (2.6 + 0.37 * k) * t + k * 2.3)
            let noise = 0.5 + 0.5 * (0.4 * a + 0.25 * b + 0.35 * c)
            return CGFloat(0.3 + 0.55 * (0.65 * wave + 0.35 * noise))
        }
    }

    /// One frame: a template image, so macOS draws it in the menu bar's colour.
    static func image(_ heights: [CGFloat]) -> NSImage {
        let image = NSImage(size: size, flipped: false) { r in
            let w: CGFloat = 1.2
            let pitch = (r.width - w) / CGFloat(heights.count - 1)
            NSColor.black.setFill()
            for (i, h) in heights.enumerated() {
                let bh = max(w, h * r.height)
                let bar = NSRect(x: CGFloat(i) * pitch, y: (r.height - bh) / 2, width: w, height: bh)
                NSBezierPath(roundedRect: bar, xRadius: w / 2, yRadius: w / 2).fill()
            }
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "Universal Dictator, listening"
        return image
    }
}
