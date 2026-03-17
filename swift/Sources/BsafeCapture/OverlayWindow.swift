import AppKit

/// NSView subclass that draws black rectangles for censored regions.
class CensorView: NSView {
    var boxes: [NSRect] = []

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        NSColor.black.setFill()
        for box in boxes {
            box.fill()
        }
    }
}

/// Creates a borderless, transparent, always-on-top overlay window for censoring.
class CensorOverlay {
    private var window: NSWindow?
    private var censorView: CensorView?
    private var displayBounds: CGRect = .zero

    /// Create the overlay window covering the given display. Must be called on the main thread.
    func setup(displayID: CGDirectDisplayID) {
        displayBounds = CGDisplayBounds(displayID)

        let window = NSWindow(
            contentRect: displayBounds,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )
        window.level = .screenSaver
        window.isOpaque = false
        window.backgroundColor = .clear
        window.ignoresMouseEvents = true
        window.sharingType = .none  // excluded from ScreenCaptureKit — prevents feedback loop
        window.collectionBehavior = [.canJoinAllSpaces, .stationary]
        window.hasShadow = false

        let view = CensorView(frame: displayBounds)
        window.contentView = view

        window.orderFrontRegardless()

        self.window = window
        self.censorView = view
    }

    /// Update overlay boxes. Converts pixel-space coordinates to display points and flips Y axis.
    /// Can be called from any thread — dispatches to main.
    func updateBoxes(_ boxes: [(x: Int32, y: Int32, w: Int32, h: Int32)], frameWidth: UInt32, frameHeight: UInt32) {
        let displayW = displayBounds.width
        let displayH = displayBounds.height
        let scaleX = displayW / CGFloat(frameWidth)
        let scaleY = displayH / CGFloat(frameHeight)

        // Convert pixel-space boxes to display-point NSRects with Y-axis flip
        // ScreenCaptureKit uses top-left origin, AppKit uses bottom-left origin
        let rects = boxes.map { box -> NSRect in
            let x = CGFloat(box.x) * scaleX
            let y = CGFloat(box.y) * scaleY
            let w = CGFloat(box.w) * scaleX
            let h = CGFloat(box.h) * scaleY
            // Flip Y: AppKit origin is bottom-left
            let flippedY = displayH - y - h
            return NSRect(x: displayBounds.origin.x + x, y: displayBounds.origin.y + flippedY, width: w, height: h)
        }

        DispatchQueue.main.async { [weak self] in
            self?.censorView?.boxes = rects
            self?.censorView?.needsDisplay = true
        }
    }
}
