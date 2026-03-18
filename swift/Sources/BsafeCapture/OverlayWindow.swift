import AppKit

/// NSView subclass that draws censored regions with optional blur and text.
class CensorView: NSView {
    var boxes: [NSRect] = []
    var blur: Double = 0.0
    var text: String? = nil
    private var effectViews: [NSVisualEffectView] = []

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        if blur <= 0.0 {
            NSColor.black.setFill()
            for box in boxes {
                box.fill()
            }
        }

        if let text {
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .center

            for box in boxes {
                let fontSize = min(box.height * 0.3, 48)
                let attrs: [NSAttributedString.Key: Any] = [
                    .foregroundColor: NSColor.white,
                    .font: NSFont.boldSystemFont(ofSize: fontSize),
                    .paragraphStyle: paragraphStyle,
                ]
                let nsText = text as NSString
                let textSize = nsText.size(withAttributes: attrs)
                let textRect = NSRect(
                    x: box.origin.x + (box.width - textSize.width) / 2,
                    y: box.origin.y + (box.height - textSize.height) / 2,
                    width: textSize.width,
                    height: textSize.height
                )
                nsText.draw(in: textRect, withAttributes: attrs)
            }
        }
    }

    /// Sync NSVisualEffectView subviews with current boxes when blur is enabled.
    func updateEffectViews() {
        // Remove old effect views
        for view in effectViews {
            view.removeFromSuperview()
        }
        effectViews.removeAll()

        guard blur > 0.0 else { return }

        for box in boxes {
            let effectView = NSVisualEffectView(frame: box)
            effectView.blendingMode = .behindWindow
            effectView.material = .fullScreenUI
            effectView.state = .active
            effectView.alphaValue = CGFloat(blur)
            addSubview(effectView)
            effectViews.append(effectView)
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
    func updateBoxes(_ boxes: [(x: Int32, y: Int32, w: Int32, h: Int32)], frameWidth: UInt32, frameHeight: UInt32, blur: Double = 0.0, text: String? = nil) {
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
            self?.censorView?.blur = blur
            self?.censorView?.text = text
            self?.censorView?.boxes = rects
            self?.censorView?.updateEffectViews()
            self?.censorView?.needsDisplay = true
        }
    }
}
