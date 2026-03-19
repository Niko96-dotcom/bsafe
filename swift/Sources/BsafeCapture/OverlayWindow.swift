import AppKit

/// NSView subclass that draws censored regions with optional blur, pixelation, and text.
class CensorView: NSView {
    var boxes: [NSRect] = []
    var blur: Double = 0.0
    var pixels: Double = 0.0
    var text: String? = nil
    /// AppKit screen frame origin for this display (used to convert to global CG coords for pixel capture).
    var screenOrigin: NSPoint = .zero
    private var effectViews: [NSVisualEffectView] = []
    private var pixelImages: [NSImage] = []
    private var cachedPixelBoxes: [NSRect] = []
    var ownerWindowNumber: Int = 0
    private let ciContext = CIContext()
    /// Movement threshold (points) before re-capturing a pixelated box.
    private let pixelCacheThreshold: CGFloat = 5.0

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        if pixels > 0.0 {
            // Draw cached pixelated images; fall back to black if capture failed
            for (i, box) in boxes.enumerated() {
                if i < pixelImages.count && pixelImages[i].size != .zero {
                    pixelImages[i].draw(in: box)
                } else {
                    NSColor.black.setFill()
                    box.fill()
                }
            }
        } else if blur <= 0.0 {
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
    /// When pixels is enabled, capture and pixelate screen regions behind the overlay.
    func updateEffectViews() {
        // Remove old effect views
        for view in effectViews {
            view.removeFromSuperview()
        }
        effectViews.removeAll()

        if pixels > 0.0 {
            // Reuse cached images when boxes haven't moved significantly
            let prevImages = pixelImages
            let prevBoxes = cachedPixelBoxes
            var newImages: [NSImage] = []
            let windowNum = CGWindowID(ownerWindowNumber)

            for (i, box) in boxes.enumerated() {
                // Check if we can reuse a cached image for this box
                if i < prevBoxes.count && i < prevImages.count && prevImages[i].size != .zero {
                    let prev = prevBoxes[i]
                    let dx = abs(box.origin.x - prev.origin.x)
                    let dy = abs(box.origin.y - prev.origin.y)
                    let dw = abs(box.width - prev.width)
                    let dh = abs(box.height - prev.height)
                    if dx < pixelCacheThreshold && dy < pixelCacheThreshold
                        && dw < pixelCacheThreshold && dh < pixelCacheThreshold {
                        newImages.append(prevImages[i])
                        continue
                    }
                }

                // Capture and pixelate this box
                // Convert view-local AppKit coords to global CG coords for CGWindowListCreateImage
                guard let primaryHeight = NSScreen.screens.first?.frame.height else {
                    newImages.append(NSImage())
                    continue
                }
                // AppKit global coords: add screen origin to view-local position
                let globalAppKitX = screenOrigin.x + box.origin.x
                let globalAppKitY = screenOrigin.y + box.origin.y
                // CG coords: flip Y using primary screen height
                let captureRect = CGRect(
                    x: globalAppKitX,
                    y: primaryHeight - globalAppKitY - box.height,
                    width: box.width,
                    height: box.height
                )
                guard let cgImage = CGWindowListCreateImage(
                    captureRect,
                    .optionOnScreenBelowWindow,
                    windowNum,
                    [.bestResolution]
                ) else {
                    newImages.append(NSImage())
                    continue
                }

                let inputScale = max(2.0, pixels * 30.0)
                let ciImage = CIImage(cgImage: cgImage)
                let filter = CIFilter(name: "CIPixellate")!
                filter.setValue(ciImage, forKey: kCIInputImageKey)
                filter.setValue(inputScale, forKey: kCIInputScaleKey)
                filter.setValue(CIVector(x: 0, y: 0), forKey: kCIInputCenterKey)

                if let output = filter.outputImage,
                   let rendered = ciContext.createCGImage(output, from: ciImage.extent) {
                    newImages.append(NSImage(cgImage: rendered, size: box.size))
                } else {
                    newImages.append(NSImage())
                }
            }

            pixelImages = newImages
            cachedPixelBoxes = boxes
            return
        }

        pixelImages.removeAll()
        cachedPixelBoxes.removeAll()

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
        // Use NSScreen.frame (AppKit coordinates, bottom-left origin) for window positioning.
        // CGDisplayBounds uses CG coordinates (top-left origin) which mispositions the window
        // on secondary displays.
        let screenFrame: NSRect
        if let match = NSScreen.screens.first(where: {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) == displayID
        }) {
            screenFrame = match.frame
        } else {
            // Fallback: convert CGDisplayBounds CG coords → AppKit coords
            let cgBounds = CGDisplayBounds(displayID)
            let primaryHeight = NSScreen.screens.first?.frame.height ?? cgBounds.height
            screenFrame = NSRect(
                x: cgBounds.origin.x,
                y: primaryHeight - cgBounds.origin.y - cgBounds.height,
                width: cgBounds.width,
                height: cgBounds.height
            )
        }
        displayBounds = screenFrame

        let window = NSWindow(
            contentRect: screenFrame,
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

        let view = CensorView(frame: NSRect(origin: .zero, size: screenFrame.size))
        view.screenOrigin = screenFrame.origin
        window.contentView = view

        window.orderFrontRegardless()

        self.window = window
        self.censorView = view
    }

    /// Update overlay boxes. Converts pixel-space coordinates to display points and flips Y axis.
    /// Clamps each box so it does not exceed 80% of the display in either dimension.
    /// Can be called from any thread — dispatches to main.
    func updateBoxes(_ boxes: [(x: Int32, y: Int32, w: Int32, h: Int32)], frameWidth: UInt32, frameHeight: UInt32, blur: Double = 0.0, pixels: Double = 0.0, text: String? = nil) {
        let displayW = displayBounds.width
        let displayH = displayBounds.height
        let scaleX = displayW / CGFloat(frameWidth)
        let scaleY = displayH / CGFloat(frameHeight)

        let maxW = displayW * 0.8
        let maxH = displayH * 0.8

        // Convert pixel-space boxes to display-point NSRects with Y-axis flip
        // ScreenCaptureKit uses top-left origin, AppKit uses bottom-left origin
        let rects = boxes.map { box -> NSRect in
            let x = CGFloat(box.x) * scaleX
            let y = CGFloat(box.y) * scaleY
            let w = min(CGFloat(box.w) * scaleX, maxW)
            let h = min(CGFloat(box.h) * scaleY, maxH)
            // Flip Y: AppKit origin is bottom-left
            let flippedY = displayH - y - h
            return NSRect(x: x, y: flippedY, width: w, height: h)
        }

        DispatchQueue.main.async { [weak self] in
            self?.censorView?.blur = blur
            self?.censorView?.pixels = pixels
            self?.censorView?.text = text
            if let windowNumber = self?.window?.windowNumber {
                self?.censorView?.ownerWindowNumber = windowNumber
            }
            self?.censorView?.boxes = rects
            self?.censorView?.updateEffectViews()
            self?.censorView?.needsDisplay = true
        }
    }
}
