import BsafeCore
import CoreImage
import CoreMedia
import Foundation
import ScreenCaptureKit

private struct PipelineStats {
    var frames: Int = 0
    var ingestMs: Double = 0
    var ageMs: Double = 0
    var sent: Int = 0
    var detApplied: Int = 0
    var detIgnored: Int = 0
    var catchupMs: Double = 0
    var catchupN: Int = 0
    var pixelMs: Double = 0
    var pixelN: Int = 0
}

/// Per-display live-v2 pipeline. All tracker/pipeline state lives on `queue`,
/// which is also the SCStream sampleHandlerQueue, so no locks are needed for
/// that state. Stats counters use a small lock for the utility-timer swap.
final class DisplayPipeline: NSObject, SCStreamOutput {
    let displayID: UInt32
    let queue: DispatchQueue
    let captureWidth: Int
    let captureHeight: Int
    let pointsWidth: Int
    let pointsHeight: Int
    let scaleFactor: Double
    let presentLead: Double
    let overlay: CensorOverlay
    let statsEnabled: Bool

    private let tracker: DisplayTracker
    private let ciContext: CIContext
    private let styleProvider: () -> CensorStyle
    private let frameSender: (UInt32, UInt32, UInt32, Int64, UInt32, Data) -> Void

    private var stream: SCStream?
    private var seq: UInt32 = 0
    private var latestBuffer: CVImageBuffer?
    private var latestSeq: UInt32?
    private var latestPts: Double?
    private var lastSentSeq: UInt32?
    private var creditPending = false
    private var recentPts: [(seq: UInt32, pts: Double)] = []

    private let statsLock = NSLock()
    private var stats = PipelineStats()

    init(
        displayID: UInt32,
        captureWidth: Int,
        captureHeight: Int,
        pointsWidth: Int,
        pointsHeight: Int,
        scaleFactor: Double,
        presentLead: Double,
        config: TrackerConfig,
        overlay: CensorOverlay,
        statsEnabled: Bool,
        styleProvider: @escaping () -> CensorStyle,
        frameSender: @escaping (UInt32, UInt32, UInt32, Int64, UInt32, Data) -> Void
    ) {
        self.displayID = displayID
        self.queue = DispatchQueue(label: "bsafe.display.\(displayID)")
        self.captureWidth = captureWidth
        self.captureHeight = captureHeight
        self.pointsWidth = pointsWidth
        self.pointsHeight = pointsHeight
        self.scaleFactor = scaleFactor
        self.presentLead = presentLead
        self.tracker = DisplayTracker(frameWidth: captureWidth, frameHeight: captureHeight, config: config)
        self.ciContext = CIContext()
        self.overlay = overlay
        self.statsEnabled = statsEnabled
        self.styleProvider = styleProvider
        self.frameSender = frameSender
        super.init()
    }

    /// Create and synchronously start one SCStream for `display`.
    /// Blocks until startCapture completes; throws on failure.
    func startStream(display: SCDisplay, fps: Int, excludingWindows: [SCWindow], delegate: SCStreamDelegate) throws {
        let filter = SCContentFilter(display: display, excludingWindows: excludingWindows)
        let config = SCStreamConfiguration()
        config.width = captureWidth
        config.height = captureHeight
        config.captureResolution = .best
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = false
        config.queueDepth = 6
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(fps))

        let st = SCStream(filter: filter, configuration: config, delegate: delegate)
        try st.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)

        let sem = DispatchSemaphore(value: 0)
        var captureError: Error?
        st.startCapture { err in
            captureError = err
            sem.signal()
        }
        sem.wait()
        if let captureError {
            throw captureError
        }
        stream = st
    }

    func stopStream() {
        stream?.stopCapture { _ in }
    }

    // MARK: - SCStreamOutput (runs on `queue`)

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .screen else { return }
        // Accept only complete frames.
        var complete = false
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as NSArray?,
           attachments.count > 0,
           let dict = attachments[0] as? NSDictionary
        {
            let wantKey = SCStreamFrameInfo.status.rawValue
            for (k, v) in dict {
                if let ks = k as? String, ks == wantKey {
                    if let num = v as? Int, num == SCFrameStatus.complete.rawValue {
                        complete = true
                    }
                    break
                }
            }
        }
        guard complete else { return }
        guard let imageBuffer = sampleBuffer.imageBuffer else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let hostNow = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        let w = CVPixelBufferGetWidth(imageBuffer)
        let h = CVPixelBufferGetHeight(imageBuffer)

        guard CVPixelBufferLockBaseAddress(imageBuffer, .readOnly) == kCVReturnSuccess else { return }
        guard let base = CVPixelBufferGetBaseAddress(imageBuffer) else {
            CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
            return
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)

        let t0 = CFAbsoluteTimeGetCurrent()
        let pyramid = LumaPyramid.fromBGRA(base, width: w, height: h, bytesPerRow: bytesPerRow)
        seq &+= 1
        let mySeq = seq
        tracker.ingestFrame(seq: mySeq, pts: pts, pyramid: pyramid)
        let ingestMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        var sendCopy: Data?
        if creditPending {
            sendCopy = copyPacked(base: base, width: w, height: h, bytesPerRow: bytesPerRow)
            creditPending = false
            lastSentSeq = mySeq
        }
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)

        latestBuffer = imageBuffer
        latestSeq = mySeq
        latestPts = pts
        recentPts.append((seq: mySeq, pts: pts))
        if recentPts.count > 96 {
            recentPts.removeFirst(recentPts.count - 96)
        }

        if statsEnabled {
            statsLock.lock()
            stats.frames += 1
            stats.ingestMs += ingestMs
            stats.ageMs += (hostNow - pts) * 1000.0
            statsLock.unlock()
        }

        if let pixels = sendCopy {
            let ptsNs = Int64((pts * 1_000_000_000.0).rounded())
            frameSender(displayID, UInt32(w), UInt32(h), ptsNs, mySeq, pixels)
            if statsEnabled {
                statsLock.lock()
                stats.sent += 1
                statsLock.unlock()
            }
        }

        publishRender()
    }

    /// Granted one frame credit (called via queue.async from the reader thread).
    func handleCredit() {
        if let buf = latestBuffer, let ls = latestSeq, ls != lastSentSeq {
            let w = CVPixelBufferGetWidth(buf)
            let h = CVPixelBufferGetHeight(buf)
            guard CVPixelBufferLockBaseAddress(buf, .readOnly) == kCVReturnSuccess else {
                creditPending = true
                return
            }
            guard let base = CVPixelBufferGetBaseAddress(buf) else {
                CVPixelBufferUnlockBaseAddress(buf, .readOnly)
                creditPending = true
                return
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buf)
            let pixels = copyPacked(base: base, width: w, height: h, bytesPerRow: bytesPerRow)
            CVPixelBufferUnlockBaseAddress(buf, .readOnly)
            lastSentSeq = ls
            if let pts = latestPts {
                let ptsNs = Int64((pts * 1_000_000_000.0).rounded())
                frameSender(displayID, UInt32(w), UInt32(h), ptsNs, ls, pixels)
                if statsEnabled {
                    statsLock.lock()
                    stats.sent += 1
                    statsLock.unlock()
                }
            }
        } else {
            creditPending = true
        }
    }

    /// Apply Python detections for frame `seq` (called via queue.async from the reader thread).
    func handleDetections(frameWidth: Int, frameHeight: Int, seq detSeq: UInt32, boxes: [TrackBox]) {
        guard frameWidth == captureWidth, frameHeight == captureHeight else {
            if statsEnabled {
                statsLock.lock()
                stats.detIgnored += 1
                statsLock.unlock()
            }
            return
        }
        let ok = tracker.applyDetections(seq: detSeq, boxes: boxes)
        if statsEnabled {
            statsLock.lock()
            if ok {
                stats.detApplied += 1
                if let lp = latestPts, let dp = ptsForSeq(detSeq) {
                    stats.catchupMs += (lp - dp) * 1000.0
                    stats.catchupN += 1
                }
            } else {
                stats.detIgnored += 1
            }
            statsLock.unlock()
        }
        publishRender()
    }

    /// Build the MSG_STATS line and reset counters. Called on the utility timer.
    func takeStatsLine() -> String? {
        guard statsEnabled else { return nil }
        statsLock.lock()
        let s = stats
        stats = PipelineStats()
        statsLock.unlock()
        let drains = overlay.takeDrainCount()
        let interval = 2.0
        func avg(_ sum: Double, _ n: Int) -> Double {
            return n > 0 ? sum / Double(n) : 0.0
        }
        let line = String(
            format: "display=%u fps=%.1f ingest_ms=%.2f age_ms=%.1f sent=%d det=%d/%d catchup_ms=%.1f render_hz=%.1f pixelate_ms=%.2f",
            displayID,
            Double(s.frames) / interval,
            avg(s.ingestMs, s.frames),
            avg(s.ageMs, s.frames),
            s.sent,
            s.detApplied,
            s.detIgnored,
            avg(s.catchupMs, s.catchupN),
            Double(drains) / interval,
            avg(s.pixelMs, s.pixelN)
        )
        return line
    }

    // MARK: - Private (all on `queue` unless noted)

    private func ptsForSeq(_ target: UInt32) -> Double? {
        for entry in recentPts.reversed() where entry.seq == target {
            return entry.pts
        }
        return nil
    }

    private func copyPacked(base: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) -> Data {
        let rowBytes = width * 4
        var out = Data(count: rowBytes * height)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let d = dst.baseAddress else { return }
            for row in 0..<height {
                let s = base.advanced(by: row * bytesPerRow)
                let dd = d.advanced(by: row * rowBytes)
                memcpy(dd, s, rowBytes)
            }
        }
        return out
    }

    private func publishRender() {
        let style = styleProvider()
        let now = CMClockGetTime(CMClockGetHostTimeClock()).seconds
        let boxes = tracker.renderBoxes(now: now, presentLead: presentLead)

        var rects: [CGRect] = []
        rects.reserveCapacity(boxes.count)
        for b in boxes {
            let x0 = floor(b.x)
            let y0 = floor(b.y)
            let x1 = ceil(b.x + b.w)
            let y1 = ceil(b.y + b.h)
            rects.append(CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
        }

        var images: [CGImage?] = Array(repeating: nil, count: rects.count)
        if style.pixels > 0.0, let buf = latestBuffer, !rects.isEmpty {
            let t0 = CFAbsoluteTimeGetCurrent()
            let cw = captureWidth
            let ch = captureHeight
            let blockSize = max(2.0, style.pixels * 15.0 * scaleFactor)
            if CVPixelBufferLockBaseAddress(buf, .readOnly) == kCVReturnSuccess {
                let full = CIImage(cvImageBuffer: buf)
                for i in 0..<rects.count {
                    var ix = Int(rects[i].origin.x)
                    var iy = Int(rects[i].origin.y)
                    var iw = Int(rects[i].width)
                    var ih = Int(rects[i].height)
                    if ix < 0 { iw += ix; ix = 0 }
                    if iy < 0 { ih += iy; iy = 0 }
                    if ix + iw > cw { iw = cw - ix }
                    if iy + ih > ch { ih = ch - iy }
                    guard iw > 0, ih > 0 else {
                        images[i] = nil
                        continue
                    }
                    // CoreImage origin is bottom-left.
                    let crop = CGRect(x: ix, y: ch - iy - ih, width: iw, height: ih)
                    // Clamp so edge blocks sample edge pixels instead of transparent (black on the layer).
                    let cropped = full.cropped(to: crop).clampedToExtent()
                    guard let filt = CIFilter(name: "CIPixellate") else {
                        images[i] = nil
                        continue
                    }
                    filt.setValue(cropped, forKey: kCIInputImageKey)
                    filt.setValue(blockSize, forKey: kCIInputScaleKey)
                    filt.setValue(CIVector(x: 0, y: 0), forKey: kCIInputCenterKey)
                    if let out = filt.outputImage, let cg = ciContext.createCGImage(out, from: crop) {
                        images[i] = cg
                    } else {
                        images[i] = nil
                    }
                }
                CVPixelBufferUnlockBaseAddress(buf, .readOnly)
            }
            if statsEnabled {
                statsLock.lock()
                stats.pixelMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                stats.pixelN += 1
                statsLock.unlock()
            }
        }

        overlay.publish(OverlayFrame(
            boxes: rects,
            images: images,
            captureWidth: captureWidth,
            captureHeight: captureHeight,
            blur: style.blur,
            pixels: style.pixels,
            text: style.text
        ))
    }
}
