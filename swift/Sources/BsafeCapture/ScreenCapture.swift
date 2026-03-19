import Foundation
import ScreenCaptureKit

class ScreenCapture: NSObject, SCStreamDelegate {
    let fps: Int
    let displayFilter: String?
    var onFrame: ((CGImage, UInt32, TimeInterval) -> Void)?

    private var streams: [(displayID: UInt32, stream: SCStream, output: DisplayStreamOutput)] = []

    init(fps: Int, displayFilter: String? = nil) {
        self.fps = fps
        self.displayFilter = displayFilter
    }

    func start() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?

        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
            if let error {
                startError = error
                semaphore.signal()
                return
            }

            guard let content, !content.displays.isEmpty else {
                startError = NSError(
                    domain: "BsafeCapture", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No displays found"]
                )
                semaphore.signal()
                return
            }

            // Filter displays based on displayFilter
            let filteredDisplays: [SCDisplay]
            let primaryID = CGMainDisplayID()
            let filter = self.displayFilter ?? "primary"

            switch filter {
            case "primary":
                if let primary = content.displays.first(where: { $0.displayID == primaryID }) {
                    filteredDisplays = [primary]
                } else {
                    // No primary found; fall back to first available display
                    filteredDisplays = [content.displays[0]]  // safe: displays.isEmpty checked above
                }
            case "secondary":
                let nonPrimary = content.displays.filter { $0.displayID != primaryID }
                guard !nonPrimary.isEmpty else {
                    startError = NSError(
                        domain: "BsafeCapture", code: 3,
                        userInfo: [NSLocalizedDescriptionKey: "No secondary display found"]
                    )
                    semaphore.signal()
                    return
                }
                filteredDisplays = [nonPrimary[0]]
            case "all":
                filteredDisplays = content.displays
            default:
                // Numeric display ID
                if let numericID = UInt32(filter) {
                    guard let match = content.displays.first(where: { $0.displayID == numericID }) else {
                        startError = NSError(
                            domain: "BsafeCapture", code: 4,
                            userInfo: [NSLocalizedDescriptionKey: "Display with ID \(filter) not found"]
                        )
                        semaphore.signal()
                        return
                    }
                    filteredDisplays = [match]
                } else {
                    startError = NSError(
                        domain: "BsafeCapture", code: 5,
                        userInfo: [NSLocalizedDescriptionKey: "Invalid display filter: \(filter)"]
                    )
                    semaphore.signal()
                    return
                }
            }

            let captureGroup = DispatchGroup()

            for display in filteredDisplays {
                let filter = SCContentFilter(display: display, excludingWindows: [])
                let config = SCStreamConfiguration()
                config.width = display.width
                config.height = display.height
                config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(self.fps))
                config.pixelFormat = kCVPixelFormatType_32BGRA
                config.showsCursor = false
                config.queueDepth = 3

                let output = DisplayStreamOutput(displayID: display.displayID) { [weak self] cgImage, displayID, timestamp in
                    self?.onFrame?(cgImage, displayID, timestamp)
                }

                do {
                    let stream = SCStream(filter: filter, configuration: config, delegate: self)
                    try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: .global())
                    captureGroup.enter()
                    stream.startCapture { captureError in
                        if let captureError {
                            fputs("BsafeCapture: failed to start capture for display \(display.displayID): \(captureError)\n", stderr)
                        }
                        captureGroup.leave()
                    }
                    self.streams.append((displayID: display.displayID, stream: stream, output: output))
                } catch {
                    fputs("BsafeCapture: failed to create stream for display \(display.displayID): \(error)\n", stderr)
                }
            }

            captureGroup.notify(queue: .global()) {
                if self.streams.isEmpty {
                    startError = NSError(
                        domain: "BsafeCapture", code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "Failed to start capture on any display"]
                    )
                }
                semaphore.signal()
            }
        }

        semaphore.wait()

        if let startError {
            throw startError
        }

        let ids = streams.map { String($0.displayID) }.joined(separator: ", ")
        print("BsafeCapture: capturing \(streams.count) display(s): [\(ids)]")
    }

    func stop() {
        for entry in streams {
            entry.stream.stopCapture { _ in }
        }
        streams.removeAll()
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("BsafeCapture: stream stopped with error: \(error)\n", stderr)
    }
}

// MARK: - Per-display stream output

/// Each display gets its own SCStreamOutput handler so frames carry the correct displayID.
class DisplayStreamOutput: NSObject, SCStreamOutput {
    let displayID: UInt32
    private let ciContext = CIContext()
    private let handler: (CGImage, UInt32, TimeInterval) -> Void

    init(displayID: UInt32, handler: @escaping (CGImage, UInt32, TimeInterval) -> Void) {
        self.displayID = displayID
        self.handler = handler
    }

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .screen else { return }
        guard let imageBuffer = sampleBuffer.imageBuffer else { return }

        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let width = CVPixelBufferGetWidth(imageBuffer)
        let height = CVPixelBufferGetHeight(imageBuffer)

        guard
            let cgImage = ciContext.createCGImage(
                ciImage, from: CGRect(x: 0, y: 0, width: width, height: height))
        else { return }

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        handler(cgImage, displayID, timestamp)
    }
}
