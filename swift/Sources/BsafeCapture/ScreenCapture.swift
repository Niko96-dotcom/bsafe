import Foundation
import ScreenCaptureKit

class ScreenCapture: NSObject, SCStreamDelegate {
    let fps: Int
    var onFrame: ((CGImage, UInt32, TimeInterval) -> Void)?

    private var streams: [(displayID: UInt32, stream: SCStream, output: DisplayStreamOutput)] = []

    init(fps: Int) {
        self.fps = fps
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

            for display in content.displays {
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
                    stream.startCapture { captureError in
                        if let captureError {
                            fputs("BsafeCapture: failed to start capture for display \(display.displayID): \(captureError)\n", stderr)
                        }
                    }
                    self.streams.append((displayID: display.displayID, stream: stream, output: output))
                } catch {
                    fputs("BsafeCapture: failed to create stream for display \(display.displayID): \(error)\n", stderr)
                }
            }

            if self.streams.isEmpty {
                startError = NSError(
                    domain: "BsafeCapture", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to start capture on any display"]
                )
            }

            semaphore.signal()
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
