import Foundation
import ScreenCaptureKit

class ScreenCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    let fps: Int
    var onFrame: ((CGImage, UInt32, TimeInterval) -> Void)?

    private var stream: SCStream?
    private var displayID: UInt32 = 0
    private let ciContext = CIContext()

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

            // TODO: support multiple displays — currently captures only the first
            guard let content, let display = content.displays.first else {
                startError = NSError(
                    domain: "BsafeCapture", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No displays found"]
                )
                semaphore.signal()
                return
            }

            self.displayID = display.displayID

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let config = SCStreamConfiguration()
            config.width = display.width
            config.height = display.height
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(self.fps))
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.showsCursor = false
            config.queueDepth = 3

            do {
                let stream = SCStream(filter: filter, configuration: config, delegate: self)
                try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: .global())
                stream.startCapture { captureError in
                    if let captureError {
                        startError = captureError
                    }
                    semaphore.signal()
                }
                self.stream = stream
            } catch {
                startError = error
                semaphore.signal()
            }
        }

        semaphore.wait()

        if let startError {
            throw startError
        }

        print("BsafeCapture: capturing display \(displayID)")
    }

    func stop() {
        stream?.stopCapture { _ in }
        stream = nil
    }

    // MARK: - SCStreamOutput

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
        onFrame?(cgImage, displayID, timestamp)
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("BsafeCapture: stream stopped with error: \(error)\n", stderr)
    }
}
