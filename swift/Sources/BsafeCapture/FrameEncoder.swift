import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum FrameEncoder {
    /// Compress a CGImage to JPEG data at the given quality (0.0–1.0).
    static func encode(_ image: CGImage, quality: Double) -> Data? {
        let data = NSMutableData()
        guard
            let dest = CGImageDestinationCreateWithData(
                data, UTType.jpeg.identifier as CFString, 1, nil)
        else {
            return nil
        }
        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: quality
        ]
        CGImageDestinationAddImage(dest, image, options as CFDictionary)
        guard CGImageDestinationFinalize(dest) else {
            return nil
        }
        return data as Data
    }
}
