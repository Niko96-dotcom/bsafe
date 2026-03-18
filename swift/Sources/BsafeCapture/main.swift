import AppKit
import Foundation

// MARK: - Argument parsing

func parseArgs() -> (socketPath: String, fps: Int) {
    let args = CommandLine.arguments
    var socketPath = ""
    var fps = 3

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--socket":
            i += 1
            guard i < args.count else {
                fputs("Error: --socket requires a path\n", stderr)
                exit(1)
            }
            socketPath = args[i]
        case "--fps":
            i += 1
            guard i < args.count, let f = Int(args[i]), f > 0 else {
                fputs("Error: --fps requires a positive integer\n", stderr)
                exit(1)
            }
            fps = f
        default:
            fputs("Unknown argument: \(args[i])\n", stderr)
            exit(1)
        }
        i += 1
    }

    guard !socketPath.isEmpty else {
        fputs("Error: --socket is required\n", stderr)
        exit(1)
    }

    return (socketPath, fps)
}

// MARK: - Censor payload parsing

/// Parse CMD_CENSOR payload: [4B display_id][4B frame_width][4B frame_height][2B box_count][boxes...]
/// Each box: [4B x (int32)][4B y (int32)][4B w (int32)][4B h (int32)]
func parseCensorPayload(_ data: Data) -> (displayID: UInt32, frameWidth: UInt32, frameHeight: UInt32, boxes: [(x: Int32, y: Int32, w: Int32, h: Int32)])? {
    let headerSize = 14  // 4 + 4 + 4 + 2
    guard data.count >= headerSize else { return nil }

    let displayID = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).bigEndian }
    let frameWidth = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self).bigEndian }
    let frameHeight = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self).bigEndian }
    let boxCount = data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: UInt16.self).bigEndian }

    let expectedSize = headerSize + Int(boxCount) * 16
    guard data.count >= expectedSize else { return nil }

    var boxes: [(x: Int32, y: Int32, w: Int32, h: Int32)] = []
    for i in 0..<Int(boxCount) {
        let offset = headerSize + i * 16
        let x = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int32.self).bigEndian }
        let y = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Int32.self).bigEndian }
        let w = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Int32.self).bigEndian }
        let h = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: Int32.self).bigEndian }
        boxes.append((x, y, w, h))
    }

    return (displayID, frameWidth, frameHeight, boxes)
}

// MARK: - Main

let (socketPath, requestedFps) = parseArgs()

print("BsafeCapture: connecting to \(socketPath), requested FPS=\(requestedFps)")

let client = SocketClient(path: socketPath)

do {
    try client.connect()
} catch {
    fputs("Failed to connect to socket: \(error)\n", stderr)
    exit(1)
}

// Read CMD_START from Python
let cmdStart: (type: UInt8, payload: Data)
do {
    cmdStart = try client.readMessage()
} catch {
    fputs("Failed to read CMD_START: \(error)\n", stderr)
    exit(1)
}
let (msgType, payload) = cmdStart
guard msgType == 0x10 else {
    fputs("Expected CMD_START (0x10), got 0x\(String(msgType, radix: 16))\n", stderr)
    exit(1)
}

let fps: Int
if let firstByte = payload.first {
    fps = Int(firstByte)
    print("BsafeCapture: server requested FPS=\(fps)")
} else {
    fps = requestedFps
}

// Initialize NSApplication for overlay windows (no dock icon)
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

// Overlay dictionary: one CensorOverlay per display, keyed by displayID
var overlays: [UInt32: CensorOverlay] = [:]
let overlaysLock = NSLock()

let capture = ScreenCapture(fps: fps)

// JPEG quality matters for detection accuracy — 0.9 was validated empirically but
// 0.7 may be a good bandwidth/accuracy tradeoff. Test and adjust if needed.
let jpegQuality = 0.7

// Set up frame callback: encode and send
capture.onFrame = { cgImage, displayID, timestamp in
    // Ensure overlay exists for this display. Create on main thread (sync) if needed.
    overlaysLock.lock()
    let hasOverlay = overlays[displayID] != nil
    overlaysLock.unlock()

    if !hasOverlay {
        DispatchQueue.main.sync {
            overlaysLock.lock()
            // Double-check after acquiring lock on main thread
            if overlays[displayID] == nil {
                let overlay = CensorOverlay()
                overlay.setup(displayID: displayID)
                overlays[displayID] = overlay
            }
            overlaysLock.unlock()
        }
    }

    guard let jpegData = FrameEncoder.encode(cgImage, quality: jpegQuality) else {
        fputs("Failed to encode frame\n", stderr)
        return
    }

    let width = UInt32(cgImage.width)
    let height = UInt32(cgImage.height)
    let tsNs = Int64(timestamp * 1_000_000_000)

    var meta = Data()
    meta.append(contentsOf: withUnsafeBytes(of: displayID.bigEndian) { Array($0) })
    meta.append(contentsOf: withUnsafeBytes(of: width.bigEndian) { Array($0) })
    meta.append(contentsOf: withUnsafeBytes(of: height.bigEndian) { Array($0) })
    meta.append(contentsOf: withUnsafeBytes(of: tsNs.bigEndian) { Array($0) })

    let payload = meta + jpegData
    do {
        try client.sendMessage(type: 0x01, payload: payload)
    } catch {
        fputs("Failed to send frame: \(error)\n", stderr)
    }
}

// Start capturing
do {
    try capture.start()
} catch {
    fputs("Failed to start capture: \(error)\n", stderr)
    exit(1)
}

print("BsafeCapture: capturing at \(fps) FPS. Waiting for commands...")

// Listen for commands from Python in background
DispatchQueue.global().async {
    while true {
        do {
            let (msgType, payload) = try client.readMessage()
            switch msgType {
            case 0x20:  // CMD_CENSOR
                guard let censor = parseCensorPayload(payload) else {
                    fputs("BsafeCapture: failed to parse CMD_CENSOR payload\n", stderr)
                    continue
                }
                overlaysLock.lock()
                let overlay = overlays[censor.displayID]
                overlaysLock.unlock()
                if let overlay {
                    overlay.updateBoxes(censor.boxes, frameWidth: censor.frameWidth, frameHeight: censor.frameHeight)
                } else {
                    fputs("BsafeCapture: no overlay for display \(censor.displayID), skipping censor\n", stderr)
                }
            case 0xFF:  // CMD_SHUTDOWN
                print("BsafeCapture: received shutdown command")
                capture.stop()
                client.disconnect()
                exit(0)
            default:
                break
            }
        } catch {
            // Connection lost
            print("BsafeCapture: connection lost, shutting down")
            capture.stop()
            exit(0)
        }
    }
}

// Run the app event loop (needed for NSWindow overlay rendering)
app.run()
