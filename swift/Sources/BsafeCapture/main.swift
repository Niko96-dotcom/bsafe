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

let capture = ScreenCapture(fps: fps)

// JPEG quality for detection — lower saves IPC bandwidth (no display needed)
let jpegQuality = 0.4

// Set up frame callback: encode and send
capture.onFrame = { cgImage, displayID, timestamp in
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

print("BsafeCapture: capturing at \(fps) FPS. Waiting for shutdown...")

// Listen for CMD_SHUTDOWN in background
DispatchQueue.global().async {
    while true {
        do {
            let (msgType, _) = try client.readMessage()
            if msgType == 0xFF {  // CMD_SHUTDOWN
                print("BsafeCapture: received shutdown command")
                capture.stop()
                client.disconnect()
                exit(0)
            }
        } catch {
            // Connection lost
            print("BsafeCapture: connection lost, shutting down")
            capture.stop()
            exit(0)
        }
    }
}

// Keep the main run loop alive
RunLoop.main.run()
