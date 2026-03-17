import Foundation

class SocketClient {
    private let path: String
    private var fd: Int32 = -1
    private let lock = NSLock()

    init(path: String) {
        self.path = path
    }

    func connect() throws {
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(
                domain: "SocketClient", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create socket"]
            )
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            throw NSError(
                domain: "SocketClient", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Socket path too long"]
            )
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let addrLen = socklen_t(
            MemoryLayout<sa_family_t>.size + pathBytes.count)
        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(fd, sockaddrPtr, addrLen)
            }
        }

        guard result == 0 else {
            let errMsg = String(cString: strerror(errno))
            close(fd)
            fd = -1
            throw NSError(
                domain: "SocketClient", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to connect: \(errMsg)"
                ]
            )
        }
    }

    func sendMessage(type: UInt8, payload: Data) throws {
        lock.lock()
        defer { lock.unlock() }

        let length = UInt32(1 + payload.count).bigEndian
        var header = Data()
        header.append(contentsOf: withUnsafeBytes(of: length) { Array($0) })
        header.append(type)

        try send(data: header)
        if !payload.isEmpty {
            try send(data: payload)
        }
    }

    /// Max message payload size (50 MB — must match Python's MAX_MESSAGE_SIZE)
    private static let maxMessageSize = 50 * 1024 * 1024

    func readMessage() throws -> (type: UInt8, payload: Data) {
        lock.lock()
        defer { lock.unlock() }

        let header = try recv(exactly: 5)
        let length = header.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let msgType = header[4]
        let payloadSize = Int(length) - 1
        guard payloadSize <= SocketClient.maxMessageSize else {
            throw NSError(
                domain: "SocketClient", code: 6,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Message too large: \(payloadSize) bytes exceeds \(SocketClient.maxMessageSize) byte limit"
                ]
            )
        }
        let payload = payloadSize > 0 ? try recv(exactly: payloadSize) : Data()
        return (msgType, payload)
    }

    func disconnect() {
        lock.lock()
        defer { lock.unlock() }

        if fd >= 0 {
            close(fd)
            fd = -1
        }
    }

    // MARK: - Private

    private func send(data: Data) throws {
        try data.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            var sent = 0
            while sent < data.count {
                let result = Darwin.send(fd, base + sent, data.count - sent, 0)
                guard result > 0 else {
                    throw NSError(
                        domain: "SocketClient", code: 4,
                        userInfo: [NSLocalizedDescriptionKey: "Send failed"]
                    )
                }
                sent += result
            }
        }
    }

    private func recv(exactly count: Int) throws -> Data {
        var buffer = Data(count: count)
        var received = 0
        while received < count {
            let result = buffer.withUnsafeMutableBytes { buf in
                Darwin.recv(fd, buf.baseAddress! + received, count - received, 0)
            }
            guard result > 0 else {
                throw NSError(
                    domain: "SocketClient", code: 5,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Connection closed"
                    ]
                )
            }
            received += result
        }
        return buffer
    }
}
