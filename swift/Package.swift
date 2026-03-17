// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BsafeCapture",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "BsafeCapture",
            path: "Sources/BsafeCapture"
        ),
    ]
)
