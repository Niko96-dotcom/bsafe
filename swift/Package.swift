// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BsafeCapture",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "BsafeCapture", targets: ["BsafeCapture"]),
        .executable(name: "bsafe-replay", targets: ["BsafeReplay"]),
    ],
    targets: [
        .target(
            name: "BsafeCore",
            path: "Sources/BsafeCore"
        ),
        .executableTarget(
            name: "BsafeCapture",
            dependencies: ["BsafeCore"],
            path: "Sources/BsafeCapture"
        ),
        .executableTarget(
            name: "BsafeReplay",
            dependencies: ["BsafeCore"],
            path: "Sources/BsafeReplay"
        ),
        .testTarget(
            name: "BsafeCoreTests",
            dependencies: ["BsafeCore"],
            path: "Tests/BsafeCoreTests"
        ),
    ]
)
