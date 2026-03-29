// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "TurboQuant",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "TurboQuant", targets: ["TurboQuant"]),
    ],
    targets: [
        .target(
            name: "TurboQuantC",
            path: "Sources/TurboQuantC",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TurboQuant",
            dependencies: ["TurboQuantC"],
            path: "Sources/TurboQuant",
            resources: [
                .process("Metal"),
            ]
        ),
        .testTarget(
            name: "TurboQuantTests",
            dependencies: ["TurboQuant"],
            path: "Tests/TurboQuantTests"
        ),
    ]
)
