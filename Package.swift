// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "UniversalDictator",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DictatorEngine", targets: ["DictatorEngine"]),
        .executable(name: "dictate-cli", targets: ["dictate-cli"]),
    ],
    targets: [
        .target(
            name: "DictatorEngine",
            resources: [.copy("Metal")]
        ),
        .executableTarget(name: "dictate-cli", dependencies: ["DictatorEngine"]),
        .testTarget(name: "DictatorEngineTests", dependencies: ["DictatorEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
