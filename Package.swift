// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "UniversalDictator",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DictatorCore", targets: ["DictatorCore"]),
        .library(name: "DictatorEngine", targets: ["DictatorEngine"]),
        .executable(name: "dictate-cli", targets: ["dictate-cli"]),
    ],
    targets: [
        // No Metal: audio, mel frontend, tokenizer, RNN-T decoder, Nemotron on the Neural Engine, and the
        // dictation API. The Mac app links only this.
        .target(name: "DictatorCore"),
        // Metal kernels and the Metal models (Nemotron, Parakeet, Voxtral), built on DictatorCore.
        .target(
            name: "DictatorEngine",
            dependencies: ["DictatorCore"],
            resources: [.copy("Metal")]
        ),
        .executableTarget(name: "dictate-cli", dependencies: ["DictatorEngine"]),
        .testTarget(name: "DictatorEngineTests", dependencies: ["DictatorEngine", "DictatorCore"]),
        // Links only DictatorCore: the dictation API on the Neural Engine with no Metal.
        .testTarget(name: "DictatorCoreTests", dependencies: ["DictatorCore"]),
    ],
    swiftLanguageModes: [.v5]
)
