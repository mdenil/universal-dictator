import Foundation

public enum EngineError: Error, CustomStringConvertible {
    case noDevice
    case library(String)
    case missingFunction(String)
    case io(String)
    case shape(String)

    public var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .library(let s): return "Metal library: \(s)"
        case .missingFunction(let s): return "missing Metal function \(s)"
        case .io(let s): return "io: \(s)"
        case .shape(let s): return "shape: \(s)"
        }
    }
}
