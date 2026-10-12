import Foundation

public enum EngineError: Error, CustomStringConvertible {
    case io(String)

    public var description: String {
        switch self {
        case .io(let s): return "io: \(s)"
        }
    }
}
