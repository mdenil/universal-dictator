import XCTest
@testable import DictatorCore

final class WeightReaderTests: XCTestCase {
    /// The portable half decoder agrees with Float16 on every finite bit pattern.
    func testHalfToFloatMatchesFloat16() {
        for bits in UInt16.min...UInt16.max {
            let ref = Float(Float16(bitPattern: bits))
            if ref.isNaN { XCTAssertTrue(halfToFloat(bits).isNaN); continue }
            XCTAssertEqual(halfToFloat(bits), ref, "bits \(bits)")
        }
    }
}
