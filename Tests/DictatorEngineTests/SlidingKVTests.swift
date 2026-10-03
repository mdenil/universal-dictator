import DictatorCore
import XCTest
@testable import DictatorEngine

final class SlidingKVTests: XCTestCase {
    /// Rows store their absolute position; after many shifts, each query's key range must cover exactly
    /// positions (q - window, q] and the stored values must equal those positions.
    func testShiftKeepsWindowAndPositions() {
        let window = 10, slack = 3, width = 4
        let kv = SlidingKV(layers: 1, window: window, slack: slack, width: width, ctx: .shared)
        var pos = 0
        for stepSize in [4, 1, 2, 3, 1, 1, 4, 2, 3, 3, 1, 2, 4, 4, 1] {
            let s = Step()
            let slot = kv.reserve(s, stepSize)
            s.run()
            let p = kv.k[0].contents.assumingMemoryBound(to: Float.self)
            for i in 0..<stepSize { for c in 0..<width { p[(slot + i) * width + c] = Float(pos + i) } }
            let r = kv.ranges(pos0: pos, n: stepSize)
            kv.count = slot + stepSize
            for i in 0..<stepSize {
                let q = pos + i
                let lo = Int(r[2 * i]), hi = Int(r[2 * i + 1])
                let got = (lo..<hi).map { Int(p[$0 * width]) }
                XCTAssertEqual(got, Array(max(0, q - window + 1)...q), "query \(q)")
            }
            pos += stepSize
        }
        XCTAssertGreaterThan(kv.base, 0, "the test should have exercised at least one shift")
    }
}
