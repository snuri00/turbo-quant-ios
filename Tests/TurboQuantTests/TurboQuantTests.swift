import XCTest
@testable import TurboQuant

final class TurboQuantTests: XCTestCase {

    func testCodebookSymmetry() {
        for bits in 1...4 {
            let cb = Codebook.get(dimension: 128, bits: bits)
            let n = cb.centroids.count
            for i in 0..<(n / 2) {
                XCTAssertEqual(cb.centroids[i] + cb.centroids[n - 1 - i], 0, accuracy: 1e-4)
            }
        }
    }

    func testCodebookCount() {
        for bits in 1...4 {
            let cb = Codebook.get(dimension: 128, bits: bits)
            XCTAssertEqual(cb.centroids.count, 1 << bits)
            XCTAssertEqual(cb.boundaries.count, (1 << bits) + 1)
        }
    }

    func testMSEQuantizeShape() {
        let q = TurboQuantMSE(headDim: 128, bits: 3)
        var x = [Float](repeating: 0, count: 128)
        for i in 0..<128 { x[i] = Float.random(in: -1...1) }

        let (indices, norm) = q.quantize(x)
        XCTAssertEqual(indices.count, 128)
        XCTAssertGreaterThan(norm, 0)
    }

    func testMSERoundtrip() {
        let q = TurboQuantMSE(headDim: 128, bits: 4)
        var x = [Float](repeating: 0, count: 128)
        for i in 0..<128 { x[i] = Float.random(in: -1...1) }

        let norm = sqrt(x.reduce(0) { $0 + $1 * $1 })
        for i in 0..<128 { x[i] /= norm }

        let (indices, qNorm) = q.quantize(x)
        let xHat = q.dequantize(indices: indices, norm: qNorm)

        XCTAssertEqual(xHat.count, 128)

        var mse: Float = 0
        for i in 0..<128 {
            let diff = x[i] - xHat[i]
            mse += diff * diff
        }
        XCTAssertLessThan(mse, 0.05)
    }

    func testQJLUnbiased() {
        let d = 128
        let nSamples = 500

        var totalBias: Float = 0

        for trial in 0..<10 {
            let qjl = QJLQuantizer(headDim: d, seed: UInt64(trial * 7))

            var query = [Float](repeating: 0, count: d)
            for i in 0..<d { query[i] = Float.random(in: -1...1) }

            var biasSum: Float = 0
            for _ in 0..<nSamples {
                var key = [Float](repeating: 0, count: d)
                for i in 0..<d { key[i] = Float.random(in: -1...1) }
                let norm = sqrt(key.reduce(0) { $0 + $1 * $1 })
                for i in 0..<d { key[i] /= norm }

                var exact: Float = 0
                for i in 0..<d { exact += query[i] * key[i] }

                let (bits, kNorm) = qjl.quantize(key)
                let estimated = qjl.estimateInnerProduct(query: query, signBits: bits, keyNorm: kNorm)

                biasSum += estimated - exact
            }

            totalBias += abs(biasSum / Float(nSamples))
        }

        let avgBias = totalBias / 10.0
        XCTAssertLessThan(avgBias, 0.1)
    }

    func testTurboProdQuantize() {
        let q = TurboQuantProd(headDim: 128, bits: 3)
        var x = [Float](repeating: 0, count: 128)
        for i in 0..<128 { x[i] = Float.random(in: -1...1) }

        let result = q.quantize(x)
        XCTAssertEqual(result.mseIndices.count, 128)
        XCTAssertGreaterThan(result.mseNorm, 0)
    }

    func testKVCacheManager() {
        let config = TurboQuantConfig(bufferSize: 4, headDim: 64, numLayers: 1, numKVHeads: 1)
        let cache = KVCacheManager(config: config)

        for _ in 0..<10 {
            var key = [Float](repeating: 0, count: 64)
            var value = [Float](repeating: 0, count: 64)
            for i in 0..<64 {
                key[i] = Float.random(in: -1...1)
                value[i] = Float.random(in: -1...1)
            }
            cache.addToken(key: key, value: value)
        }

        XCTAssertEqual(cache.totalSeqLen, 10)
        XCTAssertEqual(cache.bufferSeqLen, 4)
        XCTAssertEqual(cache.quantizedSeqLen, 6)
    }

    func testCompressionRatio() {
        let q = TurboQuantProd(headDim: 128, bits: 3)
        XCTAssertGreaterThan(q.compressionRatio, 1.0)
    }

    func testMetalAvailability() {
        let available = TurboQuant.isMetalAvailable()
        print("Metal available: \(available)")
    }
}
