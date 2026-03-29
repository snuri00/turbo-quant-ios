import Foundation
import Accelerate

public final class RotationMatrix: @unchecked Sendable {
    public let dimension: Int
    public let matrix: [Float]
    public let matrixTranspose: [Float]

    private static var cache: [String: RotationMatrix] = [:]

    public static func get(dimension: Int, seed: UInt64) -> RotationMatrix {
        let key = "\(dimension)_\(seed)"
        if let cached = cache[key] {
            return cached
        }
        let rm = RotationMatrix(dimension: dimension, seed: seed)
        cache[key] = rm
        return rm
    }

    private init(dimension: Int, seed: UInt64) {
        self.dimension = dimension
        let n = dimension

        var gaussian = [Float](repeating: 0, count: n * n)
        var rng = SeededRNG(seed: seed)
        for i in 0..<(n * n) {
            gaussian[i] = rng.nextGaussian()
        }

        var tau = [Float](repeating: 0, count: n)
        var work = [Float](repeating: 0, count: n * 64)
        var info: __CLPK_integer = 0
        var m = __CLPK_integer(n)
        var lda = __CLPK_integer(n)
        var lwork = __CLPK_integer(n * 64)

        sgeqrf_(&m, &m, &gaussian, &lda, &tau, &work, &lwork, &info)

        var signs = [Float](repeating: 0, count: n)
        for i in 0..<n {
            signs[i] = gaussian[i * n + i] >= 0 ? 1.0 : -1.0
        }

        sorgqr_(&m, &m, &m, &gaussian, &lda, &tau, &work, &lwork, &info)

        for j in 0..<n {
            for i in 0..<n {
                gaussian[j * n + i] *= signs[j]
            }
        }

        self.matrix = gaussian

        var transpose = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                transpose[j * n + i] = gaussian[i * n + j]
            }
        }
        self.matrixTranspose = transpose
    }

    public func rotate(_ vector: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: dimension)
        cblas_sgemv(CblasColMajor, CblasNoTrans, Int32(dimension), Int32(dimension),
                    1.0, matrix, Int32(dimension), vector, 1, 0.0, &result, 1)
        return result
    }

    public func rotateInverse(_ vector: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: dimension)
        cblas_sgemv(CblasColMajor, CblasNoTrans, Int32(dimension), Int32(dimension),
                    1.0, matrixTranspose, Int32(dimension), vector, 1, 0.0, &result, 1)
        return result
    }

    public func rotateBatch(_ vectors: [Float], count: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count * dimension)
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    Int32(dimension), Int32(count), Int32(dimension),
                    1.0, matrix, Int32(dimension),
                    vectors, Int32(dimension),
                    0.0, &result, Int32(dimension))
        return result
    }
}

struct SeededRNG {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }

    mutating func nextFloat() -> Float {
        return Float(next() >> 40) / Float(1 << 24)
    }

    mutating func nextGaussian() -> Float {
        let u1 = max(nextFloat(), 1e-10)
        let u2 = nextFloat()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }
}
