import Foundation
import Accelerate

public final class QJLQuantizer: @unchecked Sendable {
    public let headDim: Int
    public let projDim: Int
    private let jlMatrix: [Float]
    private let jlMatrixT: [Float]
    private let scale: Float

    public init(headDim: Int, projDim: Int? = nil, seed: UInt64 = 42) {
        self.headDim = headDim
        self.projDim = projDim ?? headDim
        self.scale = sqrt(.pi / 2.0) / Float(self.projDim)

        var rng = SeededRNG(seed: seed &+ 1000)
        var matrix = [Float](repeating: 0, count: self.projDim * headDim)
        for i in 0..<matrix.count {
            matrix[i] = rng.nextGaussian()
        }

        if self.projDim == headDim {
            var tau = [Float](repeating: 0, count: headDim)
            var work = [Float](repeating: 0, count: headDim * 64)
            var info: __CLPK_integer = 0
            var m = __CLPK_integer(headDim)
            var lwork = __CLPK_integer(headDim * 64)

            sgeqrf_(&m, &m, &matrix, &m, &tau, &work, &lwork, &info)

            var signs = [Float](repeating: 0, count: headDim)
            for i in 0..<headDim {
                signs[i] = matrix[i * headDim + i] >= 0 ? 1.0 : -1.0
            }

            sorgqr_(&m, &m, &m, &matrix, &m, &tau, &work, &lwork, &info)

            for j in 0..<headDim {
                for i in 0..<headDim {
                    matrix[j * headDim + i] *= signs[j]
                }
            }
        }

        self.jlMatrix = matrix

        var transpose = [Float](repeating: 0, count: self.projDim * headDim)
        for i in 0..<self.projDim {
            for j in 0..<headDim {
                transpose[j * self.projDim + i] = matrix[i * headDim + j]
            }
        }
        self.jlMatrixT = transpose
    }

    public func quantize(_ x: [Float]) -> (signBits: [UInt8], norm: Float) {
        var normSq: Float = 0
        vDSP_dotpr(x, 1, x, 1, &normSq, vDSP_Length(headDim))
        let norm = sqrt(normSq)

        var projected = [Float](repeating: 0, count: projDim)
        cblas_sgemv(CblasColMajor, CblasNoTrans, Int32(projDim), Int32(headDim),
                    1.0, jlMatrix, Int32(projDim), x, 1, 0.0, &projected, 1)

        let packedDim = (projDim + 7) / 8
        var signBits = [UInt8](repeating: 0, count: packedDim)
        for i in 0..<projDim {
            if projected[i] >= 0 {
                signBits[i / 8] |= UInt8(1 << (i % 8))
            }
        }

        return (signBits, norm)
    }

    public func estimateInnerProduct(query: [Float], signBits: [UInt8], keyNorm: Float) -> Float {
        var sq = [Float](repeating: 0, count: projDim)
        cblas_sgemv(CblasColMajor, CblasNoTrans, Int32(projDim), Int32(headDim),
                    1.0, jlMatrix, Int32(projDim), query, 1, 0.0, &sq, 1)

        var dot: Float = 0
        for j in 0..<projDim {
            let byteIdx = j / 8
            let bitIdx = j % 8
            let sign: Float = ((signBits[byteIdx] >> bitIdx) & 1) == 1 ? 1.0 : -1.0
            dot += sq[j] * sign
        }

        return scale * keyNorm * dot
    }
}
