import Foundation
import Accelerate

public final class TurboQuantMSE: @unchecked Sendable {
    public let headDim: Int
    public let bits: Int
    private let rotation: RotationMatrix
    private let codebook: Codebook
    private let metal: MetalKernelManager?

    public init(headDim: Int, bits: Int, seed: UInt64 = 42, metal: MetalKernelManager? = nil) {
        self.headDim = headDim
        self.bits = bits
        self.rotation = RotationMatrix.get(dimension: headDim, seed: seed)
        self.codebook = Codebook.get(dimension: headDim, bits: bits)
        self.metal = metal
    }

    public func quantize(_ x: [Float]) -> (indices: [UInt8], norm: Float) {
        let n = x.count / headDim
        var allIndices = [UInt8](repeating: 0, count: x.count)
        var allNorms = [Float](repeating: 0, count: n)

        for i in 0..<n {
            let offset = i * headDim
            let vector = Array(x[offset..<(offset + headDim)])

            var normSq: Float = 0
            vDSP_dotpr(vector, 1, vector, 1, &normSq, vDSP_Length(headDim))
            let norm = sqrt(normSq + 1e-10)
            allNorms[i] = norm

            let invNorm = 1.0 / norm
            var normalized = [Float](repeating: 0, count: headDim)
            var s = invNorm
            vDSP_vsmul(vector, 1, &s, &normalized, 1, vDSP_Length(headDim))

            let rotated = rotation.rotate(normalized)

            for j in 0..<headDim {
                allIndices[offset + j] = UInt8(codebook.quantize(rotated[j]))
            }
        }

        return (allIndices, n == 1 ? allNorms[0] : allNorms[0])
    }

    public func quantizeBatch(_ x: [Float], count: Int) -> (indices: [UInt8], norms: [Float]) {
        var allIndices = [UInt8](repeating: 0, count: count * headDim)
        var allNorms = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let offset = i * headDim
            let vector = Array(x[offset..<(offset + headDim)])

            var normSq: Float = 0
            vDSP_dotpr(vector, 1, vector, 1, &normSq, vDSP_Length(headDim))
            let norm = sqrt(normSq + 1e-10)
            allNorms[i] = norm

            let invNorm = 1.0 / norm
            var normalized = [Float](repeating: 0, count: headDim)
            var s = invNorm
            vDSP_vsmul(vector, 1, &s, &normalized, 1, vDSP_Length(headDim))

            let rotated = rotation.rotate(normalized)

            for j in 0..<headDim {
                allIndices[offset + j] = UInt8(codebook.quantize(rotated[j]))
            }
        }

        return (allIndices, allNorms)
    }

    public func dequantize(indices: [UInt8], norm: Float) -> [Float] {
        var rotatedHat = [Float](repeating: 0, count: headDim)
        for j in 0..<headDim {
            rotatedHat[j] = codebook.dequantize(Int(indices[j]))
        }

        var result = rotation.rotateInverse(rotatedHat)
        var n = norm
        vDSP_vsmul(result, 1, &n, &result, 1, vDSP_Length(headDim))
        return result
    }

    public func computeRotatedScore(query: [Float], indices: [UInt8], keyNorm: Float) -> Float {
        var qNormSq: Float = 0
        vDSP_dotpr(query, 1, query, 1, &qNormSq, vDSP_Length(headDim))
        let qNorm = sqrt(qNormSq + 1e-10)

        let invQNorm = 1.0 / qNorm
        var qNormalized = [Float](repeating: 0, count: headDim)
        var s = invQNorm
        vDSP_vsmul(query, 1, &s, &qNormalized, 1, vDSP_Length(headDim))

        let qRotated = rotation.rotate(qNormalized)

        var score: Float = 0
        for j in 0..<headDim {
            score += qRotated[j] * codebook.dequantize(Int(indices[j]))
        }

        return score * qNorm * keyNorm
    }
}
