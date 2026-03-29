import Foundation

public struct QuantizedKey: Sendable {
    public let mseIndices: [UInt8]
    public let mseNorm: Float
    public let qjlBits: [UInt8]
    public let residualNorm: Float
}

public final class TurboQuantProd: @unchecked Sendable {
    public let headDim: Int
    public let bits: Int
    private let mseQuantizer: TurboQuantMSE
    private let qjlQuantizer: QJLQuantizer

    public init(headDim: Int, bits: Int, seed: UInt64 = 42, metal: MetalKernelManager? = nil) {
        self.headDim = headDim
        self.bits = bits
        self.mseQuantizer = TurboQuantMSE(headDim: headDim, bits: max(bits - 1, 1), seed: seed, metal: metal)
        self.qjlQuantizer = QJLQuantizer(headDim: headDim, seed: seed &+ 500)
    }

    public func quantize(_ x: [Float]) -> QuantizedKey {
        let (mseIdx, mseNorm) = mseQuantizer.quantize(x)
        let xHat = mseQuantizer.dequantize(indices: mseIdx, norm: mseNorm)

        var residual = [Float](repeating: 0, count: headDim)
        for i in 0..<headDim {
            residual[i] = x[i] - xHat[i]
        }

        let (qjlBits, residualNorm) = qjlQuantizer.quantize(residual)

        return QuantizedKey(
            mseIndices: mseIdx,
            mseNorm: mseNorm,
            qjlBits: qjlBits,
            residualNorm: residualNorm
        )
    }

    public func estimateInnerProduct(query: [Float], key: QuantizedKey) -> Float {
        let mseScore = mseQuantizer.computeRotatedScore(
            query: query, indices: key.mseIndices, keyNorm: key.mseNorm
        )

        let qjlScore = qjlQuantizer.estimateInnerProduct(
            query: query, signBits: key.qjlBits, keyNorm: key.residualNorm
        )

        return mseScore + qjlScore
    }

    public func dequantize(_ key: QuantizedKey) -> [Float] {
        let xMse = mseQuantizer.dequantize(indices: key.mseIndices, norm: key.mseNorm)

        let scale = sqrt(.pi / 2.0) / Float(headDim)
        var result = xMse

        for i in 0..<headDim {
            let byteIdx = i / 8
            let bitIdx = i % 8
            let sign: Float = ((key.qjlBits[byteIdx] >> bitIdx) & 1) == 1 ? 1.0 : -1.0
            result[i] += scale * key.residualNorm * sign
        }

        return result
    }

    public var compressionRatio: Float {
        let originalBits = headDim * 16
        let mseBits = (bits - 1) * headDim
        let qjlBits = headDim
        let normBits = 32 + 32
        let compressedBits = mseBits + qjlBits + normBits
        return Float(originalBits) / Float(compressedBits)
    }
}
