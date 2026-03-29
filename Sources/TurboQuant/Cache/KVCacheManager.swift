import Foundation

public final class KVCacheManager: @unchecked Sendable {
    public let config: TurboQuantConfig
    private var quantizedKeys: [[QuantizedKey]]
    private var quantizedValues: [[[UInt8]]]
    private var valueScales: [[Float]]
    private var valueZeroPoints: [[Float]]
    private var bufferKeys: [[Float]]
    private var bufferValues: [[Float]]
    private let quantizer: TurboQuantProd

    public private(set) var quantizedSeqLen: Int = 0
    public var bufferSeqLen: Int { bufferKeys.count }
    public var totalSeqLen: Int { quantizedSeqLen + bufferSeqLen }

    public init(config: TurboQuantConfig, metal: MetalKernelManager? = nil) {
        self.config = config
        self.quantizedKeys = []
        self.quantizedValues = []
        self.valueScales = []
        self.valueZeroPoints = []
        self.bufferKeys = []
        self.bufferValues = []
        self.quantizer = TurboQuantProd(
            headDim: config.headDim,
            bits: config.keyBits,
            seed: config.rotationSeed,
            metal: metal
        )
    }

    public func addToken(key: [Float], value: [Float]) {
        bufferKeys.append(key)
        bufferValues.append(value)

        if bufferKeys.count > config.bufferSize {
            flushOldest()
        }
    }

    private func flushOldest() {
        let overflow = bufferKeys.count - config.bufferSize
        guard overflow > 0 else { return }

        for i in 0..<overflow {
            let qKey = quantizer.quantize(bufferKeys[i])
            quantizedKeys.append([qKey])

            let (qVal, scale, zp) = quantizeValue(bufferValues[i])
            quantizedValues.append([qVal])
            valueScales.append([scale])
            valueZeroPoints.append([zp])
        }

        bufferKeys.removeFirst(overflow)
        bufferValues.removeFirst(overflow)
        quantizedSeqLen += overflow
    }

    private func quantizeValue(_ v: [Float]) -> ([UInt8], Float, Float) {
        let maxVal = Float((1 << config.valueBits) - 1)
        let vMin = v.min() ?? 0
        let vMax = v.max() ?? 1
        var scale = (vMax - vMin) / maxVal
        if scale < 1e-10 { scale = 1.0 }

        var quantized = [UInt8](repeating: 0, count: v.count)
        for i in 0..<v.count {
            let q = ((v[i] - vMin) / scale).rounded()
            quantized[i] = UInt8(max(0, min(maxVal, q)))
        }

        return (quantized, scale, vMin)
    }

    public func computeScores(query: [Float]) -> [Float] {
        var scores = [Float](repeating: 0, count: totalSeqLen)

        for i in 0..<quantizedKeys.count {
            for qKey in quantizedKeys[i] {
                scores[i] = quantizer.estimateInnerProduct(query: query, key: qKey)
            }
        }

        for i in 0..<bufferKeys.count {
            var dot: Float = 0
            let key = bufferKeys[i]
            for j in 0..<config.headDim {
                dot += query[j] * key[j]
            }
            scores[quantizedSeqLen + i] = dot
        }

        return scores
    }

    public func clear() {
        quantizedKeys.removeAll()
        quantizedValues.removeAll()
        valueScales.removeAll()
        valueZeroPoints.removeAll()
        bufferKeys.removeAll()
        bufferValues.removeAll()
        quantizedSeqLen = 0
    }

    public var memorySizeBytes: Int {
        let qKeyBytes = quantizedKeys.count * (config.headDim + config.headDim / 8 + 8)
        let qValBytes = quantizedValues.count * (config.headDim + 8)
        let bufBytes = (bufferKeys.count + bufferValues.count) * config.headDim * 4
        return qKeyBytes + qValBytes + bufBytes
    }

    public var uncompressedSizeBytes: Int {
        return totalSeqLen * config.headDim * 2 * 4
    }
}
