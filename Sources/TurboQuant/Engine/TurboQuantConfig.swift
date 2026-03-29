import Foundation
import Metal

public enum QuantizationMode: String, Sendable {
    case turboMSE = "turbo_mse"
    case turboProd = "turbo_prod"
    case qjl = "qjl"
}

public struct TurboQuantConfig: Sendable {
    public var mode: QuantizationMode
    public var keyBits: Int
    public var valueBits: Int
    public var outlierBits: Int
    public var numOutlierChannels: Int
    public var bufferSize: Int
    public var rotationSeed: UInt64
    public var headDim: Int
    public var numLayers: Int
    public var numKVHeads: Int

    public init(
        mode: QuantizationMode = .turboProd,
        keyBits: Int = 3,
        valueBits: Int = 4,
        outlierBits: Int = 4,
        numOutlierChannels: Int = 8,
        bufferSize: Int = 32,
        rotationSeed: UInt64 = 42,
        headDim: Int = 128,
        numLayers: Int = 32,
        numKVHeads: Int = 8
    ) {
        self.mode = mode
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.outlierBits = outlierBits
        self.numOutlierChannels = numOutlierChannels
        self.bufferSize = bufferSize
        self.rotationSeed = rotationSeed
        self.headDim = headDim
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
    }

    public var effectiveBits: Float {
        let regular = headDim - numOutlierChannels
        return Float(numOutlierChannels * outlierBits + regular * keyBits) / Float(headDim)
    }

    public var compressionRatio: Float {
        return 16.0 / effectiveBits
    }

    public func kvCacheSizePerToken() -> Int {
        let bitsPerKey = Int(effectiveBits) * headDim + headDim + 32
        let bitsPerValue = valueBits * headDim + 32
        let totalBits = 2 * numLayers * numKVHeads * (bitsPerKey + bitsPerValue)
        return (totalBits + 7) / 8
    }
}
