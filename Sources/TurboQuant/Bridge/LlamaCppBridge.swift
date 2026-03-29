import Foundation

public final class LlamaCppBridge: @unchecked Sendable {
    public let config: TurboQuantConfig
    public let metal: MetalKernelManager?
    private var kvCaches: [KVCacheManager]

    public init(config: TurboQuantConfig) {
        self.config = config
        self.metal = MetalKernelManager()
        self.kvCaches = (0..<config.numLayers).map { _ in
            KVCacheManager(config: config, metal: self.metal)
        }
    }

    public func updateKVCache(layerIdx: Int, head: Int, key: [Float], value: [Float]) {
        guard layerIdx < kvCaches.count else { return }
        kvCaches[layerIdx].addToken(key: key, value: value)
    }

    public func getScores(layerIdx: Int, query: [Float]) -> [Float] {
        guard layerIdx < kvCaches.count else { return [] }
        return kvCaches[layerIdx].computeScores(query: query)
    }

    public func clear() {
        for cache in kvCaches {
            cache.clear()
        }
    }

    public func memoryReport() -> MemoryReport {
        var totalCompressed = 0
        var totalUncompressed = 0
        var totalTokens = 0

        for cache in kvCaches {
            totalCompressed += cache.memorySizeBytes
            totalUncompressed += cache.uncompressedSizeBytes
            totalTokens = max(totalTokens, cache.totalSeqLen)
        }

        return MemoryReport(
            compressedBytes: totalCompressed,
            uncompressedBytes: totalUncompressed,
            totalTokens: totalTokens,
            numLayers: config.numLayers,
            compressionRatio: totalUncompressed > 0 ? Float(totalUncompressed) / Float(totalCompressed) : 0
        )
    }
}

public struct MemoryReport: Sendable {
    public let compressedBytes: Int
    public let uncompressedBytes: Int
    public let totalTokens: Int
    public let numLayers: Int
    public let compressionRatio: Float

    public var compressedMB: Float { Float(compressedBytes) / 1_000_000 }
    public var uncompressedMB: Float { Float(uncompressedBytes) / 1_000_000 }

    public var description: String {
        return """
        Tokens: \(totalTokens), Layers: \(numLayers)
        Compressed: \(String(format: "%.1f", compressedMB)) MB
        Uncompressed: \(String(format: "%.1f", uncompressedMB)) MB
        Ratio: \(String(format: "%.1f", compressionRatio))x
        """
    }
}
