import Foundation

public enum TurboQuant {
    public static let version = "0.1.0"

    public static func createEngine(config: TurboQuantConfig = TurboQuantConfig()) -> LlamaCppBridge {
        return LlamaCppBridge(config: config)
    }

    public static func createQuantizer(
        headDim: Int = 128,
        bits: Int = 3,
        seed: UInt64 = 42
    ) -> TurboQuantProd {
        return TurboQuantProd(headDim: headDim, bits: bits, seed: seed)
    }

    public static func isMetalAvailable() -> Bool {
        return MetalKernelManager() != nil
    }
}
