import TurboQuant

func exampleBasicUsage() {
    let config = TurboQuantConfig(
        mode: .turboProd,
        keyBits: 3,
        valueBits: 4,
        headDim: 128,
        numLayers: 32,
        numKVHeads: 8
    )

    let engine = TurboQuant.createEngine(config: config)

    print("TurboQuant iOS Engine v\(TurboQuant.version)")
    print("Metal: \(TurboQuant.isMetalAvailable() ? "available" : "CPU fallback")")
    print("Compression: \(String(format: "%.1f", config.compressionRatio))x")
    print("Effective bits: \(String(format: "%.2f", config.effectiveBits))")

    for t in 0..<100 {
        var key = [Float](repeating: 0, count: config.headDim)
        var value = [Float](repeating: 0, count: config.headDim)
        for i in 0..<config.headDim {
            key[i] = Float.random(in: -1...1)
            value[i] = Float.random(in: -1...1)
        }
        engine.updateKVCache(layerIdx: 0, head: 0, key: key, value: value)
    }

    var query = [Float](repeating: 0, count: config.headDim)
    for i in 0..<config.headDim {
        query[i] = Float.random(in: -1...1)
    }

    let scores = engine.getScores(layerIdx: 0, query: query)
    print("Scores for 100 tokens: min=\(scores.min()!), max=\(scores.max()!)")

    let report = engine.memoryReport()
    print(report.description)
}

func exampleStandaloneQuantizer() {
    let quantizer = TurboQuant.createQuantizer(headDim: 128, bits: 3)

    var key = [Float](repeating: 0, count: 128)
    for i in 0..<128 { key[i] = Float.random(in: -1...1) }
    let norm = sqrt(key.reduce(0) { $0 + $1 * $1 })
    for i in 0..<128 { key[i] /= norm }

    let quantized = quantizer.quantize(key)
    print("MSE indices: \(quantized.mseIndices.prefix(10))...")
    print("Norm: \(quantized.mseNorm)")
    print("QJL bits: \(quantized.qjlBits.count) bytes")
    print("Compression: \(String(format: "%.1f", quantizer.compressionRatio))x")
}
