# TurboQuant iOS

Native iOS/macOS framework implementing TurboQuant KV cache compression for on-device LLM inference. Metal GPU acceleration with Apple Neural Engine support.

Based on three research papers:
- TurboQuant (arXiv:2504.19874) - Two-stage vector quantization
- QJL (arXiv:2406.03482) - 1-bit inner product estimation
- PolarQuant (arXiv:2502.02617) - Polar coordinate quantization

Companion project to [turbo-quant](https://github.com/snuri00/turbo-quant) (Python/CUDA).

## Architecture

```
Metal Compute Shaders     -- GPU-accelerated quantize, score, QJL ops
Accelerate / vDSP         -- CPU fallback with SIMD optimization
Swift Framework           -- Clean API for iOS/macOS apps
C Bridge                  -- Low-level compute primitives
```

## Supported Platforms

| Platform | GPU | ANE | Min Version |
|----------|-----|-----|-------------|
| iPhone   | Metal 3 | 16-core | iOS 16+ |
| iPad     | Metal 3 | 16-core | iPadOS 16+ |
| Mac      | Metal   | 16-core | macOS 13+ |

Without TurboQuant, context lengths are 2-4x shorter on the same hardware.

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/snuri00/turbo-quant-ios.git", from: "0.1.0")
]
```

## Usage

### Basic Engine

```swift
import TurboQuant

let config = TurboQuantConfig(
    mode: .turboProd,
    keyBits: 3,
    valueBits: 4,
    headDim: 128,
    numLayers: 32,
    numKVHeads: 8
)

let engine = TurboQuant.createEngine(config: config)

// Feed KV pairs from model layers
engine.updateKVCache(layerIdx: 0, head: 0, key: keyVector, value: valueVector)

// Compute attention scores
let scores = engine.getScores(layerIdx: 0, query: queryVector)

// Check memory usage
let report = engine.memoryReport()
print(report.description)
```

### Standalone Quantizer

```swift
let quantizer = TurboQuant.createQuantizer(headDim: 128, bits: 3)

let quantized = quantizer.quantize(keyVector)
let score = quantizer.estimateInnerProduct(query: queryVector, key: quantized)
```

## Project Structure

```
Sources/
  TurboQuant/
    Metal/              -- Metal compute shaders + kernel manager
    Engine/             -- Codebook, rotation, MSE/Prod/QJL quantizers
    Cache/              -- KV cache manager with buffer + quantized storage
    Bridge/             -- llama.cpp integration bridge
  TurboQuantC/          -- C primitives for CPU fallback
Tests/                  -- XCTest suite
Examples/               -- Usage examples
```

## Performance Targets

- Metal GPU: ~8x speedup over CPU for attention scoring
- ANE offload: matrix operations via CoreML (future)
- 3-bit quantization: zero accuracy loss at 3.5 bits
- Memory: 3-4x reduction in KV cache size

## References

```
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv:2504.19874}, year={2025}
}
```

## License

MIT
