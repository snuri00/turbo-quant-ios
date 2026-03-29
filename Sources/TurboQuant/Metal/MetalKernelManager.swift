import Foundation
import Metal

public final class MetalKernelManager: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        setupPipelines()
    }

    private func setupPipelines() {
        guard let library = try? device.makeDefaultLibrary(bundle: Bundle.module) else {
            guard let library = device.makeDefaultLibrary() else { return }
            loadPipelines(from: library)
            return
        }
        loadPipelines(from: library)
    }

    private func loadPipelines(from library: MTLLibrary) {
        let kernelNames = [
            "turbo_quantize",
            "turbo_score",
            "qjl_quantize",
            "qjl_score",
            "value_quantize",
        ]
        for name in kernelNames {
            if let function = library.makeFunction(name: name),
               let pipeline = try? device.makeComputePipelineState(function: function) {
                pipelines[name] = pipeline
            }
        }
    }

    public func pipeline(for name: String) -> MTLComputePipelineState? {
        return pipelines[name]
    }

    public func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        return device.makeBuffer(bytes: data, length: MemoryLayout<T>.stride * data.count, options: .storageModeShared)
    }

    public func makeBuffer(length: Int) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: .storageModeShared)
    }

    public func dispatch(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        gridSize: MTLSize,
        threadGroupSize: MTLSize
    ) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        for (i, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: i)
        }
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
