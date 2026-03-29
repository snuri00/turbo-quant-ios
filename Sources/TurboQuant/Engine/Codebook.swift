import Foundation
import Accelerate

public final class Codebook: @unchecked Sendable {
    public let dimension: Int
    public let bits: Int
    public let centroids: [Float]
    public let boundaries: [Float]

    private static var cache: [String: Codebook] = [:]

    public static func get(dimension: Int, bits: Int) -> Codebook {
        let key = "\(dimension)_\(bits)"
        if let cached = cache[key] {
            return cached
        }
        let cb = Codebook(dimension: dimension, bits: bits)
        cache[key] = cb
        return cb
    }

    private init(dimension: Int, bits: Int) {
        self.dimension = dimension
        self.bits = bits

        let scale = 1.0 / sqrt(Float(dimension))

        switch bits {
        case 1:
            let c = sqrt(2.0 / .pi) * scale
            self.centroids = [-c, c]
        case 2:
            self.centroids = [-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale]
        case 3:
            self.centroids = Codebook.solveGaussianLloydMax(dimension: dimension, bits: 3)
        case 4:
            self.centroids = Codebook.solveGaussianLloydMax(dimension: dimension, bits: 4)
        default:
            self.centroids = Codebook.solveGaussianLloydMax(dimension: dimension, bits: bits)
        }

        var b = [Float](repeating: 0, count: centroids.count + 1)
        b[0] = -.infinity
        b[centroids.count] = .infinity
        for i in 0..<(centroids.count - 1) {
            b[i + 1] = (centroids[i] + centroids[i + 1]) / 2
        }
        self.boundaries = b
    }

    private static func solveGaussianLloydMax(dimension: Int, bits: Int) -> [Float] {
        let k = 1 << bits
        let scale = 1.0 / sqrt(Float(dimension))
        var centroids = (0..<k).map { i in
            Float(i) / Float(k - 1) * 6.0 * scale - 3.0 * scale
        }

        let sigma = scale
        let nPoints = 10000
        let xMin = -4.0 * sigma
        let xMax = 4.0 * sigma
        let dx = (xMax - xMin) / Float(nPoints)

        let xs = (0..<nPoints).map { xMin + Float($0) * dx }
        let pdf = xs.map { x in
            (1.0 / (sigma * sqrt(2.0 * .pi))) * exp(-x * x / (2.0 * sigma * sigma))
        }

        for _ in 0..<2000 {
            var boundaries = [Float](repeating: 0, count: k + 1)
            boundaries[0] = xMin - dx
            boundaries[k] = xMax + dx
            for i in 0..<(k - 1) {
                boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2
            }

            var newCentroids = [Float](repeating: 0, count: k)
            var converged = true

            for i in 0..<k {
                var num: Float = 0
                var den: Float = 0
                for j in 0..<nPoints {
                    if xs[j] >= boundaries[i] && xs[j] < boundaries[i + 1] {
                        num += xs[j] * pdf[j]
                        den += pdf[j]
                    }
                }
                newCentroids[i] = den > 1e-15 ? num / den : (boundaries[i] + boundaries[i + 1]) / 2

                if abs(newCentroids[i] - centroids[i]) > 1e-10 {
                    converged = false
                }
            }

            centroids = newCentroids
            if converged { break }
        }

        return centroids
    }

    public func quantize(_ value: Float) -> Int {
        for i in 1..<boundaries.count {
            if value < boundaries[i] {
                return i - 1
            }
        }
        return centroids.count - 1
    }

    public func dequantize(_ index: Int) -> Float {
        return centroids[min(max(index, 0), centroids.count - 1)]
    }
}
