#include "TurboQuantC.h"
#include <math.h>

void turbo_quant_rotate_cpu(
    const float* input,
    const float* rotation,
    float* output,
    int dim
) {
    for (int j = 0; j < dim; j++) {
        float dot = 0.0f;
        for (int k = 0; k < dim; k++) {
            dot += input[k] * rotation[j * dim + k];
        }
        output[j] = dot;
    }
}

void turbo_quant_quantize_cpu(
    const float* rotated,
    const float* boundaries,
    uint8_t* indices,
    int dim,
    int num_boundaries
) {
    for (int j = 0; j < dim; j++) {
        int lo = 0, hi = num_boundaries - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rotated[j] > boundaries[mid]) lo = mid + 1;
            else hi = mid;
        }
        indices[j] = (uint8_t)lo;
    }
}

float turbo_quant_norm(const float* x, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}
