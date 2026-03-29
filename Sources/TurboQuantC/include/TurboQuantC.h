#ifndef TURBO_QUANT_C_H
#define TURBO_QUANT_C_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void turbo_quant_rotate_cpu(
    const float* input,
    const float* rotation,
    float* output,
    int dim
);

void turbo_quant_quantize_cpu(
    const float* rotated,
    const float* boundaries,
    uint8_t* indices,
    int dim,
    int num_boundaries
);

float turbo_quant_norm(const float* x, int dim);

#ifdef __cplusplus
}
#endif

#endif
