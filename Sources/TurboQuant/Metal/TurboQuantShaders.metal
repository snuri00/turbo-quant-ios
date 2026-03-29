#include <metal_stdlib>
using namespace metal;

kernel void turbo_quantize(
    device const float* input [[buffer(0)]],
    device const float* rotation [[buffer(1)]],
    device const float* boundaries [[buffer(2)]],
    device uint8_t* output [[buffer(3)]],
    device float* norms [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& num_boundaries [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint j = gid.x;

    if (j >= head_dim) return;

    device const float* x = input + batch_idx * head_dim;

    float norm_sq = 0.0;
    for (uint k = 0; k < head_dim; k++) {
        norm_sq += x[k] * x[k];
    }
    float norm_val = sqrt(norm_sq + 1e-10);

    if (j == 0) {
        norms[batch_idx] = norm_val;
    }

    float inv_norm = 1.0 / norm_val;

    float dot = 0.0;
    for (uint k = 0; k < head_dim; k++) {
        dot += x[k] * inv_norm * rotation[j * head_dim + k];
    }

    int lo = 0;
    int hi = int(num_boundaries) - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (dot > boundaries[mid]) lo = mid + 1;
        else hi = mid;
    }

    output[batch_idx * head_dim + j] = uint8_t(lo);
}

kernel void turbo_score(
    device const float* query [[buffer(0)]],
    device const float* rotation [[buffer(1)]],
    device const uint8_t* key_indices [[buffer(2)]],
    device const float* centroids [[buffer(3)]],
    device const float* key_norms [[buffer(4)]],
    device float* scores [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& num_keys [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint k_idx = gid.x;

    if (k_idx >= num_keys) return;

    device const float* q = query + q_idx * head_dim;

    float q_rotated[256];

    for (uint j = 0; j < head_dim; j++) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += q[d] * rotation[j * head_dim + d];
        }
        q_rotated[j] = dot;
    }

    device const uint8_t* k_idx_ptr = key_indices + k_idx * head_dim;
    float k_norm = key_norms[k_idx];

    float score = 0.0;
    for (uint j = 0; j < head_dim; j++) {
        score += q_rotated[j] * centroids[k_idx_ptr[j]] * k_norm;
    }

    scores[q_idx * num_keys + k_idx] = score;
}

kernel void qjl_quantize(
    device const float* input [[buffer(0)]],
    device const float* jl_matrix [[buffer(1)]],
    device uint8_t* sign_bits [[buffer(2)]],
    device float* norms [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& proj_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint byte_idx = gid.x;
    uint packed_dim = (proj_dim + 7) / 8;

    if (byte_idx >= packed_dim) return;

    device const float* x = input + batch_idx * head_dim;

    if (byte_idx == 0) {
        float norm_sq = 0.0;
        for (uint i = 0; i < head_dim; i++) {
            norm_sq += x[i] * x[i];
        }
        norms[batch_idx] = sqrt(norm_sq);
    }

    uint8_t packed = 0;
    for (uint bit = 0; bit < 8; bit++) {
        uint j = byte_idx * 8 + bit;
        if (j >= proj_dim) break;

        float dot = 0.0;
        for (uint k = 0; k < head_dim; k++) {
            dot += x[k] * jl_matrix[j * head_dim + k];
        }

        if (dot >= 0.0) packed |= (1 << bit);
    }

    sign_bits[batch_idx * packed_dim + byte_idx] = packed;
}

kernel void qjl_score(
    device const float* query [[buffer(0)]],
    device const float* jl_matrix [[buffer(1)]],
    device const uint8_t* sign_bits [[buffer(2)]],
    device const float* key_norms [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& proj_dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint k_idx = gid.x;
    uint packed_dim = (proj_dim + 7) / 8;

    float scale = 1.2533141373 / float(proj_dim);

    device const float* q = query + q_idx * head_dim;
    device const uint8_t* k_bits = sign_bits + k_idx * packed_dim;
    float k_norm = key_norms[k_idx];

    float dot = 0.0;
    for (uint j = 0; j < proj_dim; j++) {
        float sq_j = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            sq_j += q[d] * jl_matrix[j * head_dim + d];
        }

        uint byte_i = j / 8;
        uint bit_i = j % 8;
        float sign_val = ((k_bits[byte_i] >> bit_i) & 1) ? 1.0 : -1.0;

        dot += sq_j * sign_val;
    }

    scores[q_idx + k_idx] = scale * k_norm * dot; // FIXME: proper indexing
}

kernel void value_quantize(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device float* zero_points [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& max_val [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid;
    device const float* x = input + batch_idx * dim;
    device uint8_t* out = output + batch_idx * dim;

    float vmin = x[0], vmax = x[0];
    for (uint i = 1; i < dim; i++) {
        vmin = min(vmin, x[i]);
        vmax = max(vmax, x[i]);
    }

    float range = vmax - vmin;
    if (range < 1e-10) range = 1.0;
    float scale = range / float(max_val);

    scales[batch_idx] = scale;
    zero_points[batch_idx] = vmin;

    float inv_scale = 1.0 / scale;
    for (uint i = 0; i < dim; i++) {
        float q = round((x[i] - vmin) * inv_scale);
        q = clamp(q, 0.0, float(max_val));
        out[batch_idx * dim + i] = uint8_t(q);
    }
}
