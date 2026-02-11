/**
 * Metal Shaders for 3D Cell Pulse Propagation (Apple Silicon MPS)
 * 
 * Optimized compute shaders for lattice-based neural network propagation.
 * Designed for Apple M1/M2/M3 GPU architecture.
 */

#include <metal_stdlib>
using namespace metal;

// Constants
constant float SQRT2 = 1.41421356f;
// Note: Bounce angles are now stored in radians (max 2 rad)


/**
 * Fast error function approximation (Abramowitz and Stegun)
 */
inline float fast_erf(float x) {
    float sign = x >= 0 ? 1.0f : -1.0f;
    x = abs(x);
    
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float result = 1.0f - t * (0.254829592f + t * (-0.284496736f + 
                  t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)))) 
                  * exp(-x * x);
    
    return sign * result;
}


/**
 * Kernel: Compute propagation weights based on cell parameters
 * 
 * Each thread computes one row of the propagation matrix.
 */
kernel void compute_propagation_weights(
    device const float* std_devs       [[buffer(0)]],
    device const float* adjacency      [[buffer(1)]],
    device float* propagation          [[buffer(2)]],
    constant int& n_cells              [[buffer(3)]],
    constant float& threshold          [[buffer(4)]],
    uint idx                           [[thread_position_in_grid]]
) {
    if (idx >= uint(n_cells)) return;
    
    // Compute straight probability
    float std_dev = max(abs(std_devs[idx]), 0.5f);
    float z = threshold / (std_dev * SQRT2);
    float straight_prob = fast_erf(z);
    float spread_factor = 1.0f - straight_prob;
    
    // Compute propagation weights for this row
    float row_sum = 0.0f;
    
    for (int j = 0; j < n_cells; j++) {
        float weight;
        if (int(idx) == j) {
            weight = straight_prob * 0.3f;
        } else {
            weight = adjacency[idx * n_cells + j] * spread_factor * 0.7f;
        }
        propagation[idx * n_cells + j] = weight;
        row_sum += weight;
    }
    
    // Normalize row
    if (row_sum > 1e-6f) {
        for (int j = 0; j < n_cells; j++) {
            propagation[idx * n_cells + j] /= row_sum;
        }
    }
}


/**
 * Kernel: Apply bounce transformation to activations
 * 
 * Processes one element per thread.
 * Angles are already in radians (max 2 rad each).
 */
kernel void apply_bounce_transform(
    device float* activations          [[buffer(0)]],
    device const float* bounce_angles  [[buffer(1)]],
    constant int& batch_size           [[buffer(2)]],
    constant int& n_cells              [[buffer(3)]],
    uint idx                           [[thread_position_in_grid]]
) {
    uint total = uint(batch_size * n_cells);
    if (idx >= total) return;
    
    uint cell_idx = idx % uint(n_cells);
    
    // Compute average angle magnitude (angles already in radians)
    float angle_avg = (abs(bounce_angles[cell_idx * 3 + 0]) +
                      abs(bounce_angles[cell_idx * 3 + 1]) +
                      abs(bounce_angles[cell_idx * 3 + 2])) / 3.0f;
    
    float angle_factor = cos(angle_avg);
    
    // Apply transformation
    activations[idx] *= (0.5f + 0.5f * angle_factor);
}


/**
 * Kernel: Apply decay and sigmoid nonlinearity
 */
kernel void apply_decay_sigmoid(
    device float* state                [[buffer(0)]],
    constant float& decay              [[buffer(1)]],
    constant float& state_max          [[buffer(2)]],
    constant int& batch_size           [[buffer(3)]],
    constant int& n_cells              [[buffer(4)]],
    uint idx                           [[thread_position_in_grid]]
) {
    uint total = uint(batch_size * n_cells);
    if (idx >= total) return;
    
    // Apply decay
    float val = state[idx] * decay;
    
    // Apply sigmoid
    float sigmoid_input = val * 2.0f - 1.0f;
    float sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_input));
    
    float max_val = max(state_max, 0.1f);
    state[idx] = sigmoid_val * max_val;
}


/**
 * Kernel: Matrix multiplication for state propagation
 * 
 * Computes: output = state @ propagation^T
 * Uses threadgroup memory for tiling.
 */
kernel void propagate_state(
    device const float* state          [[buffer(0)]],
    device const float* propagation    [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant int& batch_size           [[buffer(3)]],
    constant int& n_cells              [[buffer(4)]],
    uint2 gid                          [[thread_position_in_grid]],
    uint2 tid                          [[thread_position_in_threadgroup]],
    uint2 tg_size                      [[threads_per_threadgroup]]
) {
    // gid.x = output column (cell), gid.y = output row (batch)
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= uint(batch_size) || col >= uint(n_cells)) return;
    
    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < n_cells; k++) {
        // state[row, k] * propagation^T[k, col] = state[row, k] * propagation[col, k]
        sum += state[row * n_cells + k] * propagation[col * n_cells + k];
    }
    
    output[row * n_cells + col] = sum;
}


/**
 * Kernel: Tiled matrix multiplication with threadgroup memory
 * 
 * More efficient for larger matrices.
 */
constant int TILE_SIZE = 16;

kernel void propagate_state_tiled(
    device const float* state          [[buffer(0)]],
    device const float* propagation    [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant int& batch_size           [[buffer(3)]],
    constant int& n_cells              [[buffer(4)]],
    uint2 gid                          [[thread_position_in_grid]],
    uint2 tid                          [[thread_position_in_threadgroup]],
    uint2 group_id                     [[threadgroup_position_in_grid]]
) {
    threadgroup float tile_state[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_prop[TILE_SIZE][TILE_SIZE];
    
    uint row = group_id.y * TILE_SIZE + tid.y;
    uint col = group_id.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    uint n_tiles = (uint(n_cells) + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < n_tiles; t++) {
        // Load state tile
        uint state_col = t * TILE_SIZE + tid.x;
        if (row < uint(batch_size) && state_col < uint(n_cells)) {
            tile_state[tid.y][tid.x] = state[row * n_cells + state_col];
        } else {
            tile_state[tid.y][tid.x] = 0.0f;
        }
        
        // Load propagation^T tile
        uint prop_row = t * TILE_SIZE + tid.y;
        if (prop_row < uint(n_cells) && col < uint(n_cells)) {
            tile_prop[tid.y][tid.x] = propagation[col * n_cells + prop_row];
        } else {
            tile_prop[tid.y][tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_state[tid.y][k] * tile_prop[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < uint(batch_size) && col < uint(n_cells)) {
        output[row * n_cells + col] = sum;
    }
}


/**
 * Kernel: Fused propagation step
 * 
 * Combines propagation, bounce transform, decay, and sigmoid.
 */
kernel void fused_propagation_step(
    device const float* state_in       [[buffer(0)]],
    device float* state_out            [[buffer(1)]],
    device const float* propagation    [[buffer(2)]],
    device const float* bounce_angles  [[buffer(3)]],
    constant float& decay              [[buffer(4)]],
    constant float& state_max          [[buffer(5)]],
    constant int& batch_size           [[buffer(6)]],
    constant int& n_cells              [[buffer(7)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint cell_idx = gid.x;
    
    if (batch_idx >= uint(batch_size) || cell_idx >= uint(n_cells)) return;
    
    // Compute propagated value
    float propagated = 0.0f;
    for (int j = 0; j < n_cells; j++) {
        propagated += state_in[batch_idx * n_cells + j] * propagation[j * n_cells + cell_idx];
    }
    
    // Apply bounce transform (angles already in radians, max 2 rad each)
    float angle_avg = (abs(bounce_angles[cell_idx * 3 + 0]) +
                      abs(bounce_angles[cell_idx * 3 + 1]) +
                      abs(bounce_angles[cell_idx * 3 + 2])) / 3.0f;
    float angle_factor = cos(angle_avg);
    propagated *= (0.5f + 0.5f * angle_factor);
    
    // Apply decay
    propagated *= decay;
    
    // Apply sigmoid
    float sigmoid_input = propagated * 2.0f - 1.0f;
    float sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_input));
    float max_val = max(state_max, 0.1f);
    
    state_out[batch_idx * n_cells + cell_idx] = sigmoid_val * max_val;
}


/**
 * Kernel: Weighted sum of history states
 */
kernel void weighted_history_sum(
    device const float* history        [[buffer(0)]],
    device const float* weights        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant int& n_steps              [[buffer(3)]],
    constant int& batch_size           [[buffer(4)]],
    constant int& n_cells              [[buffer(5)]],
    uint idx                           [[thread_position_in_grid]]
) {
    uint total = uint(batch_size * n_cells);
    if (idx >= total) return;
    
    float sum = 0.0f;
    for (int step = 0; step < n_steps; step++) {
        sum += history[step * total + idx] * weights[step];
    }
    output[idx] = sum;
}


/**
 * Kernel: Compute straight probability for all cells
 * 
 * Utility kernel for visualization/debugging.
 */
kernel void compute_straight_probabilities(
    device const float* std_devs       [[buffer(0)]],
    device float* straight_probs       [[buffer(1)]],
    constant int& n_cells              [[buffer(2)]],
    constant float& threshold          [[buffer(3)]],
    uint idx                           [[thread_position_in_grid]]
) {
    if (idx >= uint(n_cells)) return;
    
    float std_dev = max(abs(std_devs[idx]), 0.5f);
    float z = threshold / (std_dev * SQRT2);
    straight_probs[idx] = fast_erf(z);
}


/**
 * Kernel: Batch normalization for activations
 * 
 * Normalizes activations to prevent explosion/vanishing.
 */
kernel void normalize_activations(
    device float* activations          [[buffer(0)]],
    constant int& batch_size           [[buffer(1)]],
    constant int& n_cells              [[buffer(2)]],
    constant float& eps                [[buffer(3)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint cell_idx = gid.x;
    
    if (batch_idx >= uint(batch_size) || cell_idx >= uint(n_cells)) return;
    
    // Compute mean and variance for this batch
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int j = 0; j < n_cells; j++) {
        float val = activations[batch_idx * n_cells + j];
        sum += val;
        sq_sum += val * val;
    }
    
    float mean = sum / float(n_cells);
    float variance = sq_sum / float(n_cells) - mean * mean;
    float std_dev = sqrt(variance + eps);
    
    // Normalize
    float val = activations[batch_idx * n_cells + cell_idx];
    activations[batch_idx * n_cells + cell_idx] = (val - mean) / std_dev;
}
