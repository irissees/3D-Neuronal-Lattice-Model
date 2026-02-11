/**
 * CUDA Kernels for 3D Cell Pulse Propagation
 * 
 * Optimized kernels for lattice-based neural network propagation.
 * Includes propagation weight computation, state updates, and transformations.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


/**
 * Kernel: Compute propagation weights based on cell parameters
 * 
 * Uses normal distribution (std_dev) to determine straight vs bounce probability.
 * Output is an n_cells x n_cells propagation matrix.
 */
__global__ void compute_propagation_weights_kernel(
    const float* __restrict__ std_devs,      // (n_cells,)
    const float* __restrict__ adjacency,     // (n_cells, n_cells)
    float* __restrict__ propagation,         // (n_cells, n_cells) output
    const int n_cells,
    const float threshold
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_cells) {
        // Compute straight probability using error function approximation
        float std_dev = fmaxf(fabsf(std_devs[idx]), 0.5f);
        float z = threshold / (std_dev * 1.41421356f);  // sqrt(2)
        
        // Fast erf approximation (Abramowitz and Stegun)
        float t = 1.0f / (1.0f + 0.3275911f * fabsf(z));
        float erf_approx = 1.0f - t * (0.254829592f + t * (-0.284496736f + 
                          t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)))) 
                          * expf(-z * z);
        if (z < 0) erf_approx = -erf_approx;
        
        float straight_prob = erf_approx;
        float spread_factor = 1.0f - straight_prob;
        
        // Compute propagation weights for this row
        float row_sum = 0.0f;
        
        for (int j = 0; j < n_cells; j++) {
            float weight;
            if (idx == j) {
                // Self-retention based on straight probability
                weight = straight_prob * 0.3f;
            } else {
                // Neighbor spreading
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
}


/**
 * Kernel: Apply bounce transformation to activations
 * 
 * Bounce angles modulate activation magnitude.
 * Large angles = more transformation = slightly reduced activation
 */
__global__ void apply_bounce_transform_kernel(
    float* __restrict__ activations,         // (batch_size, n_cells) in/out
    const float* __restrict__ bounce_angles, // (n_cells, 3)
    const int batch_size,
    const int n_cells
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_cells;
    
    if (idx < total) {
        const int cell_idx = idx % n_cells;
        
        // Compute angle magnitude (average of absolute angles across 3 axes)
        // Angles are already in radians (max 2 rad each)
        float angle_avg = (fabsf(bounce_angles[cell_idx * 3 + 0]) +
                          fabsf(bounce_angles[cell_idx * 3 + 1]) +
                          fabsf(bounce_angles[cell_idx * 3 + 2])) / 3.0f;
        
        // Compute angle factor (angles already in radians)
        float angle_factor = cosf(angle_avg);
        
        // Apply soft transformation
        activations[idx] *= (0.5f + 0.5f * angle_factor);
    }
}


/**
 * Kernel: Apply decay and sigmoid nonlinearity
 * 
 * Combined kernel for efficiency - applies decay then bounded sigmoid.
 */
__global__ void apply_decay_sigmoid_kernel(
    float* __restrict__ state,           // (batch_size, n_cells) in/out
    const float decay,
    const float state_max,
    const int batch_size,
    const int n_cells
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_cells;
    
    if (idx < total) {
        // Apply decay
        float val = state[idx] * decay;
        
        // Apply sigmoid: sigmoid(x * 2 - 1) * max_val
        float sigmoid_input = val * 2.0f - 1.0f;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_input));
        
        float max_val = fmaxf(state_max, 0.1f);
        state[idx] = sigmoid_val * max_val;
    }
}


/**
 * Kernel: Tiled matrix multiplication for state propagation
 * 
 * Computes: output = state @ propagation^T
 * Optimized with shared memory tiling.
 */
__global__ void propagate_state_kernel(
    const float* __restrict__ state,        // (batch_size, n_cells)
    const float* __restrict__ propagation,  // (n_cells, n_cells)
    float* __restrict__ output,             // (batch_size, n_cells)
    const int batch_size,
    const int n_cells
) {
    __shared__ float tile_state[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_prop[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    const int n_tiles = (n_cells + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < n_tiles; t++) {
        // Load tiles into shared memory
        int state_col = t * TILE_SIZE + threadIdx.x;
        int prop_row = t * TILE_SIZE + threadIdx.y;
        
        if (row < batch_size && state_col < n_cells) {
            tile_state[threadIdx.y][threadIdx.x] = state[row * n_cells + state_col];
        } else {
            tile_state[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load propagation^T (swap indices for transpose)
        if (prop_row < n_cells && col < n_cells) {
            tile_prop[threadIdx.y][threadIdx.x] = propagation[col * n_cells + prop_row];
        } else {
            tile_prop[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_state[threadIdx.y][k] * tile_prop[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < batch_size && col < n_cells) {
        output[row * n_cells + col] = sum;
    }
}


/**
 * Kernel: Fused propagation step
 * 
 * Combines propagation, bounce transform, decay, and sigmoid in one kernel
 * for maximum efficiency. Uses shared memory for the propagation matrix.
 */
__global__ void fused_propagation_step_kernel(
    const float* __restrict__ state_in,      // (batch_size, n_cells)
    float* __restrict__ state_out,           // (batch_size, n_cells)
    const float* __restrict__ propagation,   // (n_cells, n_cells)
    const float* __restrict__ bounce_angles, // (n_cells, 3)
    const float decay,
    const float state_max,
    const int batch_size,
    const int n_cells
) {
    extern __shared__ float shared_mem[];
    
    const int batch_idx = blockIdx.y;
    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && cell_idx < n_cells) {
        // Compute propagated value (dot product with propagation column)
        float propagated = 0.0f;
        for (int j = 0; j < n_cells; j++) {
            propagated += state_in[batch_idx * n_cells + j] * propagation[j * n_cells + cell_idx];
        }
        
        // Apply bounce transform (angles already in radians, max 2 rad each)
        float angle_avg = (fabsf(bounce_angles[cell_idx * 3 + 0]) +
                          fabsf(bounce_angles[cell_idx * 3 + 1]) +
                          fabsf(bounce_angles[cell_idx * 3 + 2])) / 3.0f;
        float angle_factor = cosf(angle_avg);
        propagated *= (0.5f + 0.5f * angle_factor);
        
        // Apply decay
        propagated *= decay;
        
        // Apply sigmoid
        float sigmoid_input = propagated * 2.0f - 1.0f;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_input));
        float max_val = fmaxf(state_max, 0.1f);
        
        state_out[batch_idx * n_cells + cell_idx] = sigmoid_val * max_val;
    }
}


/**
 * Kernel: Compute weighted sum of history states
 * 
 * Combines multiple propagation states with learned weights.
 */
__global__ void weighted_history_sum_kernel(
    const float* __restrict__ history,    // (n_steps, batch_size, n_cells)
    const float* __restrict__ weights,    // (n_steps,) - normalized weights
    float* __restrict__ output,           // (batch_size, n_cells)
    const int n_steps,
    const int batch_size,
    const int n_cells
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_cells;
    
    if (idx < total) {
        float sum = 0.0f;
        for (int step = 0; step < n_steps; step++) {
            sum += history[step * total + idx] * weights[step];
        }
        output[idx] = sum;
    }
}


// ============================================================================
// C++ Wrapper Functions
// ============================================================================

torch::Tensor compute_propagation_weights_cuda(
    torch::Tensor std_devs,
    torch::Tensor adjacency,
    float threshold
) {
    const int n_cells = std_devs.size(0);
    auto propagation = torch::zeros({n_cells, n_cells}, std_devs.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n_cells + threads - 1) / threads;
    
    compute_propagation_weights_kernel<<<blocks, threads>>>(
        std_devs.data_ptr<float>(),
        adjacency.data_ptr<float>(),
        propagation.data_ptr<float>(),
        n_cells,
        threshold
    );
    
    return propagation;
}


torch::Tensor apply_bounce_transform_cuda(
    torch::Tensor activations,
    torch::Tensor bounce_angles
) {
    const int batch_size = activations.size(0);
    const int n_cells = activations.size(1);
    
    auto output = activations.clone();
    
    const int total = batch_size * n_cells;
    const int threads = BLOCK_SIZE;
    const int blocks = (total + threads - 1) / threads;
    
    apply_bounce_transform_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bounce_angles.data_ptr<float>(),
        batch_size,
        n_cells
    );
    
    return output;
}


torch::Tensor apply_decay_sigmoid_cuda(
    torch::Tensor state,
    float decay,
    float state_max
) {
    const int batch_size = state.size(0);
    const int n_cells = state.size(1);
    
    auto output = state.clone();
    
    const int total = batch_size * n_cells;
    const int threads = BLOCK_SIZE;
    const int blocks = (total + threads - 1) / threads;
    
    apply_decay_sigmoid_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        decay,
        state_max,
        batch_size,
        n_cells
    );
    
    return output;
}


torch::Tensor propagate_state_cuda(
    torch::Tensor state,
    torch::Tensor propagation
) {
    const int batch_size = state.size(0);
    const int n_cells = state.size(1);
    
    auto output = torch::zeros_like(state);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (n_cells + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    
    propagate_state_kernel<<<blocks, threads>>>(
        state.data_ptr<float>(),
        propagation.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        n_cells
    );
    
    return output;
}


torch::Tensor fused_propagation_step_cuda(
    torch::Tensor state_in,
    torch::Tensor propagation,
    torch::Tensor bounce_angles,
    float decay,
    float state_max
) {
    const int batch_size = state_in.size(0);
    const int n_cells = state_in.size(1);
    
    auto state_out = torch::zeros_like(state_in);
    
    const int threads = BLOCK_SIZE;
    dim3 blocks((n_cells + threads - 1) / threads, batch_size);
    
    fused_propagation_step_kernel<<<blocks, threads>>>(
        state_in.data_ptr<float>(),
        state_out.data_ptr<float>(),
        propagation.data_ptr<float>(),
        bounce_angles.data_ptr<float>(),
        decay,
        state_max,
        batch_size,
        n_cells
    );
    
    return state_out;
}


torch::Tensor weighted_history_sum_cuda(
    torch::Tensor history,
    torch::Tensor weights
) {
    const int n_steps = history.size(0);
    const int batch_size = history.size(1);
    const int n_cells = history.size(2);
    
    auto output = torch::zeros({batch_size, n_cells}, history.options());
    
    const int total = batch_size * n_cells;
    const int threads = BLOCK_SIZE;
    const int blocks = (total + threads - 1) / threads;
    
    weighted_history_sum_kernel<<<blocks, threads>>>(
        history.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        n_steps,
        batch_size,
        n_cells
    );
    
    return output;
}


// ============================================================================
// PyTorch Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_propagation_weights", &compute_propagation_weights_cuda,
          "Compute propagation weights (CUDA)");
    m.def("apply_bounce_transform", &apply_bounce_transform_cuda,
          "Apply bounce transformation (CUDA)");
    m.def("apply_decay_sigmoid", &apply_decay_sigmoid_cuda,
          "Apply decay and sigmoid (CUDA)");
    m.def("propagate_state", &propagate_state_cuda,
          "Propagate state through lattice (CUDA)");
    m.def("fused_propagation_step", &fused_propagation_step_cuda,
          "Fused propagation step (CUDA)");
    m.def("weighted_history_sum", &weighted_history_sum_cuda,
          "Weighted sum of history states (CUDA)");
}
