#include "custom-op.hpp"

namespace bevfusion{
namespace custom
{

// Create a CUDA kernel to set the last feature of each point to 0
static __global__ void set_last_feature_to_zero_cuda(float* points, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        points[idx * 5 + 4] = 0.0f;
    }
}

void set_last_feature_to_zero(nv::Tensor lidar_points, void* stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    // Get the number of points (N)
    int num_points = lidar_points.size(0);
    int threads_per_block = 256;
    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
    set_last_feature_to_zero_cuda<<<num_blocks, threads_per_block>>>(lidar_points.ptr<float>(), num_points);
    // Ensure the kernel execution is complete
    cudaStreamSynchronize(_stream);
}

} // namespace custom
} // bevfusion