/*
 * Python Bindings for Fused Radial Attention Kernel
 *
 * Provides PyTorch-compatible interface using pybind11.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of launcher
extern "C" void launch_fused_radial_attention(
    const void* Q,
    const void* K,
    const void* V,
    const int* landmark_indices,
    const int* radii,
    void* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int num_landmarks,
    int max_radius,
    cudaStream_t stream
);


/*
 * PyTorch wrapper for fused radial attention
 *
 * Args:
 *   Q: [batch, num_heads, seq_len, head_dim] query tensor (FP16)
 *   K: [batch, num_heads, seq_len, head_dim] key tensor (FP16)
 *   V: [batch, num_heads, seq_len, head_dim] value tensor (FP16)
 *   landmark_indices: [seq_len, num_landmarks] landmark positions (int32)
 *   radii: [seq_len] radial distance per query (int32)
 *
 * Returns:
 *   output: [batch, num_heads, seq_len, head_dim] attention output (FP16)
 */
torch::Tensor fused_radial_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor landmark_indices,
    torch::Tensor radii
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(landmark_indices.is_cuda(), "landmark_indices must be a CUDA tensor");
    TORCH_CHECK(radii.is_cuda(), "radii must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    TORCH_CHECK(landmark_indices.dtype() == torch::kInt32, "landmark_indices must be int32");
    TORCH_CHECK(radii.dtype() == torch::kInt32, "radii must be int32");

    TORCH_CHECK(Q.dim() == 4, "Q must be 4D");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must have same shape as Q");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must have same shape as Q");

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    const int num_landmarks = landmark_indices.size(1);

    TORCH_CHECK(landmark_indices.size(0) == seq_len, "landmark_indices must match seq_len");
    TORCH_CHECK(radii.size(0) == seq_len, "radii must match seq_len");

    // Find max radius for memory allocation
    int max_radius = radii.max().item<int>();
    max_radius = std::min(max_radius, seq_len);  // Clamp to sequence length

    // Allocate output tensor
    auto output = torch::empty_like(Q);

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_fused_radial_attention(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        landmark_indices.data_ptr<int>(),
        radii.data_ptr<int>(),
        output.data_ptr(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_landmarks,
        max_radius,
        stream
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_radial_attention_forward",
        &fused_radial_attention_forward,
        "Fused radial attention forward pass (CUDA)"
    );
}
