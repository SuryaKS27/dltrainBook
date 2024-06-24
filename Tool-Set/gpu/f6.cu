


#include <cuda_runtime.h>
#include <iostream>

#define N 512

// CUDA kernel to compute partial dot products
__global__ void dotProductKernel(const half *a, const half *b, float *partial_results) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    shared_data[lane] = __half2float(a[tid]) * __half2float(b[tid]);

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_data[lane] += shared_data[lane + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to partial_results
    if (lane == 0) {
        partial_results[blockIdx.x] = shared_data[0];
    }
}

int main() {
    half *h_a, *h_b;
    float h_result = 0.0f;

    // Allocate and initialize host memory
    h_a = (half*)malloc(N * sizeof(half));
    h_b = (half*)malloc(N * sizeof(half));

    // Initialize vectors with some values
    for (int i = 0; i < N; ++i) {
        h_a[i] = __float2half(i * 0.5f);
        h_b[i] = __float2half(i * 0.5f);
    }

    // Allocate device memory
    half *d_a, *d_b;
    float *d_partial_results;
    cudaMalloc((void**)&d_a, N * sizeof(half));
    cudaMalloc((void**)&d_b, N * sizeof(half));
    cudaMalloc((void**)&d_partial_results, (N / 256) * sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dotProductKernel<<<N / 256, 256>>>(d_a, d_b, d_partial_results);

    // Copy partial results back to host
    float *h_partial_results = (float*)malloc((N / 256) * sizeof(float));
    cudaMemcpy(h_partial_results, d_partial_results, (N / 256) * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    for (int i = 0; i < N / 256; ++i) {
        h_result += h_partial_results[i];
    }

    // Print the result
    std::cout << "Dot Product: " << h_result << std::endl;

    // Free memory
    free(h_a);
    free(h_b);
    free(h_partial_results);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_results);

    return 0;
}

