
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 512
#define THREADS_PER_BLOCK 256

// CUDA kernel to compute partial dot products
__global__ void dotProductKernel(const half *a, const half *b, float *partial_results) {
    __shared__ float shared_data[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    if (tid < N) {
        shared_data[lane] = __half2float(a[tid]) * __half2float(b[tid]);
    } else {
        shared_data[lane] = 0.0f;
    }

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

// CUDA kernel to perform final reduction of partial results
__global__ void finalReductionKernel(float *partial_results, int num_blocks) {
    __shared__ float shared_data[THREADS_PER_BLOCK];
    int lane = threadIdx.x;

    shared_data[lane] = (lane < num_blocks) ? partial_results[lane] : 0.0f;

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared_data[lane] += shared_data[lane + stride];
        }
        __syncthreads();
    }

    // Write the final result to the first element of partial_results
    if (lane == 0) {
        partial_results[0] = shared_data[0];
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
    cudaMalloc((void**)&d_partial_results, (N / THREADS_PER_BLOCK) * sizeof(float));


    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel to compute partial dot products
    dotProductKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial_results);

    // Launch kernel to perform final reduction of partial results
    finalReductionKernel<<<1, THREADS_PER_BLOCK>>>(d_partial_results, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Copy the final result back to host
    cudaMemcpy(&h_result, d_partial_results, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Dot Product: " << h_result << std::endl;

    // Free memory
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_results);

    return 0;
}
