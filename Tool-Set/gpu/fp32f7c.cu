
#include <stdio.h>

#define N 512
#define BLOCK_SIZE 256

__global__ void dot_product_float32(float *a, float *b, float *result) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    temp[tid] = (index < N) ? a[index] * b[index] : 0.0f;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Store result of this block in global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

int main() {
    float *a, *b, *result;
    float *d_a, *d_b, *d_result;
    float final_result;

    // Allocate host memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    result = (float*)malloc(sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 1.0f;
    }
    *result = 0.0f;

    // Allocate device memory
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dot_product_float32<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_result);

    // Copy result back to host
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Print result
    printf("Dot product (float32): %f\n", *result);

    // Free host memory
    free(a);
    free(b);
    free(result);

    return 0;
}

