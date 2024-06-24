
#include <stdio.h>

// CUDA kernel to compute dot product of two vectors
__global__ void dotProduct(float *a, float *b, float *result, int N) {
    // Allocate shared memory for partial sums
    __shared__ float partialSum[512];

    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Perform dot product within each block
    partialSum[threadIdx.x] = a[tid] * b[tid];

    // Synchronize threads within block
    __syncthreads();

    // Perform reduction to compute the final dot product
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store the result of this block's dot product in global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, partialSum[0]);
    }
}

int main() {
    // Initialize host vectors and variables
    float *a, *b, *result;
    float *d_a, *d_b, *d_result;
    int N = 512;
    float sum = 0.0f;

    // Allocate memory on host
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    result = (float*)malloc(sizeof(float));

    // Initialize vectors 'a' and 'b'
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f; // Initialize with 1.0
        b[i] = 1.0f; // Initialize with 1.0
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &sum, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 512 threads
    dotProduct<<<1, N>>>(d_a, d_b, d_result, N);

    // Copy result back to host
    cudaMemcpy(&sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the dot product result
    printf("Dot product: %f\n", sum);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Free host memory
    free(a);
    free(b);
    free(result);


    return 0;
}
