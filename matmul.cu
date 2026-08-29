// DATA 266 - Homework 1, CUDA matrix multiplication
// SID4 = 2733, SEED = 2733
//
// BLOCKS AND THREADS:
// Each thread computes ONE output element of C.
// row = blockIdx.y * blockDim.y + threadIdx.y   -> .y = vertical position = row
// col = blockIdx.x * blockDim.x + threadIdx.x   -> .x = horizontal position = column
// Threads are grouped into 16x16 blocks (256 threads/block).
// Grid size = (N/16, N/16) blocks, so every element of the NxN output
// is covered by exactly one thread.#include <cstdio>

#include <cstdlib>
#include <cmath>
#include <chrono>

#define SEED 2733

__global__ void matmulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmulCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void runOneSize(int N) {
    int size = N * N;
    float *A = (float*)malloc(size * sizeof(float));
    float *B = (float*)malloc(size * sizeof(float));
    float *C_cpu = (float*)malloc(size * sizeof(float));
    float *C_gpu = (float*)malloc(size * sizeof(float));

    srand(SEED);
    for (int i = 0; i < size; i++) { A[i] = (float)rand()/RAND_MAX; B[i] = (float)rand()/RAND_MAX; }

    auto cpuStart = std::chrono::high_resolution_clock::now();
    matmulCPU(A, B, C_cpu, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size * sizeof(float));
    cudaMalloc(&dB, size * sizeof(float));
    cudaMalloc(&dC, size * sizeof(float));

    cudaEvent_t t1, t2, t3, t4;
    cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3); cudaEventCreate(&t4);

    cudaEventRecord(t1);
    cudaMemcpy(dA, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(t2);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 16, N / 16);
    matmulKernel<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    cudaEventRecord(t3);

    cudaMemcpy(C_gpu, dC, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(t4);
    cudaEventSynchronize(t4);

    float h2dMs, kernelMs, d2hMs;
    cudaEventElapsedTime(&h2dMs, t1, t2);
    cudaEventElapsedTime(&kernelMs, t2, t3);
    cudaEventElapsedTime(&d2hMs, t3, t4);

    float maxDiff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    printf("\n=== N = %d ===\n", N);
    printf("CPU time:        %.4f ms\n", cpuMs);
    printf("GPU kernel time: %.4f ms\n", kernelMs);
    printf("H2D+D2H time:    %.4f ms\n", h2dMs + d2hMs);
    printf("End-to-end GPU:  %.4f ms\n", kernelMs + h2dMs + d2hMs);
    printf("Speedup:         %.2fx\n", cpuMs / (kernelMs + h2dMs + d2hMs));
    printf("Max CPU/GPU diff: %e\n", maxDiff);

    free(A); free(B); free(C_cpu); free(C_gpu);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    int sizes[] = {256, 1024, 4096};
    for (int i = 0; i < 3; i++) {
        runOneSize(sizes[i]);
    }
    return 0;
}
