

SID4 = 2733, SEED = 2733, SLICE = 733, HP_ID = 3, CLS_A = 3, CLS_B = 7

## 1. Diabetes NN — baseline vs. HP_ID-modified model

| Framework | Model | Mean Accuracy | SD |
|---|---|---|---|
| PyTorch | Baseline | 0.7456 | 0.0088 |
| PyTorch | Modified | 0.7690 | 0.0051 |
| TensorFlow | Baseline | 0.7398 | 0.0101 |
| TensorFlow | Modified | 0.7661 | 0.0051 |



## 2. CUDA matrix multiplication — CPU vs. GPU

Profiler used: nvprof.

| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | Speedup |
|---|---|---|---|---|
| 256 | 20.4408 | 0.2281 | 0.4213 | 31.47x |
| 1024 | 3429.8649 | 9.2046 | 4.6925 | 246.80x |
| 4096 | 743422.8464 | 408.9250 | 87.4966 | 1497.56x |




GPU activities:  74.11%  9.1989ms  matmulKernel(float*, float*, float*, int)
                  14.02%  1.7401ms  [CUDA memcpy HtoD]
                  11.87%  1.4740ms  [CUDA memcpy DtoH]

## 3. Crossover analysis

The GPU is already faster than the CPU at the smallest tested size, N=256 (0.65 ms GPU vs 20.44 ms CPU, a 31.47x speedup). This crossover isn't at size zero because every GPU run pays fixed overhead (cudaMalloc, kernel launch, and the H2D/D2H copies) no matter how small the matrix is. At tiny sizes that overhead can dominate, but as N grows, the CPU's O(N^3) cost grows much faster than the GPU's fixed overhead, so the speedup widens sharply — from 31.47x at N=256 to 1497.56x at N=4096.