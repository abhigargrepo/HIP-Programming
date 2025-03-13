#include <hip/hip_runtime.h>
#include <iostream>

#define N 1024*1024  // Vector size

// HIP Kernel for vector addition
template <typename T>
__global__ void vectorAdd(const T *A, const T *B, T *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Host memory allocation
    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    hipMalloc((void **)&d_A, N * sizeof(float));
    hipMalloc((void **)&d_B, N * sizeof(float));
    hipMalloc((void **)&d_C, N * sizeof(float));

    // Copy data from Host to Device
    hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim = dim3(256,1,1);
    dim3 gridDim  = dim3((N-1+blockDim.x)/blockDim.x,1,1);

    // Launch HIP Kernel
    vectorAdd<float><<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result from Device to Host
    hipMemcpy(h_C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    // Print some results
    std::cout << "Sample output: " << h_C[0] << ", " << h_C[N/2] << ", " << h_C[N-1] << std::endl;

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
