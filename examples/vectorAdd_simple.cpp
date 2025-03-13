#include <hip/hip_runtime.h>
#include <iostream>

#define N 256  // Vector size

// HIP Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
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


    // Launch HIP Kernel
    vectorAdd<<<1, 256>>>(d_A, d_B, d_C, N);

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
