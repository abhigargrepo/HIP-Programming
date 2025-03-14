#include <hip/hip_runtime.h>
#include <iostream>

#define N 20 // Vector size

__global__ void kernel(float *a) {
  __shared__ float sData[N];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  sData[threadIdx.x] = a[idx];

  __syncthreads();

  a[idx] = sData[blockDim.x - 1 - threadIdx.x];
}

int main() {
  // Host memory allocation
  float *h_A, *h_B;
  h_A = new float[N];
  h_B = new float[N];

  // Initialize host data
  for (int i = 0; i < N; i++)
    h_A[i] = i * 1.0f;

  // Device memory allocation
  float *d_A;
  hipMalloc((void **)&d_A, N * sizeof(float));

  // Copy data from Host to Device
  hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);

  // Launch HIP Kernel
  kernel<<<4, 5>>>(d_A);

  // Copy result from Device to Host
  hipMemcpy(h_B, d_A, N * sizeof(float), hipMemcpyDeviceToHost);

  // Print result
  std::cout << "Original Array:" << std::endl;
  for(int i=0; i<N; i++)
    std::cout << h_A[i] << " ";
  std::cout << std::endl;
  std::cout << "Output Array:" << std::endl;

  for(int i=0; i<N; i++)
    std::cout << h_B[i] << " ";
  std::cout << std::endl;
  
  // Free device memory
  hipFree(d_A);

  // Free host memory
  delete[] h_A;
  delete[] h_B;

  return 0;
}
